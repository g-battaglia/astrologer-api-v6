"""
Advanced endpoints - Kerykeion v6 features.

Endpoints for eclipses, planetary phenomena, planetary nodes, heliacal events,
occultations, relocated charts, fixed star discovery, primary directions,
astro-cartography, and declination aspects.

All endpoints under /api/v6/advanced/*.
"""

import asyncio
import inspect
import json
from datetime import datetime
from functools import lru_cache
from logging import getLogger
from typing import Any, Callable, cast
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from kerykeion import (
    AspectsFactory,
    ChartDataFactory,
    CompositeSubjectFactory,
    EclipseFactory,
    EphemerisDataFactory,
    FixedStarDiscoveryFactory,
    LunationFinderFactory,
    MidpointFactory,
    MundaneAspectFactory,
    PlanetaryNodesFactory,
    PlanetaryPhenomenaFactory,
    PrimaryDirectionsFactory,
    AstroCartographyFactory,
    RelocatedChartFactory,
    ReportGenerator,
    RetrogradeStationFactory,
    SecondaryProgressionFactory,
    SignIngressFactory,
    SolarArcFactory,
    TransitsTimeRangeFactory,
    ZodiacalReleasingFactory,
    ProfectionsFactory,
    FirdariaFactory,
    HoraryIndicatorsFactory,
)
from kerykeion import PTOLEMAIC_ASPECTS
from kerykeion.schemas import AstrologicalSubjectModel


def _predictive_active_aspects(aspect_orb: float, aspects: "list[str] | None") -> "list[dict]":
    """Build an active-aspects list for a predictive chart wheel from the
    scalar ``aspect_orb`` + optional aspect-name whitelist, so the rendered
    biwheel SVG and aspect grid stay consistent with the cross-aspect table
    computed by the matching ``/advanced/...`` endpoint."""
    names = list(aspects) if aspects is not None else list(PTOLEMAIC_ASPECTS)
    return [{"name": name, "orb": aspect_orb} for name in names]


from ..types.request_models import (
    DEFAULT_ACTIVE_POINTS,
    EPHEMERIS_MAX_POINT_CALCULATIONS,
    EPHEMERIS_MAX_SAMPLES,
    FIXED_STAR_WORK_UNIT_WEIGHT,
    _count_ephemeris_samples,
    AstroCartographyRequestModel,
    DeclinationAspectsRequestModel,
    DualDeclinationAspectsRequestModel,
    EclipseSearchRequestModel,
    EphemerisRequestModel,
    FixedStarDiscoveryRequestModel,
    HeliacalEventsRequestModel,
    HeliocentricReturnRequestModel,
    HeliocentricReturnDataRequestModel,
    LunarNodeCrossingRequestModel,
    LunarNodeCrossingDataRequestModel,
    LunationsRequestModel,
    MidpointsRequestModel,
    MundaneAspectsRequestModel,
    OccultationSearchRequestModel,
    PlanetaryNodesRequestModel,
    PlanetaryPhenomenaRequestModel,
    PrimaryDirectionsRequestModel,
    RelocatedChartRequestModel,
    ReportRequestModel,
    RetrogradeStationsRequestModel,
    SecondaryProgressionsRequestModel,
    SignIngressesRequestModel,
    SolarArcDirectionsRequestModel,
    TransitEventsRequestModel,
    ZodiacalReleasingRequestModel,
    ProfectionsRequestModel,
    FirdariaRequestModel,
    HoraryIndicatorsRequestModel,
)
from ..types.response_models import (
    AstroCartographyResponseModel,
    DeclinationAspectsResponseModel,
    EclipseSearchResponseModel,
    EphemerisResponseModel,
    LunationsResponseModel,
    FixedStarDiscoveryResponseModel,
    HeliacalEventsResponseModel,
    HeliocentricReturnChartResponseModel,
    LunarNodeCrossingChartResponseModel,
    MidpointsResponseModel,
    MundaneAspectsResponseModel,
    OccultationSearchResponseModel,
    PlanetaryNodesResponseModel,
    PlanetaryPhenomenaResponseModel,
    PrimaryDirectionsResponseModel,
    RelocatedChartResponseModel,
    ReportResponseModel,
    ChartDataResponseModel,
    ProgressionChartResponseModel,
    RetrogradeStationsResponseModel,
    SecondaryProgressionsResponseModel,
    SignIngressesResponseModel,
    SolarArcDirectionsResponseModel,
    TransitEventsResponseModel,
    TransitMomentsResponseModel,
    ZodiacalReleasingResponseModel,
    ProfectionsResponseModel,
    FirdariaResponseModel,
    HoraryIndicatorsResponseModel,
)
from ..utils.router_utils import (
    EXTENDED_RENDERING_FIELDS,
    RENDERING_FIELD_DEFAULTS,
    build_subject,
    calculate_heliocentric_return_chart_data,
    calculate_lunar_node_crossing_chart_data,
    chart_data_payload,
    chart_payload_from_request,
    dump,
    handle_exception,
    resolve_active_points,
    resolve_active_aspects,
    run_heavy,
)
from ..utils.logging_utils import log_request_with_body
from ..utils.occultation import (
    HELIACAL_SEARCH_TIMEOUT_S,
    OCCULTATION_PLANET_IDS,
    OCCULTATION_SEARCH_TIMEOUT_S,
    run_heliacal_search,
    run_occultation_search,
)
from ..utils.subject_kwargs import subject_factory_kwargs
from ..config.settings import settings

logger = getLogger(__name__)
router = APIRouter()


def _yearly_chunks(start_date: str, end_date: str) -> "list[tuple[str, str]]":
    """Split ``[start_date, end_date]`` into <=1-year (start, end) string pairs.

    The original endpoints are kept at the extremes so the factory's own
    date-only / datetime semantics are preserved; interior bounds are explicit
    datetimes, so chunks abut without gaps or overlaps.
    """
    from datetime import datetime, timedelta, timezone

    def naive(s: str) -> datetime:
        d = datetime.fromisoformat(s)
        if d.tzinfo is not None:
            d = d.astimezone(timezone.utc).replace(tzinfo=None)
        return d

    start = naive(start_date)
    end = naive(end_date)
    if "T" not in end_date and "t" not in end_date and " " not in end_date:
        end = end.replace(hour=23, minute=59, second=59, microsecond=999999)
    if end <= start:
        return [(start_date, end_date)]

    edges: "list[tuple[datetime, datetime]]" = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=365), end)
        edges.append((cur, nxt))
        cur = nxt

    last = len(edges) - 1
    return [
        (
            start_date if i == 0 else a.isoformat(),
            end_date if i == last else b.isoformat(),
        )
        for i, (a, b) in enumerate(edges)
    ]


async def _scan_chunked(factory, attr: str, start_date: str, end_date: str, planets):
    """Run ``factory.from_iso_range`` over yearly chunks in a worker thread,
    yielding between chunks so the process-wide EPHEMERIS_LOCK is released and the
    event loop (and /health) stays responsive during a long scan. Concatenated
    results stay chronologically ordered (chunks are sequential, each ordered).

    Chunks go through ``run_heavy`` so the scan competes for the same bounded
    pool as every other heavy computation — ``asyncio.to_thread`` would fan
    concurrent scans out onto the unbounded default executor, bypassing the
    HEAVY_MAX_WORKERS memory cap."""
    items: list = []
    for chunk_start, chunk_end in _yearly_chunks(start_date, end_date):
        part = await run_heavy(factory.from_iso_range, chunk_start, chunk_end, planets)
        items.extend(getattr(part, attr))
        await asyncio.sleep(0)
    return items


def _render_progression_chart_payload(chart_data, request_body) -> dict:
    """Render the progression / solar-arc biwheel SVGs + data payload (CPU-heavy,
    call via ``run_heavy``)."""
    from kerykeion import ChartDrawer

    # Same omit-if-default rule as router_utils.render_chart: the pinned
    # kerykeion does not know the extended options, and an unknown keyword raises
    # TypeError in ChartDrawer.__init__ — a 500 on every request, not just the
    # ones that ask for an option.
    #
    # Omit-if-default, by value rather than by truthiness: EXTENDED_RENDERING_FIELDS
    # is no longer all-boolean (glyph_size is a three-valued enum whose default,
    # "medium", is truthy).
    extended = {name: value for name in EXTENDED_RENDERING_FIELDS if (value := getattr(request_body, name, RENDERING_FIELD_DEFAULTS[name])) != RENDERING_FIELD_DEFAULTS[name]}

    drawer = ChartDrawer(
        chart_data,
        theme=request_body.theme or "classic",
        transparent_background=request_body.transparent_background,
        style=getattr(request_body, "style", "classic"),
        show_zodiac_background_ring=getattr(request_body, "show_zodiac_background_ring", True),
        **extended,
    )
    # remove_css_variables is left False to match the main charts router
    # (router_utils.render_chart), so downstream CSS-variable theming applies
    # uniformly to every chart endpoint.
    return {
        "status": "OK",
        "chart_wheel": drawer.generate_wheel_only_svg_string(minify=True),
        "chart_grid": drawer.generate_aspect_grid_only_svg_string(minify=True),
        "chart_data": dump(chart_data),
    }


# ===========================================================================
# Eclipse Search
# ===========================================================================


@router.post("/api/v6/advanced/eclipses", response_model=EclipseSearchResponseModel)
async def eclipse_search(request_body: EclipseSearchRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/eclipses`

    Search for upcoming solar and lunar eclipses, either globally or for a specific location.

    **Parameters:**
    - `latitude`, `longitude`: Location for local search (omit both for global).
    - `start_year`: Year to start searching from (default 2025).
    - `count`: Number of eclipses of each type to find (default 5): `count=2`
      returns up to 2 solar plus 2 lunar eclipses.

    **Returns:**
    - `solar_eclipses`: List of solar eclipse events.
    - `lunar_eclipses`: List of lunar eclipse events.
    - `latitude`, `longitude`: Echo of the request location (null on global searches).
    """
    log_request_with_body(logger, request, "Eclipse search request", request_body.model_dump_json())

    try:
        if request_body.latitude is not None and request_body.longitude is not None:
            result = await run_heavy(
                EclipseFactory.search_from_location,
                lat=request_body.latitude,
                lng=request_body.longitude,
                start_year=request_body.start_year,
                count=request_body.count,
            )
        else:
            result = await run_heavy(
                EclipseFactory.search_global,
                start_year=request_body.start_year,
                count=request_body.count,
            )

        return JSONResponse(
            content={
                "status": "OK",
                "solar_eclipses": dump(result.solar_eclipses),
                "lunar_eclipses": dump(result.lunar_eclipses),
                "latitude": getattr(result, "latitude", request_body.latitude),
                "longitude": getattr(result, "longitude", request_body.longitude),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Lunations
# ===========================================================================


@router.post("/api/v6/advanced/lunations", response_model=LunationsResponseModel)
async def lunations(request_body: LunationsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/lunations`

    Find lunations (New Moon, First Quarter, Full Moon, Last Quarter) within a
    date range, ordered chronologically, each with the Sun and Moon zodiac
    positions at the exact phase.

    **Parameters:**
    - `start_date`, `end_date`: ISO date(time) range (treated as UTC).
    - `phases`: Optional subset of `new`/`first_quarter`/`full`/`last_quarter`.

    **Returns:**
    - `lunations`: Ordered list of lunation events.
    """
    log_request_with_body(logger, request, "Lunations request", request_body.model_dump_json())

    try:
        result = await run_heavy(
            LunationFinderFactory.from_iso_range,
            start_date=request_body.start_date,
            end_date=request_body.end_date,
            phases=request_body.phases,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "start_jd": result.start_jd,
                "end_jd": result.end_jd,
                "lunations": dump(result.lunations),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Retrograde Stations
# ===========================================================================


@router.post("/api/v6/advanced/retrograde-stations", response_model=RetrogradeStationsResponseModel)
async def retrograde_stations(request_body: RetrogradeStationsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/retrograde-stations`

    Find planetary retrograde/direct stations (motion reversals) within a date
    range, ordered chronologically, each with the zodiac position at the station.

    **Parameters:**
    - `start_date`, `end_date`: ISO date(time) range (treated as UTC).
    - `planets`: Optional subset of Mercury..Pluto (the Sun and Moon never station).

    **Returns:**
    - `stations`: Ordered list of stations (SR = retrograde, SD = direct).
    """
    log_request_with_body(logger, request, "Retrograde stations request", request_body.model_dump_json())

    try:
        # Scan in yearly chunks on a worker thread, releasing EPHEMERIS_LOCK
        # between chunks so the single-worker event loop stays responsive.
        stations = await _scan_chunked(
            RetrogradeStationFactory,
            "stations",
            request_body.start_date,
            request_body.end_date,
            request_body.planets,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "stations": dump(stations),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Sign Ingresses
# ===========================================================================


@router.post("/api/v6/advanced/sign-ingresses", response_model=SignIngressesResponseModel)
async def sign_ingresses(request_body: SignIngressesRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/sign-ingresses`

    Find zodiac sign ingresses (30 degree boundary crossings) within a date
    range, ordered chronologically. Retrograde re-entries are included.

    **Parameters:**
    - `start_date`, `end_date`: ISO date(time) range (treated as UTC).
    - `planets`: Optional subset of Sun..Pluto plus Moon (Moon is opt-in).

    **Returns:**
    - `ingresses`: Ordered list of sign ingresses with from/to signs.
    """
    log_request_with_body(logger, request, "Sign ingresses request", request_body.model_dump_json())

    try:
        # Scan in yearly chunks on a worker thread, releasing EPHEMERIS_LOCK
        # between chunks so the single-worker event loop stays responsive.
        ingresses = await _scan_chunked(
            SignIngressFactory,
            "ingresses",
            request_body.start_date,
            request_body.end_date,
            request_body.planets,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "ingresses": dump(ingresses),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Mundane Aspectarian
# ===========================================================================


@router.post("/api/v6/advanced/mundane-aspects", response_model=MundaneAspectsResponseModel)
async def mundane_aspects(request_body: MundaneAspectsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/mundane-aspects`

    Find every exact mundane (transiting-to-transiting) aspect within a date
    range — the *aspectarian* of a printed astrological calendar — ordered
    chronologically, each with its exact UTC instant, both bodies' zodiac
    positions and retrograde flags.

    **Parameters:**
    - `start_date`, `end_date`: ISO date(time) range (treated as UTC, max ~1 year).
    - `points`: Optional body subset. Defaults to Sun..Pluto; the Moon is opt-in.
    - `aspects`: Optional aspect names. Defaults to the five Ptolemaic majors.
    - `zodiac_type`, `sidereal_mode`: Zodiac for the reported longitudes/signs
      (aspect instants are zodiac-independent).

    **Returns:**
    - `aspects`: Ordered list of exact aspects.
    """
    log_request_with_body(logger, request, "Mundane aspects request", request_body.model_dump_json())

    try:
        result = await run_heavy(
            MundaneAspectFactory.from_iso_range,
            request_body.start_date,
            request_body.end_date,
            request_body.points,
            request_body.aspects,
            request_body.zodiac_type,
            request_body.sidereal_mode,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "aspects": dump(result.aspects),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Planetary Phenomena
# ===========================================================================


@router.post("/api/v6/advanced/planetary-phenomena", response_model=PlanetaryPhenomenaResponseModel)
async def planetary_phenomena(request_body: PlanetaryPhenomenaRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/planetary-phenomena`

    Compute observational phenomena (phase angle, elongation, magnitude, morning/evening star)
    for planets at a given moment.

    Each entry also carries its `solar_phase` — the traditional classification of a planet's
    proximity to the Sun: `cazimi` (within 17 arcminutes), `combust` (within 8°30'),
    `under_the_beams` (within 17°) or `free`.

    **Parameters:**
    - `subject`: Subject defining the calculation moment.
    - `planets`: Optional list of planet names to filter.
    - `solar_phase_thresholds`: Optional override of the cazimi / combust / under-the-beams band half-widths.

    **Returns:**
    - `phenomena`: List of planetary phenomena data, each carrying its `solar_phase`.
    - `solar_phase_thresholds`: The half-widths that produced those classifications.
    """
    log_request_with_body(logger, request, "Planetary phenomena request", request_body.model_dump_json())

    try:
        subject = await run_heavy(build_subject, request_body.subject)
        # Held back unless the caller sent it: an engine that predates the
        # keyword answers an unknown kwarg with a TypeError, i.e. a 500 on the
        # whole endpoint rather than an ignored option.
        extra_kwargs = {}
        if request_body.solar_phase_thresholds is not None:
            from kerykeion.schemas import SolarPhaseThresholdsModel

            extra_kwargs["solar_phase_thresholds"] = SolarPhaseThresholdsModel(**request_body.solar_phase_thresholds.model_dump())
        result = await run_heavy(
            PlanetaryPhenomenaFactory.from_subject,
            subject,
            planets=request_body.planets,
            **extra_kwargs,
        )

        thresholds = getattr(result, "solar_phase_thresholds", None)
        return JSONResponse(
            content={
                "status": "OK",
                "iso_datetime": getattr(result, "iso_datetime", None),
                "julian_day": getattr(result, "julian_day", None),
                "phenomena": dump(result.phenomena),
                "solar_phase_thresholds": dump(thresholds) if thresholds is not None else None,
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Planetary Nodes
# ===========================================================================


@router.post("/api/v6/advanced/planetary-nodes", response_model=PlanetaryNodesResponseModel)
async def planetary_nodes(request_body: PlanetaryNodesRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/planetary-nodes`

    Compute ascending/descending nodes and perihelion/aphelion for planets.

    Each entry also carries `periapsis`, `apoapsis` and `apsis_kind` — the apsides named for the
    body the orbit turns around: `heliocentric` for a planet (repeating perihelion/aphelion),
    `geocentric` for the Moon (perigee and apogee, the latter being the Black Moon Lilith point).

    **Parameters:**
    - `subject`: Subject defining the calculation moment.
    - `method`: 'mean' or 'osculating' (default 'mean').
    - `planets`: Optional list of planet names to filter.

    **Returns:**
    - `nodes`: List of planetary node data, each with its `periapsis`, `apoapsis` and `apsis_kind`.
    """
    log_request_with_body(logger, request, "Planetary nodes request", request_body.model_dump_json())

    try:
        subject = await run_heavy(build_subject, request_body.subject)
        result = await run_heavy(
            PlanetaryNodesFactory.from_subject,
            subject,
            method=request_body.method,
            planets=request_body.planets,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "iso_datetime": getattr(result, "iso_datetime", None),
                "julian_day": getattr(result, "julian_day", None),
                "method": request_body.method,
                "nodes": dump(result.nodes),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Heliacal Events
# ===========================================================================


@router.post("/api/v6/advanced/heliacal-events", response_model=HeliacalEventsResponseModel)
async def heliacal_events(request_body: HeliacalEventsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/heliacal-events`

    Search for heliacal rising/setting events (first/last visibility) for planets and stars.

    **Parameters:**
    - `subject`: Subject defining the starting moment and observation location.
    - `count`: Number of events to return (default 5, max 20; `count` times the
      number of event types may not exceed 40).
    - `planets`: Optional list of planet/star names (max 12).
    - `event_types`: Optional subset of `heliacal_rising`, `heliacal_setting`,
      `evening_first`, `morning_last` (default: rising + setting; the last two
      apply only to Mercury and Venus).

    **Returns:**
    - `events`: List of heliacal events.
    """
    log_request_with_body(logger, request, "Heliacal events request", request_body.model_dump_json())

    try:
        event_types = None
        if request_body.event_types:
            label_to_int = {
                "heliacal_rising": 1,
                "heliacal_setting": 2,
                "evening_first": 3,
                "morning_last": 4,
            }
            event_types = [label_to_int[e] for e in request_body.event_types]
        # Native heliacal searches cannot be interrupted in a thread. The full
        # subject-build + search therefore runs under the same bounded spawn
        # budget as occultations and is hard-killed at the advertised ceiling.
        events = await run_heliacal_search(
            timeout=HELIACAL_SEARCH_TIMEOUT_S,
            subject_kwargs=subject_factory_kwargs(request_body.subject),
            count=request_body.count,
            planets=request_body.planets,
            event_types=event_types,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "events": dump(events),
            },
            status_code=200,
        )

    except TimeoutError:
        return JSONResponse(
            content={
                "status": "ERROR",
                "message": "Heliacal search timed out. Reduce planets, count, or event_types.",
                "error_type": "TimeoutError",
            },
            status_code=504,
        )
    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Occultations
# ===========================================================================


@router.post("/api/v6/advanced/occultations", response_model=OccultationSearchResponseModel)
async def occultation_search(request_body: OccultationSearchRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/occultations`

    Search for lunar occultations of a planet, visible from the observer location.

    **Parameters:**
    - `subject`: Subject defining the starting moment and observation location.
    - `planet`: Occulted body — the planet the Moon passes in front of (default 'Venus').
    - `count`: Number of events to return (default 5).

    **Returns:**
    - `events`: List of occultation events.
    """
    log_request_with_body(logger, request, "Occultation search request", request_body.model_dump_json())

    try:
        subject = await run_heavy(build_subject, request_body.subject)

        planet_id = OCCULTATION_PLANET_IDS[request_body.planet]

        events = await run_occultation_search(
            "local",
            timeout=OCCULTATION_SEARCH_TIMEOUT_S,
            julian_day=subject.julian_day,
            planet_id=planet_id,
            lat=subject.lat,
            lng=subject.lng,
            count=request_body.count,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "events": dump(events),
            },
            status_code=200,
        )

    except TimeoutError:
        return JSONResponse(
            content={
                "status": "ERROR",
                "message": "Occultation search timed out. Try reducing 'count'.",
                "error_type": "TimeoutError",
            },
            status_code=504,
        )
    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Relocated Chart
# ===========================================================================


@router.post("/api/v6/advanced/relocated-chart", response_model=RelocatedChartResponseModel)
async def relocated_chart(request_body: RelocatedChartRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/relocated-chart`

    Relocate a natal chart to a new geographic location.
    Preserves original planetary positions but recalculates houses and angles.

    **Parameters:**
    - `subject`: Original natal subject.
    - `new_latitude`, `new_longitude`: New location coordinates.
    - `new_city`, `new_nation`, `new_timezone`: Optional location metadata.
    - `active_points`: Optional list of points to calculate (defaults to the
      standard active set).

    **Returns:**
    - `subject`: Relocated astrological subject.
    """
    log_request_with_body(logger, request, "Relocated chart request", request_body.model_dump_json())

    try:
        subject = await run_heavy(
            build_subject,
            request_body.subject,
            active_points=resolve_active_points(request_body.active_points),
        )
        relocated = await run_heavy(
            RelocatedChartFactory.relocate,
            subject,
            new_lat=request_body.new_latitude,
            new_lng=request_body.new_longitude,
            new_city=request_body.new_city,
            new_nation=request_body.new_nation,
            new_tz_str=request_body.new_timezone,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "subject": dump(relocated),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Fixed Star Discovery
# ===========================================================================


@router.post("/api/v6/advanced/fixed-star-discovery", response_model=FixedStarDiscoveryResponseModel)
async def fixed_star_discovery(request_body: FixedStarDiscoveryRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/fixed-star-discovery`

    Discover prominent fixed stars in conjunction with chart points.

    **Parameters:**
    - `subject`: Subject to search for star conjunctions.
    - `orb`: Maximum orb in degrees (default 1.0).

    **Returns:**
    - `stars`: List of prominent fixed stars found.
    """
    log_request_with_body(logger, request, "Fixed star discovery request", request_body.model_dump_json())

    try:
        subject = await run_heavy(build_subject, request_body.subject)
        stars = await run_heavy(
            FixedStarDiscoveryFactory.find_prominent_stars,
            subject,
            orb=request_body.orb,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "stars": dump(stars),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Primary Directions
# ===========================================================================


@router.post("/api/v6/advanced/primary-directions", response_model=PrimaryDirectionsResponseModel)
async def primary_directions(request_body: PrimaryDirectionsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/primary-directions`

    Compute primary directions using the Placidus semi-arc method.

    **Parameters:**
    - `subject`: Natal subject.
    - `max_years`: Maximum years to project (default 100).
    - `rate_key`: 'ptolemy' or 'naibod' (default 'ptolemy').
    - `aspects`: Optional list of aspects to calculate.

    **Returns:**
    - `directions`: List of primary directions.
    - `speculum`: Speculum table with RA, declination, semi-arc data.
    """
    log_request_with_body(logger, request, "Primary directions request", request_body.model_dump_json())

    try:
        subject = await run_heavy(build_subject, request_body.subject)

        directions = await run_heavy(
            PrimaryDirectionsFactory.compute,
            subject,
            max_years=request_body.max_years,
            rate_key=request_body.rate_key,
            aspects=request_body.aspects,
        )

        speculum = await run_heavy(PrimaryDirectionsFactory.compute_speculum, subject)

        return JSONResponse(
            content={
                "status": "OK",
                "directions": dump(directions),
                "speculum": dump(speculum),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Midpoints
# ===========================================================================


@router.post("/api/v6/advanced/midpoints", response_model=MidpointsResponseModel)
async def midpoints(request_body: MidpointsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/midpoints`

    Compute the full midpoint table for a chart, with each midpoint's
    longitude on the shorter arc, sign + position-within-sign, the 90°
    dial position (`longitude % 90`, used by cosmobiology and Uranian
    astrology), and the third points that aspect the midpoint within
    `aspect_orb` degrees.

    **Parameters:**
    - `subject`: Natal/event subject.
    - `active_points`: Optional list of points to use as midpoint
      constituents. Defaults to the 14-point standard set (91 pairs).
      The table is every pair of the requested points, so its size grows
      quadratically — C(n,2). Points with no position in the requested
      perspective are excluded: `Earth` geocentrically, `Sun`
      heliocentrically.
    - `aspect_orb`: Orb in degrees for aspect-to-midpoint detection
      (default 1.0).
    - `aspects`: Optional whitelist of aspect names.
    - `compute_aspects`: If `false`, skip aspect-to-midpoint detection.

    **Returns:**
    - `midpoints`: List of midpoint entries (see `MidpointModel`).
    """
    log_request_with_body(logger, request, "Midpoints request", request_body.model_dump_json())

    try:
        # The requested constituents must also reach the subject build: points
        # outside the subject's default active set would otherwise be missing
        # from the chart and silently dropped from the midpoint table.
        subject = await run_heavy(build_subject, request_body.subject, active_points=request_body.active_points)

        result = await run_heavy(
            MidpointFactory.compute,
            subject,
            active_points=request_body.active_points,
            compute_aspects=request_body.compute_aspects,
            aspect_orb=request_body.aspect_orb,
            aspects=request_body.aspects,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "midpoints": dump(result),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


@router.post("/api/v6/advanced/zodiacal-releasing", response_model=ZodiacalReleasingResponseModel)
async def zodiacal_releasing(request_body: ZodiacalReleasingRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/zodiacal-releasing`

    Compute zodiacal releasing (aphesis) from the Part of Fortune or Spirit.
    Periods unfold from the lot's sign in zodiacal order — each sign ruling for
    its general years and subdividing into months, days and finer levels, with
    the "loosing of the bond" jump applied as the sequence circles back.

    **Parameters:**
    - `subject`: Natal subject (requires a known birth time).
    - `lot`: `fortune` or `spirit`.
    - `levels`: Subdivision levels (1-4). L1/L2 are built in full; deeper levels
      only along the target-date path.
    - `target_date`: ISO date (`YYYY-MM-DD`) used to mark the current period chain.
    - `life_cap_years`: Upper bound (in years) on how far the sequence is
      unrolled (default 100, max 120).

    **Returns:**
    - `zodiacal_releasing`: Lot, lot sign + degree, nested periods, current path.
    """
    log_request_with_body(logger, request, "Zodiacal releasing request", request_body.model_dump_json())

    try:
        subject = await run_heavy(build_subject, request_body.subject)

        result = await run_heavy(
            ZodiacalReleasingFactory.from_subject,
            subject,
            lot=request_body.lot,
            levels=request_body.levels,
            target_date=request_body.target_date,
            life_cap_years=request_body.life_cap_years,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "zodiacal_releasing": dump(result),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


@router.post("/api/v6/advanced/profections", response_model=ProfectionsResponseModel)
async def profections(request_body: ProfectionsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/profections`

    Compute annual profections — the Hellenistic year-lord technique. Each
    completed year of life activates one house counted from the Ascendant
    (age 0 = 1st house), cycling every twelve years; the Lord of the Year is
    the traditional ruler of the sign on the profected house's cusp, in the
    subject's own house system.

    **Parameters:**
    - `subject`: Natal subject (requires the twelve house cusps).
    - `target_date`: ISO date (`YYYY-MM-DD`) the current year is resolved
      against. Defaults to today in the subject's timezone.
    - `years_before` / `years_after`: The window of years around the current one.

    **Returns:**
    - `profections`: The current profection year and the surrounding window.
    """
    log_request_with_body(logger, request, "Profections request", request_body.model_dump_json())

    try:
        subject = await run_heavy(build_subject, request_body.subject)

        result = await run_heavy(
            ProfectionsFactory.from_subject,
            subject,
            target_date=request_body.target_date,
            years_before=request_body.years_before,
            years_after=request_body.years_after,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "profections": dump(result),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


@router.post("/api/v6/advanced/firdaria", response_model=FirdariaResponseModel)
async def firdaria(request_body: FirdariaRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/firdaria`

    Compute the firdaria (Persian time-lord) periods. Life divides into a
    fixed 75-year sequence of planetary periods whose order depends on the
    chart's sect — day charts open with the Sun, night charts with the Moon.
    Each planetary period subdivides into seven sub-periods opening with its
    own lord; the two node periods close the cycle undivided.

    **Parameters:**
    - `subject`: Natal subject. Requires a real sect (`is_diurnal`); a midpoint
      composite has no horizon and is rejected.
    - `target_date`: ISO date (`YYYY-MM-DD`) the current period is resolved
      against. Defaults to now in the subject's timezone.
    - `life_cap_years`: How far the timeline is unrolled (default 120).

    **Returns:**
    - `firdaria`: Sect, the major periods with sub-periods, and the current
      period/sub-period pointers.
    """
    log_request_with_body(logger, request, "Firdaria request", request_body.model_dump_json())

    try:
        subject = await run_heavy(build_subject, request_body.subject)

        result = await run_heavy(
            FirdariaFactory.from_subject,
            subject,
            target_date=request_body.target_date,
            life_cap_years=request_body.life_cap_years,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "firdaria": dump(result),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


@router.post("/api/v6/advanced/horary-indicators", response_model=HoraryIndicatorsResponseModel)
async def horary_indicators(request_body: HoraryIndicatorsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/horary-indicators`

    Assemble horary significators and the classical considerations before
    judgment for a question chart: the querent's (1st house) and quesited's
    (7th house) significators via classical rulership, the Ascendant degree
    read from the true Ascendant point (Whole Sign safe), the considerations
    as stable keys, and the chart's mutual receptions.

    **Parameters:**
    - `subject`: Chart cast for the moment of the question.
    - `is_moon_void`: Whether the Moon is void of course, when known from the
      void-of-course search. Omitted: the Moon considerations are skipped.

    **Returns:**
    - `horary_indicators`: Significators, Ascendant degree, considerations,
      mutual receptions.
    """
    log_request_with_body(logger, request, "Horary indicators request", request_body.model_dump_json())

    try:
        subject = await run_heavy(build_subject, request_body.subject)

        result = await run_heavy(
            HoraryIndicatorsFactory.from_subject,
            subject,
            is_moon_void=request_body.is_moon_void,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "horary_indicators": dump(result),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Secondary Progressions
# ===========================================================================


@router.post(
    "/api/v6/advanced/secondary-progressions",
    response_model=SecondaryProgressionsResponseModel,
)
async def secondary_progressions(request_body: SecondaryProgressionsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/secondary-progressions`

    Compute the day-for-a-year secondary-progressed chart for a target
    moment. The mapping is `progressed_days = (target − birth) /
    365.25`; the progressed chart is calculated for that moment at the
    natal location, reusing every natal calculation setting (zodiac
    type, sidereal mode, house system, perspective, active points).

    Pass exactly one of `target_iso_utc_datetime` or `target_year`.

    **Parameters:**
    - `subject`: Natal subject to progress.
    - `target_iso_utc_datetime` / `target_year`: Target moment (exactly one).
    - `active_points`: Points to calculate and use in aspect detection.
    - `compute_aspects`, `aspect_orb`, `aspects`: Progressed-to-natal aspect
      detection (on by default, 3 degree orb, Ptolemaic aspects).
    - `point_orb_adjustments`, `point_orb_adjustment_strategy`: Per-point orb tuning.

    **Returns:**
    - `progressed_subject`: Full `AstrologicalSubjectModel` for the
      progressed moment, identical in shape to a natal subject.
    - `target_iso_utc_datetime`: The resolved target moment (UTC).
    - `ephemeris_iso_utc_datetime`: The ephemeris moment the day-for-a-year
      mapping points at.
    - `progressed_points`: Per-point natal vs progressed comparison with the
      engine's `sign_changed` ingress flag.
    - `progressed_to_natal_aspects`: Progressed-to-natal aspect contacts
      (empty when `compute_aspects=false`).
    """
    log_request_with_body(logger, request, "Secondary progressions request", request_body.model_dump_json())

    try:
        active_points = request_body.active_points
        subject = await run_heavy(build_subject, request_body.subject, active_points=active_points)
        result = await run_heavy(
            SecondaryProgressionFactory.compute_full,
            subject,
            target_iso_utc_datetime=request_body.target_iso_utc_datetime,
            target_year=request_body.target_year,
            active_points=active_points,
            compute_aspects=request_body.compute_aspects,
            aspect_orb=request_body.aspect_orb,
            aspects=request_body.aspects,
            point_orb_adjustments=request_body.point_orb_adjustments,
            point_orb_adjustment_strategy=request_body.point_orb_adjustment_strategy,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "progressed_subject": dump(result.progressed_subject),
                "target_iso_utc_datetime": result.target_iso_utc_datetime,
                "ephemeris_iso_utc_datetime": result.ephemeris_iso_utc_datetime,
                # Per-point comparison with the engine's sign_changed ingress
                # flag — the whole reason clients stopped diffing sign strings.
                "progressed_points": [p.model_dump() for p in result.progressed_points],
                "progressed_to_natal_aspects": [a.model_dump() for a in result.progressed_to_natal_aspects],
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Solar Arc Directions
# ===========================================================================


@router.post(
    "/api/v6/advanced/solar-arc-directions",
    response_model=SolarArcDirectionsResponseModel,
)
async def solar_arc_directions(request_body: SolarArcDirectionsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/solar-arc-directions`

    Compute the solar arc and the directed-to-natal aspect picture for
    a target moment. The progressed Sun's longitude minus the natal
    Sun's longitude (shortest arc, signed) is the *solar arc*; that
    single arc is applied to every requested natal point.

    Pass exactly one of `target_iso_utc_datetime` or `target_year`.

    **Returns:**
    - `solar_arc_subject`: `SolarArcSubjectModel` carrying the arc,
      directed-point list (with `sign_changed` flag), and
      directed-to-natal aspect contacts.
    """
    log_request_with_body(logger, request, "Solar arc directions request", request_body.model_dump_json())

    try:
        active_points = request_body.active_points
        subject = await run_heavy(build_subject, request_body.subject, active_points=active_points)
        result = await run_heavy(
            SolarArcFactory.compute,
            subject,
            target_iso_utc_datetime=request_body.target_iso_utc_datetime,
            target_year=request_body.target_year,
            active_points=active_points,
            compute_aspects=request_body.compute_aspects,
            aspect_orb=request_body.aspect_orb,
            aspects=request_body.aspects,
            point_orb_adjustments=request_body.point_orb_adjustments,
            point_orb_adjustment_strategy=request_body.point_orb_adjustment_strategy,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "solar_arc_subject": dump(result),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Secondary Progressions — SVG Chart
# ===========================================================================


@router.post("/api/v6/chart/secondary-progressions", response_model=ProgressionChartResponseModel)
async def secondary_progressions_chart(request_body: SecondaryProgressionsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/chart/secondary-progressions`

    Compute the progressed chart and render a biwheel SVG (natal inner,
    progressed outer).

    **Parameters:**
    - `theme`, `style`, `glyph_size`, `show_zodiac_background_ring`, `transparent_background`
    - `show_motion_state`, `show_out_of_bounds`, `show_aspect_movement`,
      `show_relationship_score`, `show_ayanamsa_value`, `show_polar_fallback_note`
      (all default false)
    - `include_house_comparison`, `axis_orb_limit`, `distribution_method`,
      `custom_distribution_weights`: Chart-data computation options.
    - `language` is not accepted on this route; the SVG is rendered in English.
    - The remaining `/chart/*` rendering options are not accepted here: this route
      always returns wheel + grid, so `split_chart` and the info-panel flags it
      would switch have nothing to act on.

    **Returns:**
    - `status`: "OK"
    - `chart_wheel`: SVG of the biwheel (no aspect grid)
    - `chart_grid`: SVG of the aspect grid (separate panel)
    - `chart_data`: DualChartDataModel (Progression type)
    """
    log_request_with_body(logger, request, "Progression chart request", request_body.model_dump_json())

    try:
        active_points = request_body.active_points
        subject = await run_heavy(build_subject, request_body.subject, active_points=active_points)
        progressed = await run_heavy(
            SecondaryProgressionFactory.compute,
            subject,
            target_iso_utc_datetime=request_body.target_iso_utc_datetime,
            target_year=request_body.target_year,
        )
        chart_data = await run_heavy(
            ChartDataFactory.create_progression_chart_data,
            subject,
            progressed,
            active_points=active_points,
            # compute_aspects=false is honored the same way /advanced/secondary-
            # progressions honors it: an empty active-aspects list yields an empty
            # aspect table (None would fall back to the factory defaults).
            active_aspects=(_predictive_active_aspects(request_body.aspect_orb, request_body.aspects) if request_body.compute_aspects else []),
            include_house_comparison=request_body.include_house_comparison,
            axis_orb_limit=request_body.axis_orb_limit,
            point_orb_adjustments=request_body.point_orb_adjustments,
            point_orb_adjustment_strategy=request_body.point_orb_adjustment_strategy,
            distribution_method=request_body.distribution_method,
            custom_distribution_weights=request_body.custom_distribution_weights,
        )

        payload = await run_heavy(_render_progression_chart_payload, chart_data, request_body)
        return JSONResponse(content=payload, status_code=200)

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


@router.post("/api/v6/chart/solar-arc-directions", response_model=ProgressionChartResponseModel)
async def solar_arc_chart(request_body: SolarArcDirectionsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/chart/solar-arc-directions`

    Compute the solar arc directions and render a biwheel SVG (natal
    inner ring, directed outer ring). Houses and angles stay on the
    natal frame; every directable point is shifted forward by the
    solar arc.

    **Parameters:**
    - `theme`, `style`, `glyph_size`, `show_zodiac_background_ring`, `transparent_background`
    - `show_motion_state`, `show_out_of_bounds`, `show_aspect_movement`,
      `show_relationship_score`, `show_ayanamsa_value`, `show_polar_fallback_note`
      (all default false)
    - `include_house_comparison`, `axis_orb_limit`, `distribution_method`,
      `custom_distribution_weights`: Chart-data computation options.
    - `language` is not accepted on this route; the SVG is rendered in English.
    - The remaining `/chart/*` rendering options are not accepted here: this route
      always returns wheel + grid, so `split_chart` and the info-panel flags it
      would switch have nothing to act on.

    **Returns:**
    - `status`: "OK"
    - `chart_wheel`: SVG of the biwheel (no aspect grid)
    - `chart_grid`: SVG of the aspect grid (separate panel)
    - `chart_data`: DualChartDataModel (Progression type — solar arc
      shares the symbolic-direction structure with progressions)
    """
    log_request_with_body(logger, request, "Solar arc chart request", request_body.model_dump_json())

    try:
        active_points = request_body.active_points
        subject = await run_heavy(build_subject, request_body.subject, active_points=active_points)
        directed = await run_heavy(
            SolarArcFactory.compute_directed_subject,
            subject,
            target_iso_utc_datetime=request_body.target_iso_utc_datetime,
            target_year=request_body.target_year,
        )
        chart_data = await run_heavy(
            ChartDataFactory.create_progression_chart_data,
            subject,
            directed,
            active_points=active_points,
            active_aspects=_predictive_active_aspects(request_body.aspect_orb, request_body.aspects),
            include_house_comparison=request_body.include_house_comparison,
            axis_orb_limit=request_body.axis_orb_limit,
            point_orb_adjustments=request_body.point_orb_adjustments,
            point_orb_adjustment_strategy=request_body.point_orb_adjustment_strategy,
            distribution_method=request_body.distribution_method,
            custom_distribution_weights=request_body.custom_distribution_weights,
        )

        payload = await run_heavy(_render_progression_chart_payload, chart_data, request_body)
        return JSONResponse(content=payload, status_code=200)

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


@router.post("/api/v6/chart-data/secondary-progressions", response_model=ChartDataResponseModel)
async def secondary_progressions_chart_data(request_body: SecondaryProgressionsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/chart-data/secondary-progressions`

    Compute the progressed chart data (no SVG rendering).

    **Parameters:**
    - `subject`, `target_iso_utc_datetime` / `target_year`, `active_points`,
      `aspect_orb`, `aspects`, `compute_aspects` — as in
      `/advanced/secondary-progressions`.
    - `include_house_comparison`, `axis_orb_limit`, `distribution_method`,
      `custom_distribution_weights`: Chart-data computation options.
    - `language` is not accepted on this route.

    **Returns:**
    - `status`: "OK"
    - `chart_data`: DualChartDataModel (Progression type)
    """
    log_request_with_body(logger, request, "Progression chart-data request", request_body.model_dump_json())

    try:
        active_points = request_body.active_points
        subject = await run_heavy(build_subject, request_body.subject, active_points=active_points)
        progressed = await run_heavy(
            SecondaryProgressionFactory.compute,
            subject,
            target_iso_utc_datetime=request_body.target_iso_utc_datetime,
            target_year=request_body.target_year,
        )
        # Same aspect configuration as /chart/secondary-progressions, so the
        # data-only endpoint returns the same aspect table as the chart one.
        chart_data = await run_heavy(
            ChartDataFactory.create_progression_chart_data,
            subject,
            progressed,
            active_points=active_points,
            # compute_aspects=false is honored the same way /advanced/secondary-
            # progressions honors it: an empty active-aspects list yields an empty
            # aspect table (None would fall back to the factory defaults).
            active_aspects=(_predictive_active_aspects(request_body.aspect_orb, request_body.aspects) if request_body.compute_aspects else []),
            include_house_comparison=request_body.include_house_comparison,
            axis_orb_limit=request_body.axis_orb_limit,
            point_orb_adjustments=request_body.point_orb_adjustments,
            point_orb_adjustment_strategy=request_body.point_orb_adjustment_strategy,
            distribution_method=request_body.distribution_method,
            custom_distribution_weights=request_body.custom_distribution_weights,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "chart_data": dump(chart_data),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Astro-Cartography
# ===========================================================================


@router.post("/api/v6/advanced/astro-cartography", response_model=AstroCartographyResponseModel)
async def astro_cartography(request_body: AstroCartographyRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/astro-cartography`

    Compute astro-cartography (ACG) planetary lines showing where planets cross
    ASC/DSC/MC/IC lines on the Earth's surface.

    **Parameters:**
    - `subject`: Natal subject.
    - `step`: Longitude resolution in degrees (default 1.0).
    - `tolerance`: Altitude tolerance for line detection.
    - `lat_range_min`, `lat_range_max`: Latitude bounds (default -66 to 66).
    - `planets`: Optional list of planets to calculate.

    **Returns:**
    - `lines`: List of ACG lines with planet, type, and coordinate points.
    """
    log_request_with_body(logger, request, "Astro-cartography request", request_body.model_dump_json())

    try:
        subject = await run_heavy(build_subject, request_body.subject)

        kwargs = {
            "step": request_body.step,
            "lat_range": (request_body.lat_range_min, request_body.lat_range_max),
        }
        if request_body.tolerance is not None:
            kwargs["tolerance"] = request_body.tolerance
        if request_body.planets is not None:
            kwargs["planets"] = request_body.planets

        lines = await run_heavy(AstroCartographyFactory.compute, subject, **kwargs)

        return JSONResponse(
            content={
                "status": "OK",
                "lines": dump(lines),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Declination Aspects
# ===========================================================================


@router.post("/api/v6/advanced/declination-aspects", response_model=DeclinationAspectsResponseModel)
async def declination_aspects(request_body: DeclinationAspectsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/declination-aspects`

    Compute declination aspects (parallel / contra-parallel) within a single chart.

    **Parameters:**
    - `subject`: Subject for calculation.
    - `active_points`: Points to include (optional).
    - `orb`: Maximum orb in degrees (default 1.0).

    **Returns:**
    - `aspects`: List of declination aspects.
    """
    log_request_with_body(logger, request, "Declination aspects request", request_body.model_dump_json())

    try:
        active_points = resolve_active_points(request_body.active_points)
        subject = await run_heavy(build_subject, request_body.subject, active_points=active_points)

        aspects = await run_heavy(
            AspectsFactory.single_chart_declination_aspects,
            subject,
            active_points=active_points,
            orb=request_body.orb,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "aspects": dump(aspects),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


@router.post("/api/v6/advanced/declination-aspects/dual", response_model=DeclinationAspectsResponseModel)
async def dual_declination_aspects(request_body: DualDeclinationAspectsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/declination-aspects/dual`

    Compute declination aspects (parallel / contra-parallel) between two charts.

    **Parameters:**
    - `first_subject`, `second_subject`: Two subjects for comparison.
    - `active_points`: Points to include (optional).
    - `orb`: Maximum orb in degrees (default 1.0).

    **Returns:**
    - `aspects`: List of declination aspects.
    """
    log_request_with_body(logger, request, "Dual declination aspects request", request_body.model_dump_json())

    try:
        active_points = resolve_active_points(request_body.active_points)
        first_subject = await run_heavy(build_subject, request_body.first_subject, active_points=active_points)
        second_subject = await run_heavy(build_subject, request_body.second_subject, active_points=active_points)

        aspects = await run_heavy(
            AspectsFactory.dual_chart_declination_aspects,
            first_subject,
            second_subject,
            active_points=active_points,
            orb=request_body.orb,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "aspects": dump(aspects),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Transit Aspect Timeline
# ===========================================================================


@lru_cache(maxsize=16)
def _accepts_keyword(target: object, keyword: str) -> bool:
    """True when ``target`` takes ``keyword``; False when it does not or cannot
    be introspected (assume legacy). Lets one handler run unchanged across
    kerykeion releases that added an optional parameter."""
    try:
        return keyword in inspect.signature(cast(Callable[..., Any], target)).parameters
    except (TypeError, ValueError):
        return False


def _transit_ephemeris_factory(
    request_body: TransitEventsRequestModel,
    natal_subject: AstrologicalSubjectModel,
    active_points: list,
    *,
    calculate_dignities: bool = False,
) -> EphemerisDataFactory:
    """Compatibility wrapper around the REST/MCP shared series builder."""
    from ..utils.transit_series import build_transit_series_factory

    return build_transit_series_factory(
        start_date=request_body.start_date,
        end_date=request_body.end_date,
        step_type=request_body.step_type,
        step=request_body.step_days,
        subject_request=request_body.subject,
        natal_subject=natal_subject,
        active_points=active_points,
        calculate_dignities=calculate_dignities,
    )


@router.post("/api/v6/advanced/transit-aspect-timeline", response_model=TransitEventsResponseModel)
async def transit_aspect_timeline(request_body: TransitEventsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/transit-aspect-timeline`

    Compute transit aspect windows over a time range. Identifies when transiting planets
    form aspects to natal positions, with optional exact-moment bisection refinement.

    **Parameters:**
    - `subject`: Natal subject to track transits against.
    - `start_date`, `end_date`: ISO date range (max span 5 years, and at most
      10000 samples at the chosen step).
    - `step_type`: Sampling unit — 'days' (default), 'hours' or 'minutes'.
    - `step_days`: Step size in `step_type` units (default 1).
    - `refine_exact_moments`: Iteratively refine each exact moment between
      adjacent samples (ternary search; default false).
    - `refinement_iterations`: Refinement iterations — precision improves with
      the count: the default 12 gives roughly sub-15-minute precision at daily
      steps, and 30 reaches sub-second.
    - `active_points`, `active_aspects`: Override defaults.
    - `axis_orb_limit`: Override the maximum orb for aspects involving
      ASC/MC/DSC/IC.

    **Returns:**
    - `events`: List of transit events with applying start, exact moment, separating end.
    - `subject`: The natal subject the transits were computed against.
    """
    log_request_with_body(logger, request, "Transit events request", request_body.model_dump_json())

    try:
        active_points = resolve_active_points(request_body.active_points)
        active_aspects = resolve_active_aspects(request_body.active_aspects)
        natal_subject = await run_heavy(build_subject, request_body.subject, active_points=active_points)

        ephemeris_factory = _transit_ephemeris_factory(request_body, natal_subject, active_points)
        ephemeris_points = await run_heavy(ephemeris_factory.get_ephemeris_data_as_astrological_subjects)

        transits_factory = TransitsTimeRangeFactory(
            natal_chart=natal_subject,
            ephemeris_data_points=ephemeris_points,
            active_points=active_points,
            active_aspects=active_aspects,
            axis_orb_limit=request_body.axis_orb_limit,
        )

        result = await run_heavy(
            transits_factory.get_transit_events,
            refine_exact_moments=request_body.refine_exact_moments,
            refinement_iterations=request_body.refinement_iterations,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "events": dump(result.events),
                "subject": dump(result.subject),
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Ephemeris Data
# ===========================================================================


@lru_cache(maxsize=8)
def _ephemeris_factory_supports_fixed_stars(factory: object) -> bool:
    """True when the installed kerykeion ``EphemerisDataFactory`` accepts the
    ``active_fixed_stars`` parameter (Kerykeion >= 6.0.0a75). Cached per
    factory object so the signature inspection runs once per interpreter,
    while monkeypatched test factories still get their own entry."""
    try:
        return "active_fixed_stars" in inspect.signature(cast(Callable[..., Any], factory)).parameters
    except (TypeError, ValueError):  # non-introspectable callable: assume legacy
        return False


@router.post("/api/v6/advanced/ephemeris", response_model=EphemerisResponseModel)
async def ephemeris_data(request_body: EphemerisRequestModel, request: Request) -> Response:
    """
    **POST** `/api/v6/advanced/ephemeris`

    Generate an ephemeris table with planetary positions and house cusps
    over a date range at configurable intervals.

    **Parameters:**
    - `start_date`, `end_date`: ISO date range (at most 732 samples at the
      chosen step, and 30000 point calculations overall).
    - `step_type`: 'days', 'hours', or 'minutes'.
    - `step`: Step size (e.g. 1 day, 6 hours).
    - `latitude`, `longitude`, `timezone`: Observer location.
    - `is_dst`: Daylight-saving disambiguation for ambiguous local times.
    - `zodiac_type`, `sidereal_mode`, `houses_system_identifier`, `perspective_type`: Config.
    - `custom_ayanamsa_t0`, `custom_ayanamsa_ayan_t0`: Required pair for
      `sidereal_mode='USER'`.
    - `active_points`: Bounded point selection calculated at every sample.
    - `active_fixed_stars`: Bounded fixed-star selection calculated at every sample.
    - `include_houses`, `omit_nulls`: Compact-response controls for large tables.

    **Returns:**
    - `ephemeris`: List of samples with point source/coverage metadata and
      machine-readable omission warnings. Samples carry a `fixed_stars` key
      only when `active_fixed_stars` was requested.
    - `fixed_stars_truncated`: Present only when the requested star list
      exceeded the work budget — how many stars were requested/served and
      which were dropped.
    """
    log_request_with_body(logger, request, "Ephemeris data request", request_body.model_dump_json())

    try:
        requested_fixed_stars = list(request_body.active_fixed_stars or [])
        if requested_fixed_stars and not settings.ephemeris_fixed_stars_enabled:
            return JSONResponse(
                content={
                    "status": "ERROR",
                    "message": "fixed stars temporarily disabled on this deployment: remove active_fixed_stars and retry.",
                    "error_type": "FixedStarsDisabledError",
                },
                status_code=422,
            )
        if requested_fixed_stars and not _ephemeris_factory_supports_fixed_stars(EphemerisDataFactory):
            return JSONResponse(
                content={
                    "status": "ERROR",
                    "message": "installed kerykeion does not support fixed stars in ephemeris: remove active_fixed_stars or upgrade the runtime.",
                    "error_type": "FixedStarsUnsupportedError",
                },
                status_code=503,
            )

        start_dt = datetime.fromisoformat(request_body.start_date)
        end_dt = datetime.fromisoformat(request_body.end_date)

        # Enforce the work budget on the star list HERE, deterministically and
        # out loud: serve the affordable prefix of the request (the caller's
        # own order) and declare the cut in the response. The request model
        # only rejects when the points alone exceed the budget — clients must
        # never need to mirror this cost model to pre-trim their star list.
        # `requested_fixed_stars` stays the caller's original request for the
        # rest of the handler (it decides whether responses carry the
        # fixed_stars key at all); `served_fixed_stars` is the affordable
        # prefix actually forwarded to the factory. Keeping them separate
        # means a zero-capacity truncation still serializes fixed_stars as []
        # instead of silently dropping the key the contract promises.
        served_fixed_stars = requested_fixed_stars
        fixed_stars_truncation: dict | None = None
        if requested_fixed_stars:
            samples = _count_ephemeris_samples(
                request_body.start_date,
                request_body.end_date,
                timezone_name=request_body.timezone,
                is_dst=request_body.is_dst,
                step_type=request_body.step_type,
                step=request_body.step,
            )
            n_points = len(request_body.active_points or DEFAULT_ACTIVE_POINTS)
            affordable = max(
                0,
                (EPHEMERIS_MAX_POINT_CALCULATIONS // max(1, samples) - n_points) // FIXED_STAR_WORK_UNIT_WEIGHT,
            )
            if len(requested_fixed_stars) > affordable:
                dropped = requested_fixed_stars[affordable:]
                fixed_stars_truncation = {
                    "requested": len(requested_fixed_stars),
                    "served": affordable,
                    "dropped": dropped,
                }
                logger.info(
                    "Ephemeris fixed-star budget: serving %d of %d requested stars (%d samples, %d points).",
                    affordable,
                    len(requested_fixed_stars),
                    samples,
                    n_points,
                )
                served_fixed_stars = requested_fixed_stars[:affordable]

        # Only forward the kwarg when there are stars to serve so the
        # star-less path keeps working (and stays byte-identical) on legacy
        # factories that predate the parameter.
        fixed_stars_kwargs: dict = {"active_fixed_stars": served_fixed_stars} if served_fixed_stars else {}

        factory = EphemerisDataFactory(
            start_datetime=start_dt,
            end_datetime=end_dt,
            step_type=request_body.step_type,
            step=request_body.step,
            lat=request_body.latitude,
            lng=request_body.longitude,
            tz_str=request_body.timezone,
            is_dst=request_body.is_dst,
            zodiac_type=request_body.zodiac_type or "Tropical",
            sidereal_mode=request_body.sidereal_mode,
            houses_system_identifier=request_body.houses_system_identifier or "P",
            perspective_type=request_body.perspective_type or "Apparent Geocentric",
            custom_ayanamsa_t0=request_body.custom_ayanamsa_t0,
            custom_ayanamsa_ayan_t0=request_body.custom_ayanamsa_ayan_t0,
            active_points=(list(cast(Any, request_body.active_points)) if request_body.active_points is not None else None),
            # Defense in depth: mirror EphemerisRequestModel's table-specific
            # sample cap inside the factory as well.
            max_days=EPHEMERIS_MAX_SAMPLES,
            max_hours=EPHEMERIS_MAX_SAMPLES,
            max_minutes=EPHEMERIS_MAX_SAMPLES,
            **fixed_stars_kwargs,
        )

        data = await run_heavy(factory.get_ephemeris_data, as_model=True)

        def _serialize_body() -> bytes:
            # Serialization cost scales with samples × nested models: when the
            # caller excluded houses, skip the twelve house-cusp models at dump
            # time instead of serializing and then discarding them (the previous
            # dump-then-overwrite wasted 12 model dumps per sample).
            exclude = None if request_body.include_houses else {"houses"}
            rows = [sample.model_dump(mode="json", exclude=exclude, exclude_none=request_body.omit_nulls) for sample in data]
            for row in rows:
                if not request_body.include_houses:
                    row["houses"] = []
                # Public contract: the fixed_stars key is present only when stars
                # were requested, keeping star-less responses byte-identical
                # across kerykeion versions (newer models always carry the field).
                if not requested_fixed_stars:
                    row.pop("fixed_stars", None)
            # Same dumps arguments as starlette's JSONResponse.render, so the
            # bytes on the wire are identical to the previous in-loop path.
            # The truncation field appears ONLY when a cut happened, keeping
            # untrimmed responses byte-identical to the previous contract.
            payload: dict = {"status": "OK", "ephemeris": rows}
            if fixed_stars_truncation is not None:
                payload["fixed_stars_truncated"] = fixed_stars_truncation
            return json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=None,
                separators=(",", ":"),
            ).encode("utf-8")

        # Dumping and JSON-encoding hundreds of samples takes hundreds of
        # milliseconds: keep it off the event loop so liveness probes stay
        # responsive while a batch is being serialized.
        body = await run_heavy(_serialize_body)

        return Response(
            content=body,
            media_type="application/json",
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Transit Moments (snapshot-per-date)
# ===========================================================================


@router.post("/api/v6/advanced/transit-daily-aspects", response_model=TransitMomentsResponseModel)
async def transit_daily_aspects(request_body: TransitEventsRequestModel, request: Request) -> Response:
    """
    **POST** `/api/v6/advanced/transit-daily-aspects`

    Compute daily transit snapshots: for each date in the range, return all
    active aspects between transiting planets and natal positions.

    Returns per-day data (date + list of aspects), while transit-aspect-timeline
    returns per-aspect data (aspect + applying/exact/separating window).

    **Parameters:**
    - `subject`: Natal subject to track transits against.
    - `start_date`, `end_date`: ISO date range (max span 5 years, and at most
      10000 samples at the chosen step).
    - `step_type`, `step_days`: Sampling unit and step size (default 1 day).
    - `active_points`, `active_aspects`: Override defaults.
    - `axis_orb_limit`: Override the maximum orb for aspects involving
      ASC/MC/DSC/IC.
    - `include_transit_subjects`: Attach the full transiting chart to every
      snapshot (default false). Roughly one chart per sample: narrow
      `active_points` and the range accordingly.

    **Returns:**
    - `transits`: Per-date snapshots, each with the active aspects at that moment
      and, when `include_transit_subjects` is set, a `subject` with the transiting
      positions, signs, motion state, lunar phase and (with
      `subject.calculate_dignities`) essential dignities.
    - `subject`: The natal subject the transits were computed against.
    - `dates`: ISO dates of the snapshots.
    """
    log_request_with_body(logger, request, "Transit daily aspects request", request_body.model_dump_json())

    try:
        include_subjects = request_body.include_transit_subjects
        if include_subjects and not _accepts_keyword(TransitsTimeRangeFactory.get_transit_moments, "include_subjects"):
            return JSONResponse(
                content={
                    "status": "ERROR",
                    "message": "installed kerykeion does not support include_transit_subjects: remove the flag or upgrade the runtime.",
                    "error_type": "TransitSubjectsUnsupportedError",
                },
                status_code=503,
            )

        active_points = resolve_active_points(request_body.active_points)
        active_aspects = resolve_active_aspects(request_body.active_aspects)
        natal_subject = await run_heavy(build_subject, request_body.subject, active_points=active_points)

        ephemeris_factory = _transit_ephemeris_factory(
            request_body,
            natal_subject,
            active_points,
            # Dignities cost a pass per sample and reach the caller only through
            # the attached subjects: compute them just when they will be returned.
            calculate_dignities=include_subjects and request_body.subject.calculate_dignities,
        )
        ephemeris_points = await run_heavy(ephemeris_factory.get_ephemeris_data_as_astrological_subjects)

        transits_factory = TransitsTimeRangeFactory(
            natal_chart=natal_subject,
            ephemeris_data_points=ephemeris_points,
            active_points=active_points,
            active_aspects=active_aspects,
            axis_orb_limit=request_body.axis_orb_limit,
        )

        moments_kwargs: dict = {"include_subjects": True} if include_subjects else {}
        result = await run_heavy(transits_factory.get_transit_moments, **moments_kwargs)

        def _serialize_body() -> bytes:
            snapshots = dump(result.transits)
            assert isinstance(snapshots, list)
            # Public contract: a snapshot has a `subject` key only when it was
            # requested, keeping default responses byte-identical across
            # kerykeion versions (newer models always carry the field).
            if not include_subjects:
                for snapshot in snapshots:
                    snapshot.pop("subject", None)
            # Same dumps arguments as starlette's JSONResponse.render, so the
            # bytes on the wire match the previous in-loop path.
            return json.dumps(
                {
                    "status": "OK",
                    "transits": snapshots,
                    "subject": dump(result.subject),
                    "dates": result.dates,
                },
                ensure_ascii=False,
                allow_nan=False,
                indent=None,
                separators=(",", ":"),
            ).encode("utf-8")

        # With subjects attached the body is one chart per sample: dumping and
        # encoding it takes long enough to stall liveness probes if done on the
        # event loop.
        body = await run_heavy(_serialize_body)

        return Response(content=body, media_type="application/json", status_code=200)

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Report Generator
# ===========================================================================


@router.post("/api/v6/advanced/report", response_model=ReportResponseModel)
async def report(request_body: ReportRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/report`

    Generate a human-readable text report for an astrological subject.

    **Parameters:**
    - `subject`: Primary subject to generate report for.
    - `second_subject` + `chart_type` ('Synastry'/'Transit'/'Composite'): relational report.
    - `chart_type='Natal'` or `include_aspects=true`: include the aspect table.
    - `include_aspects`, `max_aspects`: aspect-table options.

    **Returns:**
    - `report`: Generated text report string.
    """
    log_request_with_body(logger, request, "Report request", request_body.model_dump_json())

    try:
        kwargs: dict[str, Any] = {}
        if request_body.include_aspects is not None:
            kwargs["include_aspects"] = request_body.include_aspects
        if request_body.max_aspects is not None:
            kwargs["max_aspects"] = request_body.max_aspects

        def _build_report_model():
            primary = build_subject(request_body.subject)
            chart_type = request_body.chart_type
            if chart_type in ("Synastry", "Transit", "Composite"):
                second = build_subject(request_body.second_subject)
                if chart_type == "Composite":
                    composite = CompositeSubjectFactory(primary, second).get_midpoint_composite_subject_model()
                    return ChartDataFactory.create_chart_data("Composite", composite)
                return ChartDataFactory.create_chart_data(chart_type, primary, second)
            # Single-subject: build natal chart data when aspects are wanted so the
            # aspect table is populated; otherwise a bare subject report.
            if chart_type == "Natal" or request_body.include_aspects:
                return ChartDataFactory.create_chart_data("Natal", primary)
            return primary

        model = await run_heavy(_build_report_model)
        generator = ReportGenerator(model)
        report_text = await run_heavy(generator.generate_report, **kwargs)

        return JSONResponse(
            content={
                "status": "OK",
                "report": report_text,
            },
            status_code=200,
        )

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Heliocentric Return
# ===========================================================================


@router.post("/api/v6/chart/heliocentric-return", response_model=HeliocentricReturnChartResponseModel)
async def heliocentric_return_chart(request_body: HeliocentricReturnRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/chart/heliocentric-return`

    Compute a heliocentric return chart for a given planet. The return occurs when
    the planet returns to its natal heliocentric longitude.

    **Parameters:**
    - `subject`: Natal subject.
    - `planet`: Planet whose heliocentric return to compute (e.g. 'Mars').
    - `year` or `iso_datetime`: Search start (exactly one).
    - `direction`: 'next' (default) or 'previous'.
    - `wheel_type`: 'dual' (natal + return biwheel, default) or 'single'.
    - `include_house_comparison`: House overlay for dual wheels.
    - `return_location`: Override the location the return chart is cast for.
    - The rendering options common to every `/chart/*` route, including
      `show_diurnality` and the six extended flags (`show_motion_state`,
      `show_out_of_bounds`, `show_aspect_movement`, `show_relationship_score`,
      `show_ayanamsa_value`, `show_polar_fallback_note`). "Heliocentric" here
      describes how the return instant is found, not how the wheel is cast: the
      chart takes the subject's own perspective — Apparent Geocentric by default,
      in which case it carries a diurnality line the flag switches off, but a
      heliocentric subject makes the return heliocentric too and the line is then
      omitted.

    **Returns:**
    - `return_type`: "Heliocentric"
    - `planet`: Planet name
    - `wheel_type`: "dual" | "single"
    - `chart_data` + SVG
    """
    log_request_with_body(logger, request, "Heliocentric return chart request", request_body.model_dump_json())

    try:
        chart_data = await run_heavy(calculate_heliocentric_return_chart_data, request_body)

        payload = await run_heavy(chart_payload_from_request, chart_data, request_body)
        payload["return_type"] = "Heliocentric"
        payload["planet"] = request_body.planet
        payload["wheel_type"] = request_body.wheel_type
        return JSONResponse(content=payload, status_code=200)

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


@router.post("/api/v6/chart-data/heliocentric-return", response_model=HeliocentricReturnChartResponseModel)
async def heliocentric_return_data(request_body: HeliocentricReturnDataRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/chart-data/heliocentric-return`

    Compute heliocentric return chart data (no SVG).

    **Parameters:**
    - `subject`, `planet`, `year`/`iso_datetime`, `direction`, `wheel_type`,
      `include_house_comparison`, `return_location` — as in
      `/chart/heliocentric-return`.
    - `active_points` / `active_aspects` and the other chart-data options.

    **Returns:**
    - `return_type`: "Heliocentric"
    - `planet`: Planet name
    - `wheel_type`: "dual" | "single"
    - `chart_data`: Chart data (no SVG)
    """
    log_request_with_body(logger, request, "Heliocentric return data request", request_body.model_dump_json())

    try:
        chart_data = await run_heavy(calculate_heliocentric_return_chart_data, request_body)

        payload = await run_heavy(chart_data_payload, chart_data)
        payload["return_type"] = "Heliocentric"
        payload["planet"] = request_body.planet
        payload["wheel_type"] = request_body.wheel_type
        return JSONResponse(content=payload, status_code=200)

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Lunar Node Crossing
# ===========================================================================


@router.post("/api/v6/chart/lunar-node-crossing", response_model=LunarNodeCrossingChartResponseModel)
async def lunar_node_crossing_chart(request_body: LunarNodeCrossingRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/chart/lunar-node-crossing`

    Compute the next lunar node crossing return chart. This occurs when the
    Moon's node returns to its natal position.

    **Parameters:**
    - `subject`: Natal subject.
    - `year` or `iso_datetime`: Search start (exactly one).
    - `direction`: 'next' (default) or 'previous'.
    - `wheel_type`: 'dual' (natal + crossing biwheel, default) or 'single'.
    - `include_house_comparison`: House overlay for dual wheels.
    - `return_location`: Override the location the crossing chart is cast for.
    - The rendering options common to every `/chart/*` route, including
      `show_diurnality` and the six extended flags (`show_motion_state`,
      `show_out_of_bounds`, `show_aspect_movement`, `show_relationship_score`,
      `show_ayanamsa_value`, `show_polar_fallback_note`).

    **Returns:**
    - `return_type`: "Lunar_Node_Crossing"
    - `wheel_type`: "dual" | "single"
    - `chart_data` + SVG
    """
    log_request_with_body(logger, request, "Lunar node crossing chart request", request_body.model_dump_json())

    try:
        chart_data = await run_heavy(calculate_lunar_node_crossing_chart_data, request_body)

        payload = await run_heavy(chart_payload_from_request, chart_data, request_body)
        payload["return_type"] = "Lunar_Node_Crossing"
        payload["wheel_type"] = request_body.wheel_type
        return JSONResponse(content=payload, status_code=200)

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


@router.post("/api/v6/chart-data/lunar-node-crossing", response_model=LunarNodeCrossingChartResponseModel)
async def lunar_node_crossing_data(request_body: LunarNodeCrossingDataRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/chart-data/lunar-node-crossing`

    Compute lunar node crossing data (no SVG).

    **Parameters:**
    - `subject`, `year`/`iso_datetime`, `direction`, `wheel_type`,
      `include_house_comparison`, `return_location` — as in
      `/chart/lunar-node-crossing`.
    - `active_points` / `active_aspects` and the other chart-data options.

    **Returns:**
    - `return_type`: "Lunar_Node_Crossing"
    - `wheel_type`: "dual" | "single"
    - `chart_data`: Chart data (no SVG)
    """
    log_request_with_body(logger, request, "Lunar node crossing data request", request_body.model_dump_json())

    try:
        chart_data = await run_heavy(calculate_lunar_node_crossing_chart_data, request_body)

        payload = await run_heavy(chart_data_payload, chart_data)
        payload["return_type"] = "Lunar_Node_Crossing"
        payload["wheel_type"] = request_body.wheel_type
        return JSONResponse(content=payload, status_code=200)

    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)


# ===========================================================================
# Occultation Global Search
# ===========================================================================


@router.post("/api/v6/advanced/occultations/global", response_model=OccultationSearchResponseModel)
async def occultation_search_global(request_body: OccultationSearchRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/occultations/global`

    Search for lunar occultations of a planet globally (not location-specific).

    **Parameters:**
    - `subject`: Subject defining the starting Julian Day.
    - `planet`: Occulted body — the planet the Moon passes in front of (default 'Venus').
    - `count`: Number of events to return.

    **Returns:**
    - `events`: List of global occultation events.
    """
    log_request_with_body(logger, request, "Global occultation search request", request_body.model_dump_json())

    try:
        subject = await run_heavy(build_subject, request_body.subject)

        planet_id = OCCULTATION_PLANET_IDS[request_body.planet]

        events = await run_occultation_search(
            "global",
            timeout=OCCULTATION_SEARCH_TIMEOUT_S,
            julian_day=subject.julian_day,
            planet_id=planet_id,
            count=request_body.count,
        )

        return JSONResponse(
            content={
                "status": "OK",
                "events": dump(events),
            },
            status_code=200,
        )

    except TimeoutError:
        return JSONResponse(
            content={
                "status": "ERROR",
                "message": "Occultation search timed out. Try reducing 'count'.",
                "error_type": "TimeoutError",
            },
            status_code=504,
        )
    except Exception as exc:  # pragma: no cover
        return await handle_exception(exc, request)
