from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field

from kerykeion import (
    ACGLineModel,
    HeliacalEventModel,
    IngressModel,
    SignPeriodModel,
    LunarEclipseModel,
    LunationModel,
    MidpointModel,
    MundaneAspectModel,
    OccultationModel,
    PlanetaryNodeModel,
    PrimaryDirectionModel,
    ProgressedToNatalAspectModel,
    SolarArcSubjectModel,
    SolarEclipseModel,
    SpeculumEntryModel,
    StationModel,
    RetrogradePeriodModel,
)
from kerykeion.schemas import ProgressedPointModel
from kerykeion.schemas import (
    AstrologicalSubjectModel,
    AspectModel,
    DualChartDataModel,
    EphemerisDictModel,
    KerykeionPointModel,
    MoonPhaseOverviewModel,
    PlanetaryPhenomenaModel,
    RelationshipScoreAspectModel,
    SingleChartDataModel,
    ScoreBreakdownItemModel,
    DominantsModel,
    TransitEventModel,
    TransitMomentModel,
    ZodiacalReleasingModel,
    ProfectionsModel,
    FirdariaModel,
    HoraryIndicatorsModel,
)
from kerykeion.schemas import ClassicalPlanet, RelationshipScoreDescription

# ``SolarPhaseThresholdsModel`` arrived with the a91 pin. Kept out of the block
# import above and guarded, because this module is imported by ``app.main``:
# against an engine that predates the symbol the top-level import would abort
# the whole application at startup — every endpoint down, not just the one
# field that cannot be typed. That is the same deploy-ahead reasoning as the
# local import in ``app/routers/advanced.py``, applied where an annotation
# forces the name to exist at module scope.
#
# The fallback widens the annotation instead of narrowing the contract: on such
# an engine ``PlanetaryPhenomenaFactory`` produces no thresholds at all, so the
# field is always ``None`` there and the looser type describes exactly what can
# happen. The committed ``openapi.json`` is generated against the pinned engine,
# so the published schema keeps the real model.
try:
    from kerykeion.schemas import SolarPhaseThresholdsModel
except ImportError:  # pragma: no cover - only reachable on a pre-a91 engine
    SolarPhaseThresholdsModel = Any  # type: ignore[assignment, misc]


class ValidationIssueResponseModel(BaseModel):
    """One field-level request validation failure."""

    loc: list[Union[str, int]] = Field(description="JSON path of the invalid field, beginning with 'body'.")
    msg: str = Field(description="Human-readable validation message.")
    type: str = Field(description="Stable Pydantic validation code.")


class ValidationErrorResponseModel(BaseModel):
    """Actual 422 envelope returned by the global validation handler."""

    status: Literal["ERROR"] = "ERROR"
    message: Literal["Validation failed"] = "Validation failed"
    errors: list[ValidationIssueResponseModel]


class ApplicationErrorResponseModel(BaseModel):
    """Typed application error used by capability and resource guards."""

    status: Literal["ERROR"] = "ERROR"
    message: str
    error_type: str


Public422Response = Union[ValidationErrorResponseModel, ApplicationErrorResponseModel]


class StatusResponseModel(BaseModel):
    """Response payload containing only the status field."""

    status: str = Field(description="The status of the response.")


class FixedStarMetadataResponseModel(BaseModel):
    """Metadata for a single fixed star entry in the catalog."""

    name: str = Field(description="IAU canonical name (e.g. 'Vindemiatrix', 'Deneb Algedi').")
    slug: str = Field(description="URL/identifier-safe slug (spaces -> underscores).")
    hip_number: Optional[int] = Field(default=None, description="Hipparcos catalog number.")
    nomenclature: Optional[str] = Field(default=None, description="Bayer/Flamsteed designation.")
    magnitude: Optional[float] = Field(default=None, description="Visual magnitude.")


class FixedStarsCatalogResponseModel(StatusResponseModel):
    """Response payload listing the full fixed-star catalog (libephemeris)."""

    source: Literal["libephemeris"] = Field(description="Catalog data source.")
    count: int = Field(description="Number of stars in the catalog.")
    stars: list[FixedStarMetadataResponseModel] = Field(description="Catalog entries.")


class ApiStatusResponseModel(StatusResponseModel):
    """Response payload for the API root status endpoint."""

    environment: str = Field(description="Deployment environment identifier.")
    debug: bool = Field(description="Whether debug mode is enabled.")


class EphemerisProbeStatusModel(BaseModel):
    """Sanitized ephemeris state exposed by public probes."""

    ready: bool = Field(description="Whether the sealed ephemeris runtime is usable.")
    state: str = Field(description="Current readiness state.")
    reason: Literal["ready", "provisioning", "runtime_validation_failed"] = Field(
        description="Stable, public reason code for the readiness state."
    )
    precision_tier: Optional[str] = Field(default=None, description="Available ephemeris precision tier, when known.")


class ProbeResponseModel(StatusResponseModel):
    """Public liveness/readiness response."""

    status: Literal["OK", "INITIALIZING"] = Field(description="Probe result.")
    kerykeion_version: str = Field(description="Installed kerykeion version.")
    ephemeris_ready: bool = Field(description="Compatibility alias for ephemeris.ready.")
    ephemeris: EphemerisProbeStatusModel


class SubjectResponseModel(StatusResponseModel):
    """Response payload containing a single astrological subject."""

    subject: AstrologicalSubjectModel = Field(description="Computed astrological subject.")


class ChartDataResponseModel(StatusResponseModel):
    """Response payload returning serialized chart data."""

    chart_data: Union[SingleChartDataModel, DualChartDataModel] = Field(description="Serialized chart data payload.")


class ChartResponseModel(ChartDataResponseModel):
    """Response payload returning chart data with optional rendered SVG assets."""

    chart: Optional[str] = Field(
        default=None,
        description="SVG representation of the chart when split charts are disabled.",
    )
    chart_wheel: Optional[str] = Field(
        default=None,
        description="SVG representation of the chart wheel when split charts are enabled.",
    )
    chart_grid: Optional[str] = Field(
        default=None,
        description="SVG representation of the aspect grid when split charts are enabled.",
    )


class ReturnChartResponseModel(ChartResponseModel):
    """Response payload for solar and lunar return chart requests."""

    return_type: Literal["Solar", "Lunar"] = Field(description="Type of planetary return.")
    wheel_type: Literal["dual", "single"] = Field(description="Rendered wheel configuration.")


class CompatibilityScoreResponseModel(StatusResponseModel):
    """Response payload for the compatibility score endpoint."""

    score: Optional[int] = Field(
        default=None,
        description="Numeric relationship (Ciro Discepolo) score.",
    )
    score_description: Optional[RelationshipScoreDescription] = Field(
        default=None,
        description="Categorical description of the score.",
    )
    is_destiny_sign: Optional[bool] = Field(
        default=None,
        description="Whether the subjects form a destiny-sign relationship.",
    )
    aspects: list[RelationshipScoreAspectModel] = Field(
        default_factory=list,
        description="Aspects considered for the score calculation.",
    )
    score_breakdown: list[ScoreBreakdownItemModel] = Field(
        default_factory=list,
        description="Breakdown of the scoring rules and points contributing to the total score.",
    )
    chart_data: DualChartDataModel = Field(description="Underlying chart data used to compute the score.")


class SubjectContextResponseModel(StatusResponseModel):
    """Response payload containing a single astrological subject with AI context."""

    context: str = Field(description="AI-optimized context string for the subject.")
    subject: AstrologicalSubjectModel = Field(description="Computed astrological subject.")


class ContextResponseModel(StatusResponseModel):
    """Response payload returning chart data with AI-optimized context."""

    context: str = Field(description="AI-optimized context string for the chart data.")
    chart_data: Union[SingleChartDataModel, DualChartDataModel] = Field(description="Serialized chart data payload.")


class MoonPhaseResponseModel(StatusResponseModel):
    """Response payload for moon phase details."""

    moon_phase_overview: MoonPhaseOverviewModel = Field(description="Detailed moon phase overview including illumination, upcoming phases, eclipses, and sun info.")


class MoonPhaseContextResponseModel(StatusResponseModel):
    """Response payload for moon phase details with AI-optimized context."""

    context: str = Field(description="AI-optimized XML context string for the moon phase overview.")
    moon_phase_overview: MoonPhaseOverviewModel = Field(description="Detailed moon phase overview including illumination, upcoming phases, eclipses, and sun info.")


class ReturnContextResponseModel(ContextResponseModel):
    """Response payload for solar and lunar return context requests."""

    return_type: Literal["Solar", "Lunar"] = Field(description="Type of planetary return.")
    wheel_type: Literal["dual", "single"] = Field(description="Rendered wheel configuration.")


class HeliocentricReturnContextResponseModel(ContextResponseModel):
    """Response payload for heliocentric return context requests."""

    return_type: Literal["Heliocentric"] = Field(description="Type of planetary return.")
    planet: Optional[str] = Field(default=None, description="Planet whose heliocentric return was computed (compute mode).")
    wheel_type: Literal["dual", "single"] = Field(description="Rendered wheel configuration.")


class LunarNodeCrossingContextResponseModel(ContextResponseModel):
    """Response payload for lunar node crossing context requests."""

    return_type: Literal["Lunar_Node_Crossing"] = Field(description="Type of planetary return.")
    wheel_type: Literal["dual", "single"] = Field(description="Rendered wheel configuration.")


# ===========================================================================
# v6 Advanced Feature Response Models
# ===========================================================================


class EclipseSearchResponseModel(StatusResponseModel):
    """Response payload for eclipse search results."""

    solar_eclipses: list[SolarEclipseModel] = Field(
        default_factory=list,
        description="List of solar eclipses found.",
    )
    lunar_eclipses: list[LunarEclipseModel] = Field(
        default_factory=list,
        description="List of lunar eclipses found.",
    )
    latitude: Optional[float] = Field(
        default=None,
        description="Search latitude (None for global search).",
    )
    longitude: Optional[float] = Field(
        default=None,
        description="Search longitude (None for global search).",
    )


class LunationsResponseModel(StatusResponseModel):
    """Response payload for a lunation search over a date range."""

    start_jd: Optional[float] = Field(default=None, description="Resolved start of the scan window, as a Julian Day.")
    end_jd: Optional[float] = Field(default=None, description="Resolved end of the scan window, as a Julian Day.")
    lunations: list[LunationModel] = Field(
        default_factory=list,
        description="Ordered lunations (new, first_quarter, full, last_quarter) with Sun/Moon positions.",
    )


class RetrogradeStationsResponseModel(StatusResponseModel):
    """Response payload for a retrograde/direct station search over a date range."""

    stations: list[StationModel] = Field(
        default_factory=list,
        description="Ordered planetary stations (SR=retrograde, SD=direct) with zodiac positions.",
    )


class SignIngressesResponseModel(StatusResponseModel):
    """Response payload for a zodiac sign ingress search over a date range."""

    ingresses: list[IngressModel] = Field(
        default_factory=list,
        description="Ordered sign ingresses (30 degree boundary crossings) with from/to signs and retrograde flag.",
    )


class MundaneAspectsResponseModel(StatusResponseModel):
    """Response payload for a mundane aspectarian search over a date range."""

    aspects: list[MundaneAspectModel] = Field(
        default_factory=list,
        description=("Ordered exact transiting-to-transiting aspects, each with its UTC instant, both bodies' longitudes/signs and retrograde flags."),
    )


class MoonVocWindowEntryModel(BaseModel):
    """One void-of-course window in a range scan (API JSON shape)."""

    moon_sign: str = Field(description="Sign the Moon is leaving (three-letter code).")
    next_sign: str = Field(description="Sign the Moon ingresses into, ending the void.")
    void_start: str = Field(description="Start of the void window, ISO-8601 UTC.")
    void_start_local: Optional[str] = Field(default=None, description="Void start in the request timezone (when provided).")
    void_end: str = Field(description="End of the void window (= ingress), ISO-8601 UTC.")
    void_end_local: Optional[str] = Field(default=None, description="Void end in the request timezone (when provided).")
    duration_minutes: float = Field(description="Window length in minutes.")
    # Forward reference: MoonAspectEventModel is defined later in this module
    # (next to the single-moment MoonVocModel); pydantic resolves it lazily.
    last_aspect: Optional["MoonAspectEventModel"] = Field(
        default=None,
        description="The aspect that opened the void (none for a whole-sign void).",
    )


class MoonVocWindowsResponseModel(StatusResponseModel):
    """Response payload for void-of-course Moon windows over a date range."""

    windows: list[MoonVocWindowEntryModel] = Field(
        default_factory=list,
        description="Chronologically ordered, non-overlapping void windows intersecting the range (unclipped).",
    )


class CalendarDayModel(BaseModel):
    """Per-day local layer of the astro-calendar (sun times + planetary hours)."""

    date: str = Field(description="Civil date (YYYY-MM-DD) in the request timezone.")
    sun_times: Optional[dict] = Field(
        default=None,
        description="Sun-times payload for the date (same shape as /api/v6/sun-times), or null when unavailable.",
    )
    planetary_hours: Optional[dict] = Field(
        default=None,
        description="Planetary-hours payload for the date (same shape as /api/v6/planetary-hours), or null when unavailable (e.g. polar day/night).",
    )


class AstroCalendarResponseModel(StatusResponseModel):
    """Response payload for the astro-calendar aggregator."""

    start_date: str = Field(description="Echo of the requested range start.")
    end_date: str = Field(description="Echo of the requested range end.")
    timezone: Optional[str] = Field(default=None, description="Echo of the request timezone (None when no location was provided).")
    ingresses: list[IngressModel] = Field(default_factory=list, description="Sign ingresses in range (with season markers on the Sun's cardinal crossings).")
    lunations: list[LunationModel] = Field(default_factory=list, description="Lunations in range.")
    solar_eclipses: list[SolarEclipseModel] = Field(default_factory=list, description="Solar eclipses in range.")
    lunar_eclipses: list[LunarEclipseModel] = Field(default_factory=list, description="Lunar eclipses in range.")
    retrograde_stations: list[StationModel] = Field(default_factory=list, description="Retrograde/direct stations in range.")
    voc_windows: list[MoonVocWindowEntryModel] = Field(default_factory=list, description="Void-of-course Moon windows intersecting the range.")
    aspectarian: list[MundaneAspectModel] = Field(default_factory=list, description="Exact mundane aspects in range.")
    days: list[CalendarDayModel] = Field(default_factory=list, description="Per-day sun times and planetary hours (when a location is provided).")
    sign_periods: list[SignPeriodModel] = Field(
        default_factory=list,
        description="Where each planet (Moon..Pluto) is, sign by sign, across the range: contiguous stays per planet, clipped to the range and flagged where clipped.",
    )
    retrograde_periods: list[RetrogradePeriodModel] = Field(
        default_factory=list,
        description="Retrograde spans (Mercury..Pluto, Chiron) across the range, clipped to it and flagged where clipped.",
    )


class PlanetaryPhenomenaResponseModel(StatusResponseModel):
    """Response payload for planetary phenomena calculation."""

    iso_datetime: Optional[str] = Field(default=None, description="ISO datetime of the calculation moment.")
    julian_day: Optional[float] = Field(default=None, description="Julian Day of the calculation moment.")
    phenomena: list[PlanetaryPhenomenaModel] = Field(
        default_factory=list,
        description="List of planetary phenomena (phase_angle, elongation, magnitude, solar_phase, etc.).",
    )
    solar_phase_thresholds: Optional[SolarPhaseThresholdsModel] = Field(
        default=None,
        description="Band half-widths, in degrees, that classified each entry's `solar_phase` — the request's override when one was sent, the traditional defaults otherwise.",
    )


class PlanetaryNodesResponseModel(StatusResponseModel):
    """Response payload for planetary nodes and apsides calculation."""

    iso_datetime: Optional[str] = Field(default=None, description="ISO datetime of the calculation moment.")
    julian_day: Optional[float] = Field(default=None, description="Julian Day of the calculation moment.")
    method: str = Field(description="Calculation method used ('mean' or 'osculating').")
    nodes: list[PlanetaryNodeModel] = Field(
        default_factory=list,
        description="List of planetary node data (ascending/descending nodes, perihelion/aphelion).",
    )


class HeliacalEventsResponseModel(StatusResponseModel):
    """Response payload for heliacal events search."""

    events: list[HeliacalEventModel] = Field(
        default_factory=list,
        description="List of heliacal events (rising, setting, evening first, morning last).",
    )


class OccultationSearchResponseModel(StatusResponseModel):
    """Response payload for stellar occultation search."""

    events: list[OccultationModel] = Field(
        default_factory=list,
        description="List of occultation events.",
    )


class RelocatedChartResponseModel(StatusResponseModel):
    """Response payload for relocated chart calculation."""

    subject: AstrologicalSubjectModel = Field(
        description="Relocated astrological subject (original planets, recalculated houses/angles).",
    )


class FixedStarDiscoveryResponseModel(StatusResponseModel):
    """Response payload for fixed star discovery."""

    stars: list[KerykeionPointModel] = Field(
        default_factory=list,
        description="List of prominent fixed stars found in conjunction with chart points.",
    )


class PrimaryDirectionsResponseModel(StatusResponseModel):
    """Response payload for primary directions calculation."""

    directions: list[PrimaryDirectionModel] = Field(
        default_factory=list,
        description="List of primary directions with promissor, significator, aspect, arc, and years.",
    )
    speculum: list[SpeculumEntryModel] = Field(
        default_factory=list,
        description="Speculum table with RA, declination, semi-arc, and pole for each point.",
    )


class MidpointsResponseModel(StatusResponseModel):
    """Response payload for midpoint table calculation."""

    midpoints: list[MidpointModel] = Field(
        default_factory=list,
        description=(
            "List of midpoint entries. Each carries point_a / point_b, the midpoint "
            "longitude on the shorter arc, sign + position-within-sign, the 90 deg "
            "dial position, and any third points that aspect the midpoint within orb."
        ),
    )


class ZodiacalReleasingResponseModel(StatusResponseModel):
    """Response payload for zodiacal releasing calculation."""

    zodiacal_releasing: ZodiacalReleasingModel = Field(
        description=("Zodiacal-releasing result: the lot, its sign and degree, the nested L1 periods (with sub-periods), and the current period chain for the target date."),
    )


class ProfectionsResponseModel(StatusResponseModel):
    """Response payload for annual profections."""

    profections: ProfectionsModel = Field(
        description=(
            "Annual profections: the current year (age, profected house, sign, Lord of the Year, "
            "anniversary boundaries) and the surrounding window of years."
        ),
    )


class FirdariaResponseModel(StatusResponseModel):
    """Response payload for firdaria time-lord periods."""

    firdaria: FirdariaModel = Field(
        description=(
            "Firdaria timeline: sect, the major periods with their sub-lord periods, and the "
            "current period/sub-period for the target date."
        ),
    )


class HoraryIndicatorsResponseModel(StatusResponseModel):
    """Response payload for horary indicators."""

    horary_indicators: HoraryIndicatorsModel = Field(
        description=(
            "Horary indicators: querent and quesited significators, the Ascendant degree, the "
            "considerations before judgment (as stable keys), and the chart's mutual receptions."
        ),
    )


class SecondaryProgressionsResponseModel(StatusResponseModel):
    """Response payload for secondary progressions calculation."""

    progressed_subject: AstrologicalSubjectModel = Field(
        description=("Progressed AstrologicalSubjectModel — same shape as a regular natal subject, with year/month/day fields referring to the progressed moment."),
    )
    target_iso_utc_datetime: Optional[str] = Field(default=None, description="Resolved ISO-8601 UTC timestamp of the target moment.")
    ephemeris_iso_utc_datetime: Optional[str] = Field(default=None, description="ISO-8601 UTC timestamp of the progressed ephemeris moment.")
    progressed_points: list[ProgressedPointModel] = Field(
        default_factory=list,
        description=(
            "Natal-vs-progressed comparison per point, carrying the engine's "
            "sign_changed ingress flag (the progressions counterpart of the "
            "solar-arc flag)."
        ),
    )
    progressed_to_natal_aspects: list[ProgressedToNatalAspectModel] = Field(
        default_factory=list,
        description="Aspects between progressed and natal points.",
    )


class SecondaryProgressionsContextResponseModel(StatusResponseModel):
    """Response payload for the secondary progressions context endpoint."""

    context: str = Field(description="AI-optimized context string for the progressed chart.")
    progressed_subject: AstrologicalSubjectModel = Field(
        description="Progressed AstrologicalSubjectModel for the target moment.",
    )


class SolarArcContextResponseModel(StatusResponseModel):
    """Response payload for the solar arc directions context endpoint."""

    context: str = Field(description="AI-optimized context string for the directed chart.")
    solar_arc_subject: SolarArcSubjectModel = Field(
        description="SolarArcSubjectModel: arc, directed points, directed-to-natal aspects.",
    )


class MidpointsContextResponseModel(StatusResponseModel):
    """Response payload for the midpoints context endpoint."""

    context: str = Field(description="AI-optimized context string for the midpoint table.")
    midpoints: list[MidpointModel] = Field(
        default_factory=list,
        description="List of midpoint entries (see `MidpointModel`).",
    )


class PrimaryDirectionsContextResponseModel(StatusResponseModel):
    """Response payload for the primary directions context endpoint."""

    context: str = Field(description="XML context string listing the primary directions.")
    directions: list[PrimaryDirectionModel] = Field(
        default_factory=list,
        description="Primary directions with promissor, significator, aspect, arc, and years.",
    )
    speculum: list[SpeculumEntryModel] = Field(
        default_factory=list,
        description="Speculum table with RA, declination, semi-arc, and pole for each point.",
    )


class ProgressionChartResponseModel(StatusResponseModel):
    """Response payload for the progression / solar-arc biwheel chart endpoints."""

    chart_wheel: str = Field(description="SVG of the biwheel (no aspect grid).")
    chart_grid: str = Field(description="SVG of the aspect grid (separate panel).")
    chart_data: DualChartDataModel = Field(description="Progression-type chart data.")


class TransitBatchEntryModel(BaseModel):
    """One per-date entry of a transit batch response."""

    date: str = Field(description="ISO date(time) of the transit moment.")
    chart_data: DualChartDataModel = Field(description="Transit chart data for the date.")


class TransitBatchResponseModel(StatusResponseModel):
    """Response payload for the batch transit endpoint."""

    results: list[TransitBatchEntryModel] = Field(
        default_factory=list,
        description="Per-date transit chart data, in chronological order.",
    )


class SolarArcDirectionsResponseModel(StatusResponseModel):
    """Response payload for solar arc directions calculation."""

    solar_arc_subject: SolarArcSubjectModel = Field(
        description=(
            "SolarArcSubjectModel — solar_arc (signed degrees), directed_points (natal vs directed positions, sign_changed flag), and directed_to_natal_aspects (the actionable timing list)."
        ),
    )


class AstroCartographyResponseModel(StatusResponseModel):
    """Response payload for astro-cartography (ACG) line calculation."""

    lines: list[ACGLineModel] = Field(
        default_factory=list,
        description="List of ACG lines with planet, line_type (ASC/DSC/MC/IC), and coordinate points.",
    )


class SunTimesModel(BaseModel):
    """Sunrise / sunset / solar-noon / day-length for a date + location."""

    date: str = Field(description="Civil date (YYYY-MM-DD) the times are computed for.")
    timezone: str = Field(description="IANA timezone the local strings are expressed in.")
    latitude: float = Field(description="Observer latitude in degrees.")
    longitude: float = Field(description="Observer longitude in degrees.")
    sunrise: Optional[str] = Field(default=None, description="Sunrise, ISO-8601 UTC.")
    sunrise_local: Optional[str] = Field(default=None, description="Sunrise local wall-clock time, HH:MM, rounded to the nearest minute; an instant in the last half-minute of the day clamps to 23:59 rather than wrapping to 00:00.")
    sunset: Optional[str] = Field(default=None, description="Sunset, ISO-8601 UTC.")
    sunset_local: Optional[str] = Field(default=None, description="Sunset local wall-clock time, HH:MM, rounded to the nearest minute (23:59-clamped, never wrapped). Above roughly 60 degrees of latitude the paired sunset can genuinely fall past local midnight, so this string can read EARLIER than sunrise_local — the ISO `sunset` field carries the real date; this field cannot.")
    solar_noon: Optional[str] = Field(default=None, description="Solar noon, ISO-8601 UTC.")
    solar_noon_local: Optional[str] = Field(default=None, description="Solar noon local wall-clock time, HH:MM, rounded to the nearest minute.")
    day_length: Optional[str] = Field(default=None, description="Day length, H:MM, rounded to the nearest minute.")
    is_polar_day: bool = Field(default=False, description="True when the Sun stays above the horizon all day.")
    is_polar_night: bool = Field(default=False, description="True when the Sun stays below the horizon all day.")
    civil_dawn: Optional[str] = Field(default=None, description="Civil dawn (Sun at -6 degrees), ISO-8601 UTC.")
    civil_dawn_local: Optional[str] = Field(default=None, description="Civil dawn, ISO-8601 in the request timezone.")
    civil_dusk: Optional[str] = Field(default=None, description="Civil dusk (Sun at -6 degrees), ISO-8601 UTC.")
    civil_dusk_local: Optional[str] = Field(default=None, description="Civil dusk, ISO-8601 in the request timezone (may fall on the next civil date).")
    nautical_dawn: Optional[str] = Field(default=None, description="Nautical dawn (Sun at -12 degrees), ISO-8601 UTC.")
    nautical_dawn_local: Optional[str] = Field(default=None, description="Nautical dawn, ISO-8601 in the request timezone.")
    nautical_dusk: Optional[str] = Field(default=None, description="Nautical dusk (Sun at -12 degrees), ISO-8601 UTC.")
    nautical_dusk_local: Optional[str] = Field(default=None, description="Nautical dusk, ISO-8601 in the request timezone (may fall on the next civil date).")
    astronomical_dawn: Optional[str] = Field(default=None, description="Astronomical dawn (Sun at -18 degrees), ISO-8601 UTC.")
    astronomical_dawn_local: Optional[str] = Field(default=None, description="Astronomical dawn, ISO-8601 in the request timezone.")
    astronomical_dusk: Optional[str] = Field(default=None, description="Astronomical dusk (Sun at -18 degrees), ISO-8601 UTC.")
    astronomical_dusk_local: Optional[str] = Field(default=None, description="Astronomical dusk, ISO-8601 in the request timezone (may fall on the next civil date).")


class SunTimesResponseModel(StatusResponseModel):
    """Response payload for the sun-times endpoint."""

    sun_times: SunTimesModel


class PlanetaryHourModel(BaseModel):
    """One planetary hour in the day's 24-hour Chaldean sequence."""

    index: int = Field(description="1-based position in the 24-hour sequence (1..24).")
    ruler: ClassicalPlanet = Field(description="Classical ruling planet of the hour.")
    is_day: bool = Field(description="True for the 12 day hours (sunrise→sunset), false for night.")
    start: str = Field(description="Hour start, ISO-8601 UTC.")
    end: str = Field(description="Hour end, ISO-8601 UTC.")


class PlanetaryHoursModel(BaseModel):
    """Planetary-hour state for a requested moment plus the full 24-hour table."""

    date: str = Field(description="Civil date of the planetary day's sunrise (YYYY-MM-DD).")
    timezone: str = Field(description="IANA timezone identifier.")
    latitude: float = Field(description="Observer latitude in degrees.")
    longitude: float = Field(description="Observer longitude in degrees.")
    day_ruler: ClassicalPlanet = Field(description="Planet ruling the whole day (by weekday).")
    current_ruler: ClassicalPlanet = Field(description="Ruler of the hour containing the requested moment.")
    current_index: int = Field(description="1-based index of the current hour.")
    current_is_day: bool = Field(description="Whether the current hour is a day hour.")
    sunrise: str = Field(description="Sunrise opening the day hours, ISO-8601 UTC.")
    sunset: str = Field(description="Sunset dividing day and night hours, ISO-8601 UTC.")
    next_sunrise: str = Field(description="Sunrise closing the night hours, ISO-8601 UTC.")
    hours: list[PlanetaryHourModel] = Field(min_length=24, max_length=24, description="All 24 planetary hours in order.")


class PlanetaryHoursResponseModel(StatusResponseModel):
    """Response payload for the planetary-hours endpoint."""

    planetary_hours: PlanetaryHoursModel


class MoonAspectEventModel(BaseModel):
    """A single exact aspect the Moon perfects to another body."""

    planet: str = Field(description="The body the Moon aspects (Sun, Mercury, …).")
    aspect: str = Field(description="Aspect name (conjunction, sextile, square, trine, opposition).")
    degrees: float = Field(description="Aspect angle in degrees (0, 60, 90, 120, 180).")
    time: str = Field(description="Exact aspect time, ISO-8601 UTC.")
    time_local: Optional[str] = Field(
        default=None,
        description="Exact aspect time, ISO-8601 in the request timezone (null when the request carries no timezone).",
    )


class MoonVocModel(BaseModel):
    """Void-of-course state for a moment: the Moon makes no further exact aspect
    before leaving its current sign during the void window."""

    is_void: bool = Field(description="True if the Moon is void of course at the requested moment.")
    moon_sign: str = Field(description="Sign the Moon currently occupies.")
    next_sign: str = Field(description="Sign the Moon ingresses into next.")
    ingress: str = Field(description="Time the Moon enters the next sign, ISO-8601 UTC.")
    ingress_local: str = Field(description="Ingress, ISO-8601 in the request timezone.")
    void_start: str = Field(description="Start of the void window (Moon's last in-sign aspect), ISO-8601 UTC.")
    void_start_local: str = Field(description="Void-window start, ISO-8601 in the request timezone.")
    void_end: str = Field(description="End of the void window (= ingress), ISO-8601 UTC.")
    void_end_local: str = Field(description="Void-window end, ISO-8601 in the request timezone.")
    last_aspect: Optional[MoonAspectEventModel] = Field(default=None, description="The Moon's last exact aspect before ingress (none if the whole sign is void).")
    next_aspect: Optional[MoonAspectEventModel] = Field(
        default=None,
        description="The Moon's first exact aspect after the ingress, in the next sign — the aspect that ends the void lull (none only if the Moon makes no aspect throughout the next sign).",
    )


class MoonVocResponseModel(StatusResponseModel):
    """Response payload for the void-of-course Moon endpoint."""

    moon_voc: MoonVocModel


class DominantsResponseModel(StatusResponseModel):
    """Response payload for the dominants endpoint."""

    dominants: DominantsModel = Field(
        description="Computed dominants: ranked planets, signs, elements, modalities and houses "
        "(plus polarity, hemispheres and quadrants for the modern method), the convenience "
        "'dominant_*' winners, and the optional per-rule 'score_breakdown'."
    )


class DeclinationAspectsResponseModel(StatusResponseModel):
    """Response payload for declination aspects calculation."""

    aspects: list[AspectModel] = Field(
        default_factory=list,
        description="List of declination aspects (parallel / contra-parallel).",
    )


class TransitEventsResponseModel(StatusResponseModel):
    """Response payload for transit events calculation."""

    events: list[TransitEventModel] = Field(
        default_factory=list,
        description="Transit events with applying start, exact moment, and separating end.",
    )
    subject: Optional[AstrologicalSubjectModel] = Field(
        default=None,
        description="Natal subject data.",
    )


class FixedStarsTruncationModel(BaseModel):
    """Declared truncation of the requested fixed-star list.

    When the requested stars would push the ephemeris past its work budget,
    the route serves the affordable prefix of the request (a deterministic
    cut, in the caller's own order) and says so here — clients never need to
    mirror the engine's cost model to pre-trim.
    """

    requested: int = Field(description="How many fixed stars the request carried.")
    served: int = Field(description="How many fit the work budget and were calculated.")
    dropped: list[str] = Field(
        default_factory=list,
        description="The star names that were dropped, in request order.",
    )


class EphemerisResponseModel(StatusResponseModel):
    """Response payload for ephemeris data generation."""

    ephemeris: list[EphemerisDictModel] = Field(
        default_factory=list,
        description="Ephemeris data points with planetary positions and house cusps.",
    )
    fixed_stars_truncated: Optional[FixedStarsTruncationModel] = Field(
        default=None,
        description=(
            "Present only when the requested active_fixed_stars exceeded the work "
            "budget: how many were requested/served and which were dropped."
        ),
    )


class TransitMomentsResponseModel(StatusResponseModel):
    """Response payload for transit moment snapshots."""

    transits: list[TransitMomentModel] = Field(
        default_factory=list,
        description=(
            "Transit moment snapshots with active aspects at each date. Each snapshot carries "
            "`subject` — the full transiting chart it was computed from — only when the request "
            "set `include_transit_subjects`."
        ),
    )
    subject: Optional[AstrologicalSubjectModel] = Field(default=None, description="Natal subject data.")
    dates: Optional[list[str]] = Field(default=None, description="ISO dates of transit moments.")


class ReportResponseModel(StatusResponseModel):
    """Response payload for text report generation."""

    report: str = Field(description="Generated text report for the astrological subject.")


class HeliocentricReturnChartResponseModel(ChartResponseModel):
    """Response payload for heliocentric return chart."""

    return_type: str = Field(description="Return type (e.g. 'Heliocentric').")
    planet: str = Field(description="Planet of the heliocentric return.")
    wheel_type: Literal["dual", "single"] = Field(description="Wheel configuration.")


class LunarNodeCrossingChartResponseModel(ChartResponseModel):
    """Response payload for lunar node crossing chart."""

    return_type: str = Field(default="Lunar_Node_Crossing", description="Return type.")
    wheel_type: Literal["dual", "single"] = Field(description="Wheel configuration.")
