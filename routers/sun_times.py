"""
Sun-times and planetary-hours endpoints.

Thin REST wrappers over the kerykeion computation engine:
  - ``/api/v6/sun-times``        → :class:`kerykeion.SunTimesFactory`
  - ``/api/v6/planetary-hours``  → :class:`kerykeion.PlanetaryHoursFactory`

All astronomy (sunrise/sunset via the ephemeris backend, Chaldean hour division)
lives in kerykeion; these handlers only adapt the request, call the factory, and
serialise the resulting model to the API's JSON shape.
"""

from datetime import timedelta
from logging import getLogger
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from kerykeion import PlanetaryHoursFactory, SunTimesFactory
from kerykeion.schemas import (
    PlanetaryHoursModel as KrPlanetaryHoursModel,
    SunTimesModel as KrSunTimesModel,
)

from ..types.request_models import PlanetaryHoursRequestModel, SunTimesRequestModel
from ..types.response_models import PlanetaryHoursResponseModel, SunTimesResponseModel
from ..utils.logging_utils import log_request_with_body
from ..utils.router_utils import handle_exception, iso_utc, iso_utc_opt, local_hm, local_iso, run_heavy

logger = getLogger(__name__)

router = APIRouter()


def _duration_hm(span: Optional[timedelta]) -> Optional[str]:
    """Format a duration as ``H:MM``, rounded to the nearest minute, or ``None``.

    Rounded rather than floored, for the same reason as :func:`local_hm`: a day
    length of 16:19:52 is 16:20, and truncating it lost up to 59 s on about half
    of all values. ``int(x + 0.5)`` rather than ``round()`` because the latter
    rounds half to even, which would send 16:19:30 down to 16:19 while sending
    16:20:30 up to 16:21 — a rule no almanac uses and nobody would predict.
    Durations here are non-negative, so the shift is unambiguous.
    """
    if span is None:
        return None
    total_minutes = int(span.total_seconds() / 60.0 + 0.5)
    return f"{total_minutes // 60}:{total_minutes % 60:02d}"


def _sun_times_payload(model: KrSunTimesModel) -> dict:
    """Serialise a kerykeion ``SunTimesModel`` to the API JSON shape."""
    tz = ZoneInfo(model.timezone)
    return {
        "status": "OK",
        "sun_times": {
            "date": model.date,
            "timezone": model.timezone,
            "latitude": model.latitude,
            "longitude": model.longitude,
            "sunrise": iso_utc_opt(model.sunrise),
            "sunrise_local": local_hm(model.sunrise, tz),
            "sunset": iso_utc_opt(model.sunset),
            "sunset_local": local_hm(model.sunset, tz),
            "solar_noon": iso_utc_opt(model.solar_noon),
            "solar_noon_local": local_hm(model.solar_noon, tz),
            "day_length": _duration_hm(model.day_length),
            "is_polar_day": model.is_polar_day,
            "is_polar_night": model.is_polar_night,
            # Twilight dusk can fall in the small hours of the next civil date
            # (evening crossing), so the local strings use local_iso (date + time),
            # unlike the same-day sunrise/sunset which stay HH:MM.
            "civil_dawn": iso_utc_opt(model.civil_dawn),
            "civil_dawn_local": local_iso(model.civil_dawn, tz) if model.civil_dawn else None,
            "civil_dusk": iso_utc_opt(model.civil_dusk),
            "civil_dusk_local": local_iso(model.civil_dusk, tz) if model.civil_dusk else None,
            "nautical_dawn": iso_utc_opt(model.nautical_dawn),
            "nautical_dawn_local": local_iso(model.nautical_dawn, tz) if model.nautical_dawn else None,
            "nautical_dusk": iso_utc_opt(model.nautical_dusk),
            "nautical_dusk_local": local_iso(model.nautical_dusk, tz) if model.nautical_dusk else None,
            "astronomical_dawn": iso_utc_opt(model.astronomical_dawn),
            "astronomical_dawn_local": local_iso(model.astronomical_dawn, tz) if model.astronomical_dawn else None,
            "astronomical_dusk": iso_utc_opt(model.astronomical_dusk),
            "astronomical_dusk_local": local_iso(model.astronomical_dusk, tz) if model.astronomical_dusk else None,
        },
    }


def _planetary_hours_payload(model: KrPlanetaryHoursModel) -> dict:
    """Serialise a kerykeion ``PlanetaryHoursModel`` to the API JSON shape."""
    idx = model.current_index - 1
    return {
        "status": "OK",
        "planetary_hours": {
            "date": model.date,
            "timezone": model.timezone,
            "latitude": model.latitude,
            "longitude": model.longitude,
            "day_ruler": model.day_ruler,
            "current_index": model.current_index,
            "current_ruler": model.current_ruler,
            "current_is_day": model.hours[idx].is_diurnal if 0 <= idx < len(model.hours) else False,
            "sunrise": iso_utc(model.sunrise),
            "sunset": iso_utc(model.sunset),
            "next_sunrise": iso_utc(model.next_sunrise),
            "hours": [
                {
                    "index": hour.index,
                    "ruler": hour.ruler,
                    "is_day": hour.is_diurnal,
                    "start": iso_utc(hour.start),
                    "end": iso_utc(hour.end),
                }
                for hour in model.hours
            ],
        },
    }


@router.post("/api/v6/sun-times", response_model=SunTimesResponseModel)
async def sun_times(request_body: SunTimesRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/sun-times`

    Sunrise, sunset, solar noon and day length for a civil date at a location,
    computed by kerykeion's ``SunTimesFactory`` (apparent, refracted upper limb).
    Civil / nautical / astronomical twilight (Sun at -6 / -12 / -18 degrees) is
    also reported, geometric (no refraction).

    **Returns:** `status` + `sun_times` { date, timezone, latitude, longitude,
    sunrise, sunrise_local, sunset, sunset_local, solar_noon, solar_noon_local,
    day_length, is_polar_day, is_polar_night, civil_dawn(_local),
    civil_dusk(_local), nautical_dawn(_local), nautical_dusk(_local),
    astronomical_dawn(_local), astronomical_dusk(_local) }. Rise/set ``_local``
    strings are HH:MM; twilight ``_local`` strings are full ISO-8601, since an
    evening dusk can fall on the next civil date. A field is null when the event
    does not occur: rise/set on polar day/night, and twilight on polar day —
    during polar night the deeper nautical/astronomical twilight can still be
    present while civil is null.
    """
    log_request_with_body(logger, request, "Sun times request", request_body.model_dump_json())
    try:
        model = await run_heavy(
            SunTimesFactory.from_date,
            request_body.year,
            request_body.month,
            request_body.day,
            latitude=request_body.latitude,
            longitude=request_body.longitude,
            tz_str=request_body.timezone,
        )
        return JSONResponse(content=_sun_times_payload(model), status_code=200)
    except Exception as exc:  # pragma: no cover - defensive
        return await handle_exception(exc, request)


@router.post("/api/v6/planetary-hours", response_model=PlanetaryHoursResponseModel)
async def planetary_hours(request_body: PlanetaryHoursRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/planetary-hours`

    The 24 Chaldean planetary hours for the planetary day containing the requested
    moment, computed by kerykeion's ``PlanetaryHoursFactory``. Day and night are
    each divided into twelve unequal hours (sunrise→sunset, sunset→next sunrise);
    the first hour is ruled by the weekday's planet, then the Chaldean order cycles.

    **Returns:** `status` + `planetary_hours` { date, timezone, latitude, longitude,
    day_ruler, current_index, current_ruler, current_is_day, sunrise, sunset,
    next_sunrise, hours[] }.
    """
    log_request_with_body(logger, request, "Planetary hours request", request_body.model_dump_json())
    try:
        model = await run_heavy(
            PlanetaryHoursFactory.from_datetime,
            request_body.year,
            request_body.month,
            request_body.day,
            request_body.hour,
            request_body.minute,
            latitude=request_body.latitude,
            longitude=request_body.longitude,
            tz_str=request_body.timezone,
        )
        return JSONResponse(content=_planetary_hours_payload(model), status_code=200)
    except Exception as exc:  # pragma: no cover - defensive
        return await handle_exception(exc, request)
