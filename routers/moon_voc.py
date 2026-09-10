"""
Void-of-course Moon endpoint.

Thin REST wrapper over kerykeion's :class:`VoidOfCourseMoonFactory`: the whole
algorithm (sign ingress, exact-aspect search, void window) lives in kerykeion;
this handler only adapts the request and serialises the resulting model.
"""

from logging import getLogger
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from kerykeion import VoidOfCourseMoonFactory
from kerykeion.schemas import (
    VoidOfCourseAspectModel,
    VoidOfCourseMoonModel,
    VoidOfCourseWindowModel,
)

from ..types.request_models import MoonVocRequestModel, MoonVocWindowsRequestModel
from ..types.response_models import MoonVocResponseModel, MoonVocWindowsResponseModel
from ..utils.logging_utils import log_request_with_body
from ..utils.router_utils import handle_exception, iso_utc, local_iso, run_heavy

logger = getLogger(__name__)

router = APIRouter()


def _aspect_payload(aspect: Optional[VoidOfCourseAspectModel], tz: Optional[ZoneInfo]) -> Optional[dict]:
    """Serialise an aspect event to the API JSON shape, or ``None``.

    ``time_local`` is ``None`` when no timezone is available (the windows range
    endpoint makes the timezone optional; the single-moment endpoint always has one).
    """
    if aspect is None:
        return None
    return {
        "planet": aspect.planet,
        "aspect": aspect.aspect,
        "degrees": aspect.aspect_degrees,
        "time": iso_utc(aspect.exact_time),
        "time_local": local_iso(aspect.exact_time, tz) if tz else None,
    }


def _moon_voc_payload(model: VoidOfCourseMoonModel, tz: ZoneInfo) -> dict:
    """Serialise a kerykeion ``VoidOfCourseMoonModel`` to the API JSON shape."""
    return {
        "status": "OK",
        "moon_voc": {
            "is_void": model.is_void_of_course,
            "moon_sign": model.moon_sign,
            "next_sign": model.next_sign,
            "ingress": iso_utc(model.ingress),
            "ingress_local": local_iso(model.ingress, tz),
            "void_start": iso_utc(model.void_start),
            "void_start_local": local_iso(model.void_start, tz),
            "void_end": iso_utc(model.void_end),
            "void_end_local": local_iso(model.void_end, tz),
            "last_aspect": _aspect_payload(model.last_aspect, tz),
            "next_aspect": _aspect_payload(model.next_aspect, tz),
        },
    }


@router.post("/api/v6/moon-voc", response_model=MoonVocResponseModel)
async def moon_voc(request_body: MoonVocRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/moon-voc`

    Void-of-course Moon for a moment, computed by kerykeion's
    ``VoidOfCourseMoonFactory``. The Moon is *void of course* once it has perfected
    its last exact Ptolemaic aspect (conjunction, sextile, square, trine,
    opposition) to a traditional planet (Sun, Mercury, Venus, Mars, Jupiter,
    Saturn) while in its current sign, and stays void until it ingresses the next
    sign. The result is geocentric, so no location is required.

    **Returns:** `status` + `moon_voc` { is_void, moon_sign, next_sign, ingress,
    void_start, void_end, last_aspect, next_aspect } (signs are three-letter codes).
    """
    log_request_with_body(logger, request, "Moon void-of-course request", request_body.model_dump_json())
    try:
        model = await run_heavy(
            VoidOfCourseMoonFactory.from_datetime,
            request_body.year,
            request_body.month,
            request_body.day,
            request_body.hour,
            request_body.minute,
            tz_str=request_body.timezone,
            zodiac_type=request_body.zodiac_type,
            sidereal_mode=request_body.sidereal_mode,
        )
        tz = ZoneInfo(request_body.timezone)
        return JSONResponse(content=_moon_voc_payload(model, tz), status_code=200)
    except Exception as exc:  # pragma: no cover - defensive
        return await handle_exception(exc, request)


def _window_payload(window: VoidOfCourseWindowModel, tz: Optional[ZoneInfo]) -> dict:
    """Serialise a void window to the API JSON shape (local fields only with a tz)."""
    return {
        "moon_sign": window.moon_sign,
        "next_sign": window.next_sign,
        "void_start": iso_utc(window.void_start),
        "void_start_local": local_iso(window.void_start, tz) if tz else None,
        "void_end": iso_utc(window.void_end),
        "void_end_local": local_iso(window.void_end, tz) if tz else None,
        "duration_minutes": window.duration_minutes,
        "last_aspect": _aspect_payload(window.last_aspect, tz),
    }


@router.post("/api/v6/advanced/moon-voc-windows", response_model=MoonVocWindowsResponseModel)
async def moon_voc_windows(request_body: MoonVocWindowsRequestModel, request: Request) -> JSONResponse:
    """
    **POST** `/api/v6/advanced/moon-voc-windows`

    Every void-of-course Moon window intersecting a date range, computed by
    kerykeion's ``VoidOfCourseMoonFactory.from_iso_range``. Each window runs from
    the Moon's last exact Ptolemaic aspect in a sign to its ingress into the next
    sign (~13-14 windows per month). Windows are **unclipped**: the first may
    start before `start_date` and the last may end after `end_date`.

    **Parameters:**
    - `start_date`, `end_date`: ISO date(time) range (treated as UTC, max ~1 year).
    - `timezone`: Optional IANA zone — adds `*_local` ISO strings to each window.
    - `zodiac_type`, `sidereal_mode`: Zodiac (sign boundaries shift when sidereal).

    **Returns:** `status` + `windows[]` { moon_sign, next_sign, void_start(_local),
    void_end(_local), duration_minutes, last_aspect }.
    """
    log_request_with_body(logger, request, "Moon VoC windows request", request_body.model_dump_json())
    try:
        result = await run_heavy(
            VoidOfCourseMoonFactory.from_iso_range,
            request_body.start_date,
            request_body.end_date,
            zodiac_type=request_body.zodiac_type,
            sidereal_mode=request_body.sidereal_mode,
        )
        tz = ZoneInfo(request_body.timezone) if request_body.timezone else None
        return JSONResponse(
            content={
                "status": "OK",
                "windows": [_window_payload(w, tz) for w in result.windows],
            },
            status_code=200,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return await handle_exception(exc, request)
