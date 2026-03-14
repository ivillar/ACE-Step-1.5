"""Backward-compatible shim — schedules moved to acestep.models.dit.schedules."""

from acestep.models.dit.schedules import (  # noqa: F401
    cosine_schedule,
    linear_schedule,
    logsnr_schedule,
)
