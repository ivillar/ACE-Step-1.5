"""Noise schedules for diffusion sampling.

Provides schedule functions that map a number of steps to a descending
sequence of timestep values in (0, 1].  All schedules return the
timestep list *without* the trailing 0 — the caller handles the final
step separately.

The cosine schedule uses the exponential sigma schedule from Hugging Face's
``CosineDPMSolverMultistepScheduler`` (as used in Stable Audio Open),
mapped to rectified-flow ``t`` where ``t=1`` is pure noise and ``t=0`` is
clean data.
"""

import math


def _compute_exponential_sigmas(
    num_steps: int,
    sigma_min: float = 0.3,
    sigma_max: float = 500.0,
) -> list[float]:
    """Exponential sigma schedule matching HF CosineDPMSolverMultistepScheduler.

    Log-linear in sigma: sigmas from sigma_max down to sigma_min (high to low).
    Implementation follows diffusers EDM/k-diffusion style.
    """
    if num_steps <= 0:
        return []
    if num_steps == 1:
        return [sigma_max]
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    sigmas = [
        math.exp(log_min + (log_max - log_min) * i / (num_steps - 1))
        for i in range(num_steps)
    ]
    return list(reversed(sigmas))


def _cosine_hf_schedule(
    num_steps: int,
    shift: float = 1.0,
    sigma_schedule: str = "exponential",
    sigma_min: float = 0.3,
    sigma_max: float = 500.0,
) -> list[float]:
    """Build a t-schedule from HF-style exponential sigma schedule.

    Uses the same exponential sigma schedule as Hugging Face's
    CosineDPMSolverMultistepScheduler (Stable Audio Open), then maps
    sigma -> t for rectified flow: t = (sigma - sigma_min) / (sigma_max - sigma_min),
    and optionally applies the shift transformation.

    Args:
        num_steps: Number of diffusion steps.
        shift: When != 1.0, applies shift to t after mapping.
        sigma_schedule: "exponential" (default) or "karras"; only exponential is implemented.
        sigma_min: Minimum sigma (HF default 0.3).
        sigma_max: Maximum sigma (HF default 500).

    Returns:
        Descending list of num_steps timestep values in (0, 1].
    """
    if sigma_schedule != "exponential":
        raise ValueError(
            f"sigma_schedule must be 'exponential', got {sigma_schedule!r}"
        )
    sigmas = _compute_exponential_sigmas(num_steps, sigma_min, sigma_max)
    sigma_range = sigma_max - sigma_min
    t_list: list[float] = []
    for sigma_val in sigmas:
        t_val = (
            (sigma_val - sigma_min) / sigma_range
            if sigma_range > 0
            else 1.0
        )
        t_val = max(1e-7, min(1.0, t_val))
        if shift != 1.0 and t_val > 0:
            t_val = shift * t_val / (1.0 + (shift - 1.0) * t_val)
        t_list.append(t_val)
    return t_list


def linear_schedule(num_steps: int, shift: float = 1.0) -> list[float]:
    """Uniform linspace schedule with optional shift transformation.

    Args:
        num_steps: Number of diffusion steps.
        shift: When != 1.0, applies ``t = shift * t / (1 + (shift - 1) * t)``.

    Returns:
        Descending list of ``num_steps`` timestep values in (0, 1].
    """
    raw = [1.0 - i / num_steps for i in range(num_steps)]
    if shift != 1.0:
        raw = [shift * t / (1.0 + (shift - 1.0) * t) for t in raw]
    return raw


def cosine_schedule(num_steps: int, shift: float = 1.0) -> list[float]:
    """Cosine schedule for rectified-flow diffusion sampling.

    Uses the exponential sigma schedule of Hugging Face's
    ``CosineDPMSolverMultistepScheduler`` (as used in Stable Audio Open).
    Sigmas are mapped linearly to rectified-flow ``t`` in [0, 1] so that
    high sigma (noise) corresponds to t=1 and low sigma to t=0; then the
    optional shift transformation is applied.

    Args:
        num_steps: Number of diffusion steps.
        shift: When != 1.0, applies ``t = shift * t / (1 + (shift - 1) * t)``.

    Returns:
        Descending list of ``num_steps`` timestep values in (0, 1].
    """
    return _cosine_hf_schedule(
        num_steps,
        shift=shift,
        sigma_schedule="exponential",
        sigma_min=0.3,
        sigma_max=500.0,
    )


def logsnr_schedule(num_steps: int, sigma_max: float = 1.0) -> list[float]:
    """Log-SNR-uniform schedule (used by Stable Audio Open for RF models).

    Spaces timesteps uniformly in log-SNR space between ``sigma_max`` and 0,
    then converts back to the ``t`` parameterisation via the logistic sigmoid.

    Args:
        num_steps: Number of diffusion steps.
        sigma_max: Maximum sigma (typically 1.0 for rectified flow).

    Returns:
        Descending list of ``num_steps`` timestep values in (0, 1].
    """
    logsnr_max = (
        math.log((1.0 - sigma_max) / sigma_max + 1e-6) if sigma_max < 1.0 else -6.0
    )
    logsnr_min = 2.0
    logsnr_vals = [
        logsnr_max + (logsnr_min - logsnr_max) * i / num_steps
        for i in range(num_steps)
    ]
    t_vals = [1.0 / (1.0 + math.exp(v)) for v in logsnr_vals]
    t_vals[0] = sigma_max
    return t_vals
