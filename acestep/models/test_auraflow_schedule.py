"""Tests for the AuraFlow continuous timestep schedule generation."""

import unittest


def _time_snr_shift(alpha: float, t: float) -> float:
    """Reference implementation of the time_snr_shift function."""
    if alpha == 1.0:
        return t
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def _build_auraflow_schedule(
    num_steps: int,
    shift: float = 3.0,
) -> list[float]:
    """Build an AuraFlow-style continuous flow-matching timestep schedule.

    Replicates the schedule logic added to ``generate_audio`` for
    ``infer_method="auraflow"``.

    Args:
        num_steps: Number of diffusion steps.
        shift: Timestep shift factor.

    Returns:
        Monotonically decreasing list of timestep values in (0, 1].
    """
    raw = [1.0 - i / num_steps for i in range(num_steps)]
    if shift != 1.0:
        raw = [shift * t / (1.0 + (shift - 1.0) * t) for t in raw]
    return raw


class TestTimeSNRShift(unittest.TestCase):
    """Verify the time_snr_shift warping function."""

    def test_identity_when_alpha_one(self):
        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            self.assertAlmostEqual(_time_snr_shift(1.0, t), t)

    def test_known_values_shift_173(self):
        result = _time_snr_shift(1.73, 0.5)
        expected = 1.73 * 0.5 / (1.0 + 0.73 * 0.5)
        self.assertAlmostEqual(result, expected, places=10)

    def test_known_values_shift_3(self):
        result = _time_snr_shift(3.0, 0.5)
        expected = 3.0 * 0.5 / (1.0 + 2.0 * 0.5)
        self.assertAlmostEqual(result, expected, places=10)

    def test_boundary_at_zero(self):
        self.assertAlmostEqual(_time_snr_shift(3.0, 0.0), 0.0)

    def test_boundary_at_one(self):
        self.assertAlmostEqual(_time_snr_shift(3.0, 1.0), 1.0)
        self.assertAlmostEqual(_time_snr_shift(1.73, 1.0), 1.0)


class TestBuildAuraflowSchedule(unittest.TestCase):
    """Verify the AuraFlow continuous schedule construction."""

    def test_correct_length(self):
        for n in [1, 4, 8, 16, 32]:
            schedule = _build_auraflow_schedule(n, shift=3.0)
            self.assertEqual(len(schedule), n)

    def test_monotonically_decreasing(self):
        schedule = _build_auraflow_schedule(8, shift=3.0)
        for i in range(len(schedule) - 1):
            self.assertGreater(
                schedule[i], schedule[i + 1],
                f"Schedule not decreasing at index {i}: {schedule[i]} <= {schedule[i + 1]}",
            )

    def test_starts_at_one(self):
        for s in [1.0, 1.73, 3.0]:
            schedule = _build_auraflow_schedule(8, shift=s)
            self.assertAlmostEqual(schedule[0], 1.0)

    def test_all_values_positive(self):
        schedule = _build_auraflow_schedule(8, shift=3.0)
        for val in schedule:
            self.assertGreater(val, 0.0)

    def test_shift_one_produces_linear(self):
        schedule = _build_auraflow_schedule(8, shift=1.0)
        expected = [1.0 - i / 8 for i in range(8)]
        for actual, exp in zip(schedule, expected):
            self.assertAlmostEqual(actual, exp, places=10)

    def test_shift_applies_snr_warping(self):
        shift = 3.0
        n = 8
        schedule = _build_auraflow_schedule(n, shift=shift)
        raw = [1.0 - i / n for i in range(n)]
        expected = [_time_snr_shift(shift, t) for t in raw]
        for actual, exp in zip(schedule, expected):
            self.assertAlmostEqual(actual, exp, places=10)

    def test_single_step(self):
        schedule = _build_auraflow_schedule(1, shift=3.0)
        self.assertEqual(len(schedule), 1)
        self.assertAlmostEqual(schedule[0], 1.0)

    def test_shift_173_matches_classic_auraflow(self):
        """Verify shift=1.73 gives a gentler schedule than shift=3."""
        sched_173 = _build_auraflow_schedule(8, shift=1.73)
        sched_3 = _build_auraflow_schedule(8, shift=3.0)
        # Higher shift pushes more timesteps toward 1.0 (slower initial descent).
        # At index 4 (midpoint), shift=3 should be higher than shift=1.73.
        self.assertGreater(sched_3[4], sched_173[4])


if __name__ == "__main__":
    unittest.main()
