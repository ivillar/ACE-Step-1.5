"""Tests for noise schedule functions."""

import unittest

from acestep.core.generation.samplers.schedules import (
    cosine_schedule,
    linear_schedule,
    logsnr_schedule,
)


class TestLinearSchedule(unittest.TestCase):
    """Verify linear_schedule produces correct descending timestep sequences."""

    def test_length(self):
        result = linear_schedule(10)
        self.assertEqual(len(result), 10)

    def test_starts_at_one(self):
        result = linear_schedule(8)
        self.assertAlmostEqual(result[0], 1.0, places=7)

    def test_descending(self):
        result = linear_schedule(20)
        for a, b in zip(result[:-1], result[1:]):
            self.assertGreater(a, b)

    def test_no_shift(self):
        result = linear_schedule(4, shift=1.0)
        expected = [1.0, 0.75, 0.5, 0.25]
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, places=7)

    def test_with_shift(self):
        result = linear_schedule(4, shift=3.0)
        self.assertAlmostEqual(result[0], 1.0, places=7)
        for v in result:
            self.assertGreater(v, 0.0)
            self.assertLessEqual(v, 1.0)


class TestCosineSchedule(unittest.TestCase):
    """Verify cosine_schedule (HF CosineDPMSolverMultistepScheduler–based) behavior."""

    def test_length(self):
        result = cosine_schedule(10)
        self.assertEqual(len(result), 10)

    def test_starts_at_one(self):
        result = cosine_schedule(8)
        self.assertAlmostEqual(result[0], 1.0, places=5)

    def test_descending(self):
        result = cosine_schedule(20)
        for a, b in zip(result[:-1], result[1:]):
            self.assertGreater(a, b)

    def test_values_in_range(self):
        """All timesteps in (0, 1] and last step > 0."""
        result = cosine_schedule(20)
        for v in result:
            self.assertGreater(v, 0.0)
            self.assertLessEqual(v, 1.0)
        self.assertGreater(result[-1], 0.0)

    def test_deterministic(self):
        """Same inputs produce same schedule."""
        a = cosine_schedule(8, shift=1.0)
        b = cosine_schedule(8, shift=1.0)
        self.assertEqual(a, b)

    def test_with_shift(self):
        """Shift transformation applies; schedule remains descending and in range."""
        result = cosine_schedule(8, shift=3.0)
        self.assertEqual(len(result), 8)
        for v in result:
            self.assertGreater(v, 0.0)
            self.assertLessEqual(v, 1.0)
        for i in range(len(result) - 1):
            self.assertGreater(result[i], result[i + 1])


class TestLogSNRSchedule(unittest.TestCase):
    """Verify logsnr_schedule produces valid descending timesteps."""

    def test_length(self):
        result = logsnr_schedule(10)
        self.assertEqual(len(result), 10)

    def test_starts_at_sigma_max(self):
        result = logsnr_schedule(10, sigma_max=1.0)
        self.assertAlmostEqual(result[0], 1.0, places=7)

    def test_descending(self):
        result = logsnr_schedule(20)
        for a, b in zip(result[:-1], result[1:]):
            self.assertGreater(a, b)

    def test_all_positive(self):
        result = logsnr_schedule(20, sigma_max=0.9)
        for v in result:
            self.assertGreater(v, 0.0)


if __name__ == "__main__":
    unittest.main()
