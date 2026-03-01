"""Tests for acestep.generation_helpers."""

import unittest

from acestep.generation_helpers import (
    accumulate_lm_time_costs,
    build_user_metadata,
    format_seed_string,
    safe_parse_metadata_value,
)


class TestBuildUserMetadata(unittest.TestCase):
    """Tests for build_user_metadata."""

    def test_all_fields_populated(self):
        result = build_user_metadata(
            bpm=120, keyscale="C Major", timesignature="4/4", duration=180,
        )
        self.assertEqual(result, {
            "bpm": 120, "keyscale": "C Major",
            "timesignature": "4/4", "duration": 180,
        })

    def test_none_bpm_excluded(self):
        result = build_user_metadata(
            bpm=None, keyscale="Am", timesignature="3/4", duration=60,
        )
        self.assertNotIn("bpm", result)
        self.assertEqual(result["keyscale"], "Am")

    def test_zero_bpm_excluded(self):
        result = build_user_metadata(bpm=0, keyscale="", timesignature="", duration=None)
        self.assertIsNone(result)

    def test_negative_duration_excluded(self):
        result = build_user_metadata(bpm=120, keyscale="", timesignature="", duration=-1)
        self.assertEqual(result, {"bpm": 120})

    def test_na_keyscale_excluded(self):
        result = build_user_metadata(bpm=90, keyscale="N/A", timesignature="", duration=None)
        self.assertEqual(result, {"bpm": 90})

    def test_whitespace_keyscale_stripped(self):
        result = build_user_metadata(
            bpm=None, keyscale="  D Minor  ", timesignature="", duration=None,
        )
        self.assertEqual(result, {"keyscale": "D Minor"})

    def test_empty_fields_returns_none(self):
        result = build_user_metadata(bpm=None, keyscale="", timesignature="", duration=None)
        self.assertIsNone(result)

    def test_extras_merged(self):
        result = build_user_metadata(
            bpm=120, keyscale="", timesignature="", duration=None,
            extras={"caption": "My song", "language": "en"},
        )
        self.assertEqual(result, {"bpm": 120, "caption": "My song", "language": "en"})

    def test_extras_alone_returns_dict(self):
        result = build_user_metadata(
            bpm=None, keyscale="", timesignature="", duration=None,
            extras={"caption": "test"},
        )
        self.assertEqual(result, {"caption": "test"})

    def test_duration_truncated_to_int(self):
        result = build_user_metadata(bpm=None, keyscale="", timesignature="", duration=90.7)
        self.assertEqual(result, {"duration": 90})

    def test_unparseable_bpm_excluded(self):
        result = build_user_metadata(
            bpm="not-a-number", keyscale="C", timesignature="", duration=None,
        )
        self.assertEqual(result, {"keyscale": "C"})


class TestFormatSeedString(unittest.TestCase):
    """Tests for format_seed_string."""

    def test_none_returns_empty(self):
        self.assertEqual(format_seed_string(None), "")

    def test_single_int(self):
        self.assertEqual(format_seed_string(42), "42")

    def test_list_of_ints(self):
        self.assertEqual(format_seed_string([1, 2, 3]), "1,2,3")

    def test_empty_list_returns_empty(self):
        self.assertEqual(format_seed_string([]), "")

    def test_single_element_list(self):
        self.assertEqual(format_seed_string([99]), "99")


class TestAccumulateLmTimeCosts(unittest.TestCase):
    """Tests for accumulate_lm_time_costs."""

    def test_normal_accumulation(self):
        totals = {"phase1_time": 1.0, "phase2_time": 2.0, "total_time": 3.0}
        chunk = {
            "extra_outputs": {
                "time_costs": {
                    "phase1_time": 0.5, "phase2_time": 0.3, "total_time": 0.8,
                },
            },
        }
        accumulate_lm_time_costs(totals, chunk)
        self.assertAlmostEqual(totals["phase1_time"], 1.5)
        self.assertAlmostEqual(totals["phase2_time"], 2.3)
        self.assertAlmostEqual(totals["total_time"], 3.8)

    def test_missing_extra_outputs(self):
        totals = {"phase1_time": 1.0, "phase2_time": 2.0, "total_time": 3.0}
        accumulate_lm_time_costs(totals, {})
        self.assertAlmostEqual(totals["phase1_time"], 1.0)

    def test_empty_time_costs(self):
        totals = {"phase1_time": 0.0, "phase2_time": 0.0, "total_time": 0.0}
        accumulate_lm_time_costs(totals, {"extra_outputs": {"time_costs": {}}})
        self.assertAlmostEqual(totals["total_time"], 0.0)

    def test_none_extra_outputs(self):
        totals = {"phase1_time": 1.0, "phase2_time": 2.0, "total_time": 3.0}
        accumulate_lm_time_costs(totals, {"extra_outputs": None})
        self.assertAlmostEqual(totals["phase1_time"], 1.0)

    def test_total_time_derived_from_phases(self):
        """When total_time is absent, it should be derived from phase sums."""
        totals = {"phase1_time": 0.0, "phase2_time": 0.0, "total_time": 0.0}
        chunk = {
            "extra_outputs": {
                "time_costs": {"phase1_time": 1.0, "phase2_time": 2.0},
            },
        }
        accumulate_lm_time_costs(totals, chunk)
        self.assertAlmostEqual(totals["total_time"], 3.0)


class TestSafeParseMetadataValue(unittest.TestCase):
    """Tests for safe_parse_metadata_value."""

    def test_int_parse(self):
        result = safe_parse_metadata_value("bpm", {"bpm": "120"}, as_int=True)
        self.assertEqual(result, 120)
        self.assertIsInstance(result, int)

    def test_float_parse(self):
        result = safe_parse_metadata_value(
            "duration", {"duration": "90.5"}, as_float=True,
        )
        self.assertAlmostEqual(result, 90.5)
        self.assertIsInstance(result, float)

    def test_missing_key_returns_none(self):
        result = safe_parse_metadata_value("bpm", {}, as_int=True)
        self.assertIsNone(result)

    def test_empty_value_returns_none(self):
        result = safe_parse_metadata_value("bpm", {"bpm": ""}, as_int=True)
        self.assertIsNone(result)

    def test_zero_value_returns_none(self):
        result = safe_parse_metadata_value("bpm", {"bpm": "0"}, as_int=True)
        self.assertIsNone(result)

    def test_negative_value_returns_none(self):
        result = safe_parse_metadata_value(
            "duration", {"duration": "-5"}, as_float=True,
        )
        self.assertIsNone(result)

    def test_unparseable_returns_none(self):
        result = safe_parse_metadata_value(
            "bpm", {"bpm": "not-a-number"}, as_int=True,
        )
        self.assertIsNone(result)

    def test_raw_number_no_cast(self):
        """Without as_int or as_float, returns the raw parsed float."""
        result = safe_parse_metadata_value("bpm", {"bpm": "120"})
        self.assertAlmostEqual(result, 120.0)

    def test_numeric_input_converted_to_string(self):
        """Numeric dict values are stringified before parsing."""
        result = safe_parse_metadata_value("bpm", {"bpm": 130}, as_int=True)
        self.assertEqual(result, 130)


if __name__ == "__main__":
    unittest.main()
