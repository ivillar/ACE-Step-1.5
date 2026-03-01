"""Unit tests for acestep.cli.parsing."""

import unittest

from acestep.cli.parsing import (
    parse_bool,
    parse_description_hints,
    parse_number,
    parse_timesteps_input,
)


class TestParseDescriptionHints(unittest.TestCase):
    """Tests for parse_description_hints."""

    def test_empty_returns_defaults(self):
        lang, inst = parse_description_hints("")
        self.assertIsNone(lang)
        self.assertFalse(inst)

    def test_detects_english(self):
        lang, inst = parse_description_hints("An English pop song")
        self.assertEqual(lang, "en")
        self.assertFalse(inst)

    def test_detects_chinese(self):
        lang, _ = parse_description_hints("a 中文 ballad")
        self.assertEqual(lang, "zh")

    def test_detects_instrumental_keyword(self):
        _, inst = parse_description_hints("instrumental jazz")
        self.assertTrue(inst)

    def test_detects_pure_music(self):
        _, inst = parse_description_hints("Pure music ambient")
        self.assertTrue(inst)

    def test_detects_solo_suffix(self):
        _, inst = parse_description_hints("guitar solo")
        self.assertTrue(inst)

    def test_none_input(self):
        lang, inst = parse_description_hints(None)
        self.assertIsNone(lang)
        self.assertFalse(inst)


class TestParseNumber(unittest.TestCase):
    """Tests for parse_number."""

    def test_integer_string(self):
        self.assertEqual(parse_number("120"), 120.0)

    def test_float_string(self):
        self.assertAlmostEqual(parse_number("3.14"), 3.14)

    def test_embedded_number(self):
        self.assertAlmostEqual(parse_number("bpm is 128 ok"), 128.0)

    def test_no_number(self):
        self.assertIsNone(parse_number("hello"))

    def test_negative_number(self):
        self.assertAlmostEqual(parse_number("-5.5"), -5.5)


class TestParseTimestepsInput(unittest.TestCase):
    """Tests for parse_timesteps_input."""

    def test_none_returns_none(self):
        self.assertIsNone(parse_timesteps_input(None))

    def test_empty_string(self):
        self.assertIsNone(parse_timesteps_input(""))

    def test_list_of_floats(self):
        self.assertEqual(parse_timesteps_input([0.97, 0.5, 0.0]), [0.97, 0.5, 0.0])

    def test_list_of_ints(self):
        self.assertEqual(parse_timesteps_input([1, 0]), [1.0, 0.0])

    def test_bracket_string(self):
        self.assertEqual(parse_timesteps_input("[0.97, 0.5, 0]"), [0.97, 0.5, 0.0])

    def test_comma_separated(self):
        self.assertEqual(parse_timesteps_input("0.97,0.5,0"), [0.97, 0.5, 0.0])

    def test_invalid_string(self):
        self.assertIsNone(parse_timesteps_input("not-a-list"))

    def test_mixed_type_list(self):
        self.assertIsNone(parse_timesteps_input(["a", "b"]))


class TestParseBool(unittest.TestCase):
    """Tests for parse_bool."""

    def test_truthy_values(self):
        for val in ("true", "True", "1", "yes", "y", "Y"):
            self.assertTrue(parse_bool(val), msg=val)

    def test_falsy_values(self):
        for val in ("false", "0", "no", "n", "anything"):
            self.assertFalse(parse_bool(val), msg=val)


if __name__ == "__main__":
    unittest.main()
