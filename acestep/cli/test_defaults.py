"""Unit tests for acestep.cli.defaults."""

import argparse
import unittest

from acestep.cli.defaults import (
    apply_optional_defaults,
    default_instruction_for_task,
)
from acestep.inference import GenerationConfig, GenerationParams


class TestDefaultInstructionForTask(unittest.TestCase):
    """Tests for default_instruction_for_task."""

    def test_text2music_returns_default(self):
        result = default_instruction_for_task("text2music")
        self.assertIn("Fill the audio semantic mask", result)

    def test_lego_formats_track(self):
        result = default_instruction_for_task("lego", ["drums"])
        self.assertIn("DRUMS", result)

    def test_extract_formats_track(self):
        result = default_instruction_for_task("extract", ["vocals"])
        self.assertIn("VOCALS", result)

    def test_complete_formats_tracks(self):
        result = default_instruction_for_task("complete", ["bass", "guitar"])
        self.assertIn("bass", result)
        self.assertIn("guitar", result)

    def test_lego_fallback_track(self):
        result = default_instruction_for_task("lego")
        self.assertIn("GUITAR", result)


class TestApplyOptionalDefaults(unittest.TestCase):
    """Tests for apply_optional_defaults."""

    def test_fills_none_attributes(self):
        args = argparse.Namespace(duration=None, batch_size=None)
        params_defaults = GenerationParams()
        config_defaults = GenerationConfig()
        apply_optional_defaults(args, params_defaults, config_defaults)
        self.assertEqual(args.duration, params_defaults.duration)
        self.assertEqual(args.batch_size, config_defaults.batch_size)

    def test_preserves_existing_values(self):
        args = argparse.Namespace(duration=42.0, batch_size=5)
        apply_optional_defaults(args, GenerationParams(), GenerationConfig())
        self.assertEqual(args.duration, 42.0)
        self.assertEqual(args.batch_size, 5)


if __name__ == "__main__":
    unittest.main()
