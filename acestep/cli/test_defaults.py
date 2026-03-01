"""Unit tests for acestep.cli.defaults."""

import argparse
import unittest

from acestep.cli.defaults import (
    apply_optional_defaults,
    build_all_defaults,
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


class TestBuildAllDefaults(unittest.TestCase):
    """Tests for build_all_defaults."""

    def test_returns_all_expected_keys(self):
        result = build_all_defaults(GenerationParams(), GenerationConfig())
        expected_keys = {
            "duration", "bpm", "keyscale", "timesignature", "vocal_language",
            "inference_steps", "seed", "guidance_scale", "use_adg",
            "cfg_interval_start", "cfg_interval_end", "shift", "infer_method",
            "noise_schedule", "timesteps", "repainting_start", "repainting_end",
            "audio_cover_strength", "thinking", "lm_temperature",
            "lm_cfg_scale", "lm_top_k", "lm_top_p", "lm_negative_prompt",
            "use_cot_metas", "use_cot_caption", "use_cot_lyrics",
            "use_cot_language", "use_constrained_decoding",
            "batch_size", "allow_lm_batch", "use_random_seed", "seeds",
            "lm_batch_chunk_size", "constrained_decoding_debug",
            "audio_format", "sample_mode", "sample_query", "use_format",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_values_match_dataclass_defaults(self):
        params = GenerationParams()
        config = GenerationConfig()
        result = build_all_defaults(params, config)
        self.assertEqual(result["duration"], params.duration)
        self.assertEqual(result["batch_size"], config.batch_size)
        self.assertEqual(result["audio_format"], config.audio_format)
        self.assertFalse(result["sample_mode"])


class TestGenerationParamsFromNamespace(unittest.TestCase):
    """Tests for GenerationParams.from_namespace."""

    def test_picks_known_fields(self):
        ns = argparse.Namespace(
            caption="a pop song", duration=60.0, bpm=120,
            irrelevant_key="ignored",
        )
        params = GenerationParams.from_namespace(ns)
        self.assertEqual(params.caption, "a pop song")
        self.assertEqual(params.duration, 60.0)
        self.assertEqual(params.bpm, 120)

    def test_ignores_unknown_fields(self):
        ns = argparse.Namespace(totally_fake=True)
        params = GenerationParams.from_namespace(ns)
        self.assertFalse(hasattr(params, "totally_fake"))

    def test_uses_dataclass_defaults_for_missing(self):
        ns = argparse.Namespace(caption="hello")
        params = GenerationParams.from_namespace(ns)
        defaults = GenerationParams()
        self.assertEqual(params.seed, defaults.seed)
        self.assertEqual(params.inference_steps, defaults.inference_steps)


class TestGenerationConfigFromNamespace(unittest.TestCase):
    """Tests for GenerationConfig.from_namespace."""

    def test_picks_known_fields(self):
        ns = argparse.Namespace(batch_size=4, audio_format="wav", extra="ignored")
        config = GenerationConfig.from_namespace(ns)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.audio_format, "wav")

    def test_uses_defaults_for_missing(self):
        ns = argparse.Namespace()
        config = GenerationConfig.from_namespace(ns)
        defaults = GenerationConfig()
        self.assertEqual(config.batch_size, defaults.batch_size)


if __name__ == "__main__":
    unittest.main()
