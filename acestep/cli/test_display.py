"""Unit tests for acestep.cli.display."""

import unittest

from acestep.cli.display import build_meta_dict, summarize_lyrics
from acestep.inference import GenerationParams


class TestSummarizeLyrics(unittest.TestCase):
    """Tests for summarize_lyrics."""

    def test_none(self):
        self.assertEqual(summarize_lyrics(None), "none")

    def test_empty_string(self):
        self.assertEqual(summarize_lyrics(""), "none")

    def test_whitespace(self):
        self.assertEqual(summarize_lyrics("   "), "none")

    def test_short_text(self):
        self.assertEqual(summarize_lyrics("La la la"), "La la la")

    def test_long_text(self):
        long = "x" * 100
        result = summarize_lyrics(long)
        self.assertIn("100 chars", result)

    def test_multiline_collapsed(self):
        self.assertEqual(summarize_lyrics("line1\nline2"), "line1 line2")


class TestBuildMetaDict(unittest.TestCase):
    """Tests for build_meta_dict."""

    def test_empty_params(self):
        params = GenerationParams(bpm=None, timesignature=None, keyscale=None, duration=None)
        self.assertIsNone(build_meta_dict(params))

    def test_partial_params(self):
        params = GenerationParams(bpm=120, keyscale="C major")
        meta = build_meta_dict(params)
        self.assertEqual(meta["bpm"], 120)
        self.assertEqual(meta["keyscale"], "C major")

    def test_all_params(self):
        params = GenerationParams(
            bpm=90, keyscale="Am", timesignature="3/4", duration=60.0,
        )
        meta = build_meta_dict(params)
        self.assertEqual(len(meta), 4)


if __name__ == "__main__":
    unittest.main()
