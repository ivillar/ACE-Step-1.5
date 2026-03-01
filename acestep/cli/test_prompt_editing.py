"""Unit tests for acestep.cli.prompt_editing."""

import unittest

from acestep.cli.prompt_editing import (
    extract_caption_lyrics,
    extract_cot_metadata,
    extract_instruction,
)


class TestExtractCaptionLyrics(unittest.TestCase):
    """Tests for extract_caption_lyrics."""

    def test_basic_extraction(self):
        prompt = "# Caption\nUpbeat pop\n\n# Lyric\nLa la la"
        caption, lyrics = extract_caption_lyrics(prompt)
        self.assertEqual(caption, "Upbeat pop")
        self.assertEqual(lyrics, "La la la")

    def test_no_match(self):
        caption, lyrics = extract_caption_lyrics("Random text with no markers")
        self.assertIsNone(caption)
        self.assertIsNone(lyrics)

    def test_trims_chat_template_markers(self):
        prompt = "# Caption\nA song\n\n# Lyric\nHello world<|im_end|>extra stuff"
        caption, lyrics = extract_caption_lyrics(prompt)
        self.assertEqual(caption, "A song")
        self.assertEqual(lyrics, "Hello world")

    def test_empty_caption_returns_none(self):
        prompt = "# Caption\n\n# Lyric\nSome lyrics"
        caption, lyrics = extract_caption_lyrics(prompt)
        self.assertIsNone(caption)
        self.assertEqual(lyrics, "Some lyrics")


class TestExtractInstruction(unittest.TestCase):
    """Tests for extract_instruction."""

    def test_basic(self):
        prompt = "# Instruction\nFill the mask\n\n# Caption\nA song"
        self.assertEqual(extract_instruction(prompt), "Fill the mask")

    def test_no_match(self):
        self.assertIsNone(extract_instruction("No instruction here"))


class TestExtractCotMetadata(unittest.TestCase):
    """Tests for extract_cot_metadata."""

    def test_basic_extraction(self):
        prompt = "<think>\nbpm: 120\nkeyscale: C major\n</think>"
        meta = extract_cot_metadata(prompt)
        self.assertEqual(meta["bpm"], "120")
        self.assertEqual(meta["keyscale"], "C major")

    def test_multiline_value(self):
        prompt = "<think>\ncaption: A beautiful\n  orchestral piece\n</think>"
        meta = extract_cot_metadata(prompt)
        self.assertEqual(meta["caption"], "A beautiful orchestral piece")

    def test_no_think_block(self):
        self.assertEqual(extract_cot_metadata("no think block"), {})

    def test_multiple_think_blocks_uses_last(self):
        prompt = (
            "<think>\nbpm: 90\n</think>\n"
            "some text\n"
            "<think>\nbpm: 120\n</think>"
        )
        meta = extract_cot_metadata(prompt)
        self.assertEqual(meta["bpm"], "120")


if __name__ == "__main__":
    unittest.main()
