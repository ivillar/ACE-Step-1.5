"""Formatted-prompt file editing and text extraction for the LM edit pipeline."""

import re
from typing import Dict, Optional, Tuple


def edit_formatted_prompt_via_file(formatted_prompt: str, instruction_path: str) -> str:
    """Write *formatted_prompt* to *instruction_path*, wait for user edits, then read back."""
    try:
        with open(instruction_path, "w", encoding="utf-8") as f:
            f.write(formatted_prompt)
    except Exception as e:
        print(f"WARNING: Failed to write {instruction_path}: {e}")
        return formatted_prompt

    print("\n--- Final Draft Saved ---")
    print(f"Saved to {instruction_path}")
    print("Edit the file now. Press Enter when ready to continue.")
    input()

    try:
        with open(instruction_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"WARNING: Failed to read {instruction_path}: {e}")
        return formatted_prompt


def extract_caption_lyrics(formatted_prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort extraction of caption and lyrics from a formatted prompt string."""
    matches = list(
        re.finditer(r"# Caption\n(.*?)\n+# Lyric\n(.*)", formatted_prompt, re.DOTALL)
    )
    if not matches:
        return None, None

    caption = matches[-1].group(1).strip()
    lyrics = matches[-1].group(2)

    cut_markers = [
        "<|eot_id|>", "<|start_header_id|>", "<|assistant|>",
        "<|user|>", "<|system|>", "<|im_end|>", "<|im_start|>",
    ]
    cut_at = len(lyrics)
    for marker in cut_markers:
        pos = lyrics.find(marker)
        if pos != -1:
            cut_at = min(cut_at, pos)
    lyrics = lyrics[:cut_at].rstrip()

    return caption or None, lyrics or None


def extract_instruction(formatted_prompt: str) -> Optional[str]:
    """Best-effort extraction of instruction text from a formatted prompt string."""
    match = re.search(r"# Instruction\n(.*?)\n\n", formatted_prompt, re.DOTALL)
    if not match:
        return None
    instruction = match.group(1).strip()
    return instruction or None


def extract_cot_metadata(formatted_prompt: str) -> Dict[str, str]:
    """Best-effort extraction of COT metadata (supports multi-line values)."""
    matches = list(
        re.finditer(r"<think>\n(.*?)\n</think>", formatted_prompt, re.DOTALL)
    )
    if not matches:
        return {}
    block = matches[-1].group(1)
    metadata: Dict[str, str] = {}
    current_key: Optional[str] = None
    current_value_lines: list[str] = []

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        key_match = re.match(r"^(\w+):\s*(.*)", line)
        if key_match:
            if current_key:
                metadata[current_key] = " ".join(current_value_lines).strip()
            current_key = key_match.group(1).strip().lower()
            current_value_lines = [key_match.group(2).strip()]
        elif current_key:
            current_value_lines.append(line)

    if current_key and current_value_lines:
        metadata[current_key] = " ".join(current_value_lines).strip()

    return metadata


def install_prompt_edit_hook(
    llm_handler,
    instruction_path: str,
    preloaded_prompt: Optional[str] = None,
) -> None:
    """Monkey-patch *llm_handler.build_formatted_prompt_with_cot* to allow user editing."""
    original = llm_handler.build_formatted_prompt_with_cot
    cache: dict = {}

    def wrapped(
        caption, lyrics, cot_text, is_negative_prompt=False, negative_prompt="NO USER INPUT"
    ):
        prompt = original(
            caption, lyrics, cot_text,
            is_negative_prompt=is_negative_prompt,
            negative_prompt=negative_prompt,
        )
        if is_negative_prompt:
            conditional_prompt = original(
                caption, lyrics, cot_text,
                is_negative_prompt=False,
                negative_prompt=negative_prompt,
            )
            cached = cache.get(conditional_prompt)
            if cached and (cached.get("edited_caption") or cached.get("edited_lyrics")):
                return original(
                    cached.get("edited_caption") or caption,
                    cached.get("edited_lyrics") or lyrics,
                    cot_text,
                    is_negative_prompt=True,
                    negative_prompt=negative_prompt,
                )
            return prompt

        cached = cache.get(prompt)
        if cached:
            return cached["edited_prompt"]

        if getattr(llm_handler, "_skip_prompt_edit", False):
            cache[prompt] = {
                "edited_prompt": prompt,
                "edited_caption": None,
                "edited_lyrics": None,
            }
            return prompt

        if preloaded_prompt is not None:
            edited = preloaded_prompt
        else:
            edited = edit_formatted_prompt_via_file(prompt, instruction_path)

        edited_caption, edited_lyrics = extract_caption_lyrics(edited)
        if edited != prompt:
            print("INFO: Using edited draft for audio-token prompt.")
            if edited_caption or edited_lyrics:
                llm_handler._edited_caption = edited_caption
                llm_handler._edited_lyrics = edited_lyrics
            edited_instruction = extract_instruction(edited)
            if edited_instruction:
                llm_handler._edited_instruction = edited_instruction
            edited_metas = extract_cot_metadata(edited)
            if edited_metas:
                llm_handler._edited_metas = edited_metas

        cache[prompt] = {
            "edited_prompt": edited,
            "edited_caption": edited_caption,
            "edited_lyrics": edited_lyrics,
        }
        return edited

    llm_handler.build_formatted_prompt_with_cot = wrapped
