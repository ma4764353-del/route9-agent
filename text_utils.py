"""Shared text helpers with no audio-hardware dependencies -- safe to import
from both the local voice script (voice_main.py) and the headless server
(server.py), which runs on infrastructure with no sound card at all."""

import re


def strip_markdown_for_speech(text: str) -> str:
    """The brain sometimes writes markdown (bold, numbered lists) since the
    same prompt also serves the text-based Phase 1 CLI. TTS reads that
    literally ("star star"), so strip it before it ever reaches Aura-2 --
    a defensive backstop alongside the system-prompt instruction not to
    use it in the first place."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # **bold** -> bold
    text = re.sub(r"(?<!\w)\*(.*?)\*(?!\w)", r"\1", text)  # *italic* -> italic
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)  # "1. " list markers
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)  # "- "/"* " bullets
    text = re.sub(r"\n{2,}", ". ", text)  # paragraph breaks -> a natural pause
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\.\s*\.", ".", text)  # avoid ".." where a sentence already ended in "."
    return re.sub(r"\s{2,}", " ", text).strip()
