"""Boundary-anchored voice commands and regex jargon replacement.

Two mechanisms live here:

1. ``parse`` -- voice commands that are anchored at utterance boundaries:
   * ``whole``  : the pattern must be the entire utterance (after
                  normalization) e.g. "stop listening".
   * ``prefix`` : the pattern must occupy the start of the utterance; the
                  remainder of the utterance is returned as the argument,
                  e.g. "new paragraph The GPU works" -> new_paragraph("The GPU works").

2. ``apply_dictionary`` -- regex jargon / custom-dictionary replacement applied
   to regular dictation text (and to command arguments), driven by
   ``custom_dictionary`` in config.json.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace (keeps word content)."""
    t = _PUNCT.sub(" ", (text or "").lower())
    return " ".join(t.split())


class CommandProcessor:
    def __init__(self, config):
        cfg = config.get("commands", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.commands: List[dict] = list(cfg.get("list", []) or [])
        self.dictionary: Dict[str, str] = dict(config.get("custom_dictionary", {}) or {})
        self._compiled: List[Tuple[re.Pattern, str]] = []
        for pattern, repl in self.dictionary.items():
            try:
                self._compiled.append((re.compile(pattern), repl))
            except re.error as exc:
                print(f"[commands] skipping invalid regex {pattern!r}: {exc}")

    # ------------------------------------------------------------------ #
    def parse(self, text: str) -> Tuple[Optional[str], str, Optional[str]]:
        """Return ``(command_name, argument_text, matched_pattern)``.

        ``command_name`` is ``None`` when the utterance is plain dictation.
        """
        if not self.enabled:
            return None, text, None
        t = normalize(text)
        if not t:
            return None, text, None
        for cmd in self.commands:
            name = cmd.get("name")
            anchor = cmd.get("anchoring", "whole")
            for pattern in cmd.get("patterns", []):
                p = normalize(pattern)
                if not p:
                    continue
                if anchor == "whole" and t == p:
                    return name, "", pattern
                if anchor == "prefix" and (t == p or t.startswith(p + " ")):
                    # Drop the first len(p.split()) words from the ORIGINAL text
                    # so the argument keeps the ASR's original casing/punctuation.
                    words = (text or "").split()
                    skip = len(p.split())
                    argument = " ".join(words[skip:]).strip() if len(words) > skip else ""
                    return name, argument, pattern
        return None, text, None

    # ------------------------------------------------------------------ #
    def apply_dictionary(self, text: str) -> str:
        if not text:
            return text
        for rx, repl in self._compiled:
            try:
                text = rx.sub(repl, text)
            except re.error:
                continue
        return text

    def process(self, text: str) -> dict:
        """One-shot helper: returns command (if any) plus cleaned text.

        ``result["text"]`` is what should be typed into the target window:
        the command argument for prefix commands, otherwise the dictation
        text with the custom dictionary applied.
        """
        name, argument, pattern = self.parse(text)
        if name is not None:
            cleaned = self.apply_dictionary(argument).strip()
        else:
            cleaned = self.apply_dictionary(text).strip()
        return {"command": name, "pattern": pattern, "argument": argument, "text": cleaned}
