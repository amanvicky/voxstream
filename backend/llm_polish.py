"""Optional local-LLM text polish hook (Ollama / LM Studio, OpenAI-compatible).

When enabled, finalized dictation clauses (> 25 chars) are sent to a local
OpenAI-compatible chat endpoint that is instructed to fix punctuation,
strip filler words (um / uh / like / you know) and preserve technical and
code terms verbatim -- never to answer, comment, or change meaning.

The mandatory hard timeout (``llm_timeout_s``, default **1.5 s**) is
enforced twice:
  * the HTTP request itself uses a 1.4 s socket timeout, and
  * the caller waits at most ``timeout`` seconds on the worker thread via
    ``join`` before giving up and returning the raw verified ASR text.

That guarantees live dictation never freezes on a busy or absent local
LLM server: worst case the unmodified transcript is injected instead.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple
from urllib.parse import urlsplit

_SYSTEM_PROMPT = (
    "You are a dictation text polisher for a voice-to-text app. "
    "Fix punctuation and capitalization. Remove filler words and sounds "
    "(um, uh, er, ah, like, you know, I mean, sort of, kind of). "
    "Preserve technical terms, code identifiers, file paths, numbers and "
    "the original language exactly. Never answer, explain, add words, or "
    "change meaning. Return ONLY the cleaned text, nothing else. "
    "If the input is already clean or not natural speech, return it "
    "unchanged."
)

_MIN_LEN = 25  # polish only finalized clauses longer than this


class LLMPolisher:
    """Hard-timeout-bounded OpenAI-compatible chat polish hook."""

    def __init__(self, config=None):
        self.config = config
        self.timeouts = 0          # telemetry: hard-timeout fallback count
        self.errors = 0
        self._model_cache: Optional[str] = None
        self._model_cache_at = 0.0
        self._reload()

    def _reload(self) -> None:
        cfg = self.config
        self.enabled = bool(cfg.get("llm_polish_enabled", False)) if cfg else False
        self.endpoint = str(cfg.get("llm_endpoint") or
                            "http://127.0.0.1:11434/v1/chat/completions") if cfg \
            else "http://127.0.0.1:11434/v1/chat/completions"
        self.model = str(cfg.get("llm_model") or "auto") if cfg else "auto"
        self.timeout = float(cfg.get("llm_timeout_s", 1.5) or 1.5) if cfg else 1.5

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    @property
    def base_url(self) -> str:
        parts = urlsplit(self.endpoint)
        return f"{parts.scheme}://{parts.netloc}"

    # ------------------------------------------------------------------ #
    def _discover_model(self) -> Optional[str]:
        """/models listing for the Ollama & LM Studio OpenAI-compatible APIs."""
        if self.model and self.model.lower() != "auto":
            return self.model
        now = time.time()
        if self._model_cache and now - self._model_cache_at < 30.0:
            return self._model_cache
        try:
            req = urllib.request.Request(
                f"{self.base_url}/models",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
            models = data.get("data") or data.get("models") or []
            for entry in models:
                mid = entry.get("id") or entry.get("name")
                if mid:
                    self._model_cache = str(mid)
                    self._model_cache_at = now
                    return self._model_cache
        except Exception:
            pass
        return None

    def _request(self, text: str, out: dict) -> None:
        """Worker-thread body: POST chat/completions (short socket timeout)."""
        try:
            payload = {
                "model": self._discover_model() or self.model or "local-model",
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 1024,
            }
            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST")
            # socket-level cap slightly below the hard deadline
            with urllib.request.urlopen(req, timeout=max(0.2, self.timeout - 0.1)) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
            content = ""
            try:
                content = (data["choices"][0]["message"]["content"] or "")
            except (KeyError, IndexError, TypeError):
                # Some LM Studio builds return {"content": "..."} directly.
                content = data.get("content") or ""
            content = content.strip().strip('"')
            out["text"] = content if content else text
            out["ok"] = bool(content) and content != text
        except Exception:
            out["ok"] = False
            out["text"] = text

    # ------------------------------------------------------------------ #
    def polish(self, text: str) -> Tuple[str, bool]:
        """Return ``(text, polished)``; never blocks longer than the timeout."""
        self._reload()
        if not self.enabled or not text:
            return text, False
        if len(text) <= _MIN_LEN:
            return text, False

        out: dict = {"text": text, "ok": False}
        worker = threading.Thread(target=self._request, args=(text, out),
                                  name="voxstream-llm", daemon=True)
        started = time.time()
        worker.start()
        worker.join(self.timeout)  # <-- the mandatory hard deadline
        if worker.is_alive():
            # Server offline/busy/slow: abandon the request, use raw ASR text.
            self.timeouts += 1
            return text, False
        if out["ok"]:
            return out["text"], True
        self.errors += 1
        return text, False
