"""Local SQLite + in-memory ring buffer storing the last N finalized utterances.

The database survives restarts (useful for "scratch that" bookkeeping and for
a reviewable transcript log); the ring buffer mirrors the newest rows for
O(1) access from the realtime pipeline.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS utterances (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL    NOT NULL,
    text           TEXT    NOT NULL,
    lang           TEXT,
    translated     INTEGER DEFAULT 0,
    engine         TEXT,
    confidence     REAL,
    injected       INTEGER DEFAULT 0,
    injected_chars INTEGER DEFAULT 0
);
"""


class HistoryStore:
    """SQLite-backed history capped at ``max_items`` (default 50) rows."""

    def __init__(self, db_path, max_items: int = 50):
        self.max_items = max(1, int(max_items))
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._ring: deque = deque(maxlen=self.max_items)
        self._load_ring()

    # ------------------------------------------------------------------ #
    def _load_ring(self) -> None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, ts, text, lang, translated, engine, confidence, injected, "
                "injected_chars FROM utterances ORDER BY id DESC LIMIT ?",
                (self.max_items,),
            )
            for row in cur.fetchall():
                self._ring.append(self._row_to_dict(row))

    @staticmethod
    def _row_to_dict(row) -> dict:
        return {
            "id": row[0],
            "ts": row[1],
            "text": row[2],
            "lang": row[3],
            "translated": bool(row[4]),
            "engine": row[5],
            "confidence": row[6],
            "injected": bool(row[7]),
            "injected_chars": row[8],
        }

    def _prune(self) -> None:
        # Keep only the newest max_items rows in the database.
        self._conn.execute(
            "DELETE FROM utterances WHERE id NOT IN "
            "(SELECT id FROM utterances ORDER BY id DESC LIMIT ?)",
            (self.max_items,),
        )

    # ------------------------------------------------------------------ #
    def add(self, text: str, lang: Optional[str] = None, translated: bool = False,
            engine: str = "whisper", confidence: Optional[float] = None,
            injected: bool = False, injected_chars: int = 0) -> dict:
        with self._lock:
            ts = time.time()
            cur = self._conn.execute(
                "INSERT INTO utterances (ts, text, lang, translated, engine, confidence, "
                "injected, injected_chars) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, text, lang, int(translated), engine, confidence,
                 int(injected), int(injected_chars)),
            )
            self._prune()
            self._conn.commit()
            row = {
                "id": cur.lastrowid, "ts": ts, "text": text, "lang": lang,
                "translated": translated, "engine": engine, "confidence": confidence,
                "injected": injected, "injected_chars": injected_chars,
            }
            self._ring.append(row)
            return dict(row)

    def last(self, n: int | None = None) -> list[dict]:
        """Newest-last list (``last(1)`` is the most recent utterance)."""
        with self._lock:
            items = list(self._ring)
            if n is not None:
                items = items[-max(1, int(n)):]
            return [dict(x) for x in items]

    def delete_last(self) -> Optional[dict]:
        with self._lock:
            if not self._ring:
                return None
            row = self._ring.pop()
            self._conn.execute("DELETE FROM utterances WHERE id = ?", (row["id"],))
            self._conn.commit()
            return dict(row)

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM utterances")
            self._conn.commit()
            self._ring.clear()

    def count(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM utterances")
            return int(cur.fetchone()[0])

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass
