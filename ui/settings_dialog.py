"""Settings & dictionary editor dialog.

One place for everything users would otherwise hand-edit in config.json:

* **Hotkey customizer**      - global mic toggle (default Ctrl+Shift+Space)
* **Jargon table**           - add/edit/delete custom word replacements
                               (regex -> replacement, e.g. "a sink" -> "async")
* **AI Polish**              - enable toggle, endpoint, model + a
                               "Test Local LLM (<1.5s)" probe that proves the
                               hard-timeout fallback live
* **Utterance history**      - view/copy the last 50 dictated utterances

Saving writes config.json (shared with the backend) and notifies the
controller, which re-registers the hotkey and asks the backend to
re-apply settings live (``refresh_settings``).
"""

from __future__ import annotations

import re
import time
from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QFormLayout,
                               QHBoxLayout, QKeySequenceEdit, QLabel,
                               QLineEdit, QListWidget, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

from backend import CONFIG, DATA_DIR
from backend.history_store import HistoryStore
from backend.llm_polish import LLMPolisher

DARK_STYLE = """
QDialog { background:#12141b; color:#e2e8f0; }
QLabel { color:#e2e8f0; }
QLineEdit { background:#1e2230; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:5px; }
QPushButton { background:#1e2230; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:5px 12px; }
QPushButton:hover { background:#28304a; }
QTableWidget { background:#1e2230; color:#e2e8f0; border:1px solid #334155; gridline-color:#334155; }
QHeaderView::section { background:#1a1f2c; color:#94a3b8; border:none; padding:4px; }
QListWidget { background:#1e2230; color:#e2e8f0; border:1px solid #334155; }
QCheckBox { color:#e2e8f0; }
QKeySequenceEdit { background:#1e2230; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:5px; }
"""

_MOD_MAP = {"ctrl": "Ctrl", "shift": "Shift", "alt": "Alt", "win": "Meta"}


def _key_sequence_to_config(sequence: QKeySequence) -> Optional[dict]:
    """Convert a captured QKeySequence into hotkey config {mods, key}."""
    if sequence.isEmpty():
        return None
    text = sequence.toString().lower()
    mods = []
    for token, name in (("ctrl+", "ctrl"), ("shift+", "shift"),
                        ("alt+", "alt"), ("meta+", "win")):
        if token in text:
            mods.append(name)
            text = text.replace(token, "")
    text = text.strip()
    if not text:
        return None
    known = {"space": "space", "return": "return", "enter": "return",
             "tab": "tab", "escape": "esc", "esc": "esc"}
    if text in known:
        key = known[text]
    elif len(text) == 1 and (text.isalnum() or text in "+-=[];',./\\"):
        key = text
    elif text.startswith("f") and text[1:].isdigit() and 1 <= int(text[1:]) <= 12:
        key = text
    else:
        return None
    if not mods:
        return None  # global hotkeys must include a modifier
    return {"mods": mods, "key": key}


class _StubConfig:
    """Minimal config view for a throwaway LLMPolisher probe."""

    def __init__(self, values: dict):
        self.d = values

    def get(self, key, default=None):
        return self.d.get(key, default)


class SettingsDialog(QDialog):
    """Tray -> Settings... : everything without hand-editing config.json."""

    def __init__(self, on_saved: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VoxStream - Settings")
        self.setMinimumSize(560, 560)
        self.setStyleSheet(DARK_STYLE)
        self._on_saved = on_saved

        layout = QVBoxLayout(self)

        # --- hotkey -------------------------------------------------------
        layout.addWidget(QLabel("<b>Global mic-toggle hotkey</b>"))
        hk = CONFIG.get("hotkey", {}) or {}
        seq = " + ".join([_MOD_MAP.get(m, m.capitalize())
                          for m in (hk.get("mods") or [])] +
                         [str(hk.get("key") or "space").capitalize()])
        self.hotkey_edit = QKeySequenceEdit(QKeySequence(seq))
        layout.addWidget(self.hotkey_edit)

        # --- jargon table ---------------------------------------------------
        layout.addSpacing(8)
        layout.addWidget(QLabel("<b>Custom jargon & replacements "
                                "(regex → replacement)</b>"))
        self.jargon_table = QTableWidget(0, 2)
        self.jargon_table.setHorizontalHeaderLabels(["Match (regex)", "Replace with"])
        self.jargon_table.horizontalHeader().setStretchLastSection(True)
        self.jargon_table.horizontalHeader().setMinimumSectionSize(200)
        self.jargon_table.verticalHeader().setVisible(False)
        for pattern, repl in (CONFIG.get("custom_dictionary", {}) or {}).items():
            self._add_jargon_row(pattern, repl)
        layout.addWidget(self.jargon_table, stretch=2)
        jrow = QHBoxLayout()
        btn_add = QPushButton("Add")
        btn_add.clicked.connect(lambda: self._add_jargon_row("", ""))
        jrow.addWidget(btn_add)
        btn_del = QPushButton("Delete selected")
        btn_del.clicked.connect(self._delete_jargon)
        jrow.addWidget(btn_del)
        jrow.addStretch(1)
        layout.addLayout(jrow)

        # --- AI polish --------------------------------------------------------
        layout.addSpacing(8)
        layout.addWidget(QLabel("<b>AI Polish (local LLM, hard 1.5s timeout)</b>"))
        self.llm_enabled = QCheckBox("Enable AI Polish")
        self.llm_enabled.setChecked(bool(CONFIG.get("llm_polish_enabled", False)))
        layout.addWidget(self.llm_enabled)
        form = QFormLayout()
        self.llm_endpoint = QLineEdit(str(CONFIG.get("llm_endpoint") or ""))
        form.addRow("Endpoint:", self.llm_endpoint)
        self.llm_model = QLineEdit(str(CONFIG.get("llm_model") or "auto"))
        form.addRow("Model:", self.llm_model)
        layout.addLayout(form)
        lrow = QHBoxLayout()
        self.btn_llm_test = QPushButton("Test Local LLM (<1.5s)")
        self.btn_llm_test.clicked.connect(self._test_llm)
        lrow.addWidget(self.btn_llm_test)
        self.llm_test_result = QLabel("")
        self.llm_test_result.setStyleSheet("color:#94a3b8;")
        lrow.addWidget(self.llm_test_result, stretch=1)
        layout.addLayout(lrow)

        # --- history -----------------------------------------------------------
        layout.addSpacing(8)
        layout.addWidget(QLabel("<b>Recent dictation (last 50)</b>"))
        self.history_list = QListWidget()
        self._fill_history()
        layout.addWidget(self.history_list, stretch=3)
        hrow = QHBoxLayout()
        btn_copy = QPushButton("Copy selected")
        btn_copy.clicked.connect(self._copy_selected)
        hrow.addWidget(btn_copy)
        btn_copy_all = QPushButton("Copy all")
        btn_copy_all.clicked.connect(self._copy_all)
        hrow.addWidget(btn_copy_all)
        hrow.addStretch(1)
        layout.addLayout(hrow)

        # --- save / close ---------------------------------------------------------
        brow = QHBoxLayout()
        brow.addStretch(1)
        btn_save = QPushButton("Save settings")
        btn_save.setDefault(True)
        btn_save.clicked.connect(self.save)
        brow.addWidget(btn_save)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        brow.addWidget(btn_close)
        layout.addLayout(brow)

    # ------------------------------------------------------------------ #
    def _add_jargon_row(self, pattern: str, repl: str) -> None:
        row = self.jargon_table.rowCount()
        self.jargon_table.insertRow(row)
        self.jargon_table.setItem(row, 0, QTableWidgetItem(str(pattern)))
        self.jargon_table.setItem(row, 1, QTableWidgetItem(str(repl)))

    def _delete_jargon(self) -> None:
        for item in sorted(self.jargon_table.selectedIndexes(), reverse=True):
            self.jargon_table.removeRow(item.row())

    def _fill_history(self) -> None:
        try:
            db = DATA_DIR / str(CONFIG.get("history.db") or "data/history.db")
            store = HistoryStore(db, int(CONFIG.get("history.max_items", 50) or 50))
            rows = store.last(50)
            store.close()
        except Exception as exc:
            self.history_list.addItem(f"(history unavailable: {exc})")
            return
        for row in rows:
            stamp = time.strftime("%m-%d %H:%M", time.localtime(row.get("ts", 0)))
            tags = "[EN] " if row.get("translated") else ""
            self.history_list.addItem(f"{stamp}  {tags}{row.get('text', '')}")

    def _copy_selected(self) -> None:
        item = self.history_list.currentItem()
        if item:
            QApplication.clipboard().setText(item.text())

    def _copy_all(self) -> None:
        QApplication.clipboard().setText(
            "\n".join(self.history_list.item(i).text()
                      for i in range(self.history_list.count())))

    # -- LLM probe -------------------------------------------------------- #
    def _test_llm(self) -> None:
        self.btn_llm_test.setEnabled(False)
        self.llm_test_result.setText("Testing...")
        self.llm_test_result.setStyleSheet("color:#94a3b8;")
        QApplication.processEvents()
        stub = _StubConfig({
            "llm_polish_enabled": True,
            "llm_endpoint": self.llm_endpoint.text().strip(),
            "llm_model": self.llm_model.text().strip() or "auto",
            "llm_timeout_s": 1.5,
        })
        polisher = LLMPolisher(stub)
        sample = "um so this is a test dictation clause long enough to need polishing okay"
        t0 = time.time()
        text, polished = polisher.polish(sample)
        elapsed = time.time() - t0
        if polished:
            self.llm_test_result.setText(f"OK in {elapsed:.2f}s: {text}")
            self.llm_test_result.setStyleSheet("color:#22d3ee;")
        elif elapsed < 1.5:
            self.llm_test_result.setText(
                f"No change from the model ({elapsed:.2f}s) - endpoint reachable but "
                "returned the text unchanged or unavailable")
            self.llm_test_result.setStyleSheet("color:#f59e0b;")
        else:
            self.llm_test_result.setText(
                f"Fallback after hard timeout ({elapsed:.2f}s) - raw ASR text is used. "
                "Is the local LLM server running?")
            self.llm_test_result.setStyleSheet("color:#ef4444;")
        self.btn_llm_test.setEnabled(True)

    # -- persistence ---------------------------------------------------------- #
    def save(self) -> None:
        # hotkey
        hk = _key_sequence_to_config(self.hotkey_edit.keySequence())
        if hk:
            CONFIG.set("hotkey", hk)
        # jargon
        jargon = {}
        for row in range(self.jargon_table.rowCount()):
            pattern_item = self.jargon_table.item(row, 0)
            repl_item = self.jargon_table.item(row, 1)
            pattern = (pattern_item.text() if pattern_item else "").strip()
            repl = (repl_item.text() if repl_item else "")
            if not pattern:
                continue
            try:
                re.compile(pattern)  # validate
            except re.error:
                continue  # silently skip invalid rows
            jargon[pattern] = repl
        CONFIG.set("custom_dictionary", jargon)
        # llm polish
        CONFIG.set("llm_polish_enabled", bool(self.llm_enabled.isChecked()))
        CONFIG.set("llm_endpoint", self.llm_endpoint.text().strip())
        CONFIG.set("llm_model", self.llm_model.text().strip() or "auto")
        CONFIG.save()

        if self._on_saved:
            try:
                self._on_saved()
            except Exception:
                pass
        self.accept()
