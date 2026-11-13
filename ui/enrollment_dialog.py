"""5-second voice enrollment dialog ("Only Me" speaker lock).

Records a 5-second voiceprint from the configured microphone, saves it to
``profiles/<name>.wav`` and asks the backend to build the VoxLock embedding
(a WS ``enroll`` command).  Also lists existing profiles with
activate/delete actions.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem,
                               QProgressBar, QPushButton, QVBoxLayout)

from backend import CONFIG, PROFILES_DIR

try:
    import sounddevice as sd
    from scipy.io import wavfile
    _HAS_AUDIO = True
except Exception:  # pragma: no cover
    _HAS_AUDIO = False

DARK_STYLE = """
QDialog { background:#12141b; color:#e2e8f0; }
QLabel { color:#e2e8f0; }
QLineEdit { background:#1e2230; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:6px; }
QPushButton { background:#1e2230; color:#e2e8f0; border:1px solid #334155; border-radius:6px; padding:6px 14px; }
QPushButton:hover { background:#28304a; }
QPushButton:disabled { color:#64748b; }
QListWidget { background:#1e2230; border:1px solid #334155; border-radius:6px; }
QProgressBar { background:#1e2230; border:1px solid #334155; border-radius:6px; text-align:center; }
QProgressBar::chunk { background:#22d3ee; border-radius:5px; }
"""

PENDING_SAMPLE = "_pending_enrollment.wav"


def record_sample(seconds: float = 5.0, device=None,
                  out_path=None) -> Path:
    """Record ``seconds`` of 16 kHz mono int16 audio to a WAV file.

    Shared by the enrollment dialog and the onboarding wizard.  Raises on
    audio failure; the caller owns UI feedback.  ``out_path`` defaults to
    ``profiles/_pending_enrollment.wav``.
    """
    if not _HAS_AUDIO:
        raise RuntimeError("sounddevice is unavailable - cannot record")
    path = Path(out_path) if out_path else (PROFILES_DIR / PENDING_SAMPLE)
    rate = 16000
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1,
                   dtype="int16", device=device)
    sd.wait()
    wavfile.write(str(path), rate, np.asarray(audio, dtype=np.int16))
    return path


def resolve_device():
    """Resolve the configured/default input device index for recording."""
    dev = CONFIG.get("audio.input_device")
    if isinstance(dev, int):
        try:
            info = sd.query_devices(dev)
            if info.get("max_input_channels", 0) > 0:
                return dev
        except Exception:
            pass
    try:
        return int(sd.default.device[0])
    except Exception:
        return None


# Backwards-compatible alias
_resolve_device = resolve_device


class EnrollmentDialog(QDialog):
    """Record 5 s, enroll a VoxLock profile, manage existing profiles."""

    enrollRequested = Signal(str, str)   # name, wav_path  -> controller sends WS enroll
    setProfileRequested = Signal(str)
    deleteProfileRequested = Signal(str)
    _recordDone = Signal(str, bool)      # message, error (marshals thread -> GUI)

    RECORD_SECONDS = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VoxStream - Voice enrollment (VoxLock)")
        self.setMinimumWidth(420)
        self.setStyleSheet(DARK_STYLE)
        self._recording = False
        self._last_wav: Path | None = None
        self._pending_name: str | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("<b>Single-speaker lock ('Only Me')</b>")
        title.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(title)
        hint = QLabel("Record about 5 seconds of your normal speaking voice. "
                      "After enrollment, VoxLock only transcribes speech that "
                      "matches your voiceprint.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#94a3b8; font-size:11px;")
        layout.addWidget(hint)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Profile name:"))
        self.name_edit = QLineEdit("Me")
        name_row.addWidget(self.name_edit, stretch=1)
        layout.addLayout(name_row)

        self.status = QLabel("Ready.")
        self.status.setStyleSheet("color:#94a3b8;")
        layout.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        btn_row = QHBoxLayout()
        self.btn_record = QPushButton("Record 5 seconds")
        self.btn_record.clicked.connect(self._start_record)
        btn_row.addWidget(self.btn_record)
        self.btn_enroll = QPushButton("Enroll voiceprint")
        self.btn_enroll.setEnabled(False)
        self.btn_enroll.clicked.connect(self._enroll)
        btn_row.addWidget(self.btn_enroll)
        layout.addLayout(btn_row)

        layout.addWidget(QLabel("Saved profiles:"))
        self.profile_list = QListWidget()
        self.profile_list.itemDoubleClicked.connect(self._activate_selected)
        layout.addWidget(self.profile_list, stretch=1)

        mgmt_row = QHBoxLayout()
        self.btn_activate = QPushButton("Set active")
        self.btn_activate.clicked.connect(self._activate_selected)
        mgmt_row.addWidget(self.btn_activate)
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._delete_selected)
        mgmt_row.addWidget(self.btn_delete)
        mgmt_row.addStretch(1)
        layout.addLayout(mgmt_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.clicked.connect(lambda *_: self.reject())
        layout.addWidget(buttons)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick_progress)
        self._recordDone.connect(self._on_record_done)
        self.refresh_profiles()
        self._update_status_line()

    # ------------------------------------------------------------------ #
    def _update_status_line(self) -> None:
        active = CONFIG.get("voxlock.active_profile")
        enabled = bool(CONFIG.get("voxlock.enabled", False))
        if active and enabled:
            self.status.setText(f"VoxLock active - only transcribing profile: {active}")
            self.status.setStyleSheet("color:#22d3ee;")
        else:
            self.status.setText("VoxLock inactive - all speakers are transcribed.")
            self.status.setStyleSheet("color:#f59e0b;")

    def refresh_profiles(self) -> None:
        self.profile_list.clear()
        for meta in _list_local_profiles():
            label = f"{meta.get('name')}  ({meta.get('duration_s', '?')}s, sim {meta.get('self_similarity', '?')})"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, meta.get("name"))
            self.profile_list.addItem(item)

    def _selected_profile(self) -> str | None:
        item = self.profile_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _activate_selected(self, *_args) -> None:
        name = self._selected_profile()
        if name:
            self.setProfileRequested.emit(name)

    def _delete_selected(self, *_args) -> None:
        name = self._selected_profile()
        if name:
            self.deleteProfileRequested.emit(name)

    # ------------------------------------------------------------------ #
    def _start_record(self) -> None:
        if not _HAS_AUDIO:
            self.status.setText("sounddevice unavailable - cannot record.")
            return
        if self._recording:
            return
        self._recording = True
        self.btn_record.setEnabled(False)
        self.btn_enroll.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Recording... speak normally for 5 seconds.")
        self.status.setStyleSheet("color:#22d3ee;")
        self._rec_start = time.time()
        self._timer.start()
        device = _resolve_device()
        threading.Thread(target=self._record_thread, args=(device,), daemon=True).start()

    def _tick_progress(self) -> None:
        elapsed = time.time() - self._rec_start
        self.progress.setValue(min(100, int(100 * elapsed / self.RECORD_SECONDS)))

    def _record_thread(self, device) -> None:
        try:
            wav_path = record_sample(self.RECORD_SECONDS, device=device)
            self._last_wav = wav_path
            self._recordDone.emit(
                "Recorded 5.0 s - review the profile name, then enroll.", False)
        except Exception as exc:
            self._recordDone.emit(f"Recording failed: {exc}", True)

    def _on_record_done(self, message: str, error: bool = False) -> None:
        self._recording = False
        self._timer.stop()
        self.progress.setValue(100 if not error else 0)
        self.btn_record.setEnabled(True)
        self.btn_enroll.setEnabled(not error and self._last_wav is not None)
        self.status.setText(message)
        self.status.setStyleSheet("color:#ef4444;" if error else "color:#22d3ee;")

    def _enroll(self) -> None:
        if not self._last_wav:
            return
        name = (self.name_edit.text() or "Me").strip() or "Me"
        self._pending_name = name
        self.btn_enroll.setEnabled(False)
        self.status.setText("Building voiceprint (offline)...")
        self.status.setStyleSheet("color:#94a3b8;")
        self.enrollRequested.emit(name, str(self._last_wav))

    # -- controller callbacks --------------------------------------------- #
    def on_enroll_result(self, result: dict) -> None:
        name = self._pending_name or result.get("profile", {}).get("name")
        if result.get("ok"):
            meta = result.get("profile", {})
            sim = meta.get("self_similarity", 0)
            self.status.setText(
                f"Enrolled '{name}' (voiceprint consistency {sim:.2f}). "
                "VoxLock is now active.")
            self.status.setStyleSheet("color:#22d3ee;")
        else:
            self.status.setText(str(result.get("message", "enrollment failed")))
            self.status.setStyleSheet("color:#ef4444;")
        self.btn_enroll.setEnabled(True)
        self.refresh_profiles()
        self._update_status_line()

    def on_profile_changed(self, ok: bool, message: str = "") -> None:
        self._update_status_line()
        self.refresh_profiles()


def _list_local_profiles() -> list[dict]:
    import json
    out = []
    for p in sorted(PROFILES_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def _resolve_device():
    dev = CONFIG.get("audio.input_device")
    if isinstance(dev, int):
        try:
            info = sd.query_devices(dev)
            if info.get("max_input_channels", 0) > 0:
                return dev
        except Exception:
            pass
    try:
        return int(sd.default.device[0])
    except Exception:
        return None
