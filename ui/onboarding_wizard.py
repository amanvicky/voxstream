"""First-run onboarding & audio diagnostic wizard (3 steps, non-blocking).

Shown automatically on first launch (``first_run_completed: false``) and
re-openable anytime via Tray -> "Setup & Mic Check":

  Step 1  Live Mic & Level Check   - WASAPI device list, live RMS bar,
                                     one-click privacy-settings shortcut.
  Step 2  5 s Voiceprint Enrollment - record the "Only Me" sample right
                                     here (reuses enrollment_dialog logic),
                                     or skip to use "Anyone" mode.
  Step 3  Hotkey & Target Demo      - global shortcut recap, crosshair
                                     "Lock Window" vs "Active Cursor" demo;
                                     finishing sets ``first_run_completed``.

The wizard is a plain top-level QWizard shown with ``show()`` (never
``exec()``) so dictation/HUD keep working while it is open.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QProgressBar, QPushButton,
                               QVBoxLayout, QWizard, QWizardPage)

from backend import CONFIG
from ui.enrollment_dialog import DARK_STYLE, record_sample, resolve_device
from ui.floating_hud import MeterWidget

try:
    import sounddevice as sd
    _HAS_AUDIO = True
except Exception:  # pragma: no cover
    sd = None
    _HAS_AUDIO = False

WIZARD_STYLE = DARK_STYLE + """
QWizard { background:#12141b; }
QWizard QLabel { color:#e2e8f0; }
QPushButton { background:#1e2230; color:#e2e8f0; border:1px solid #334155;
              border-radius:6px; padding:6px 14px; }
QPushButton:hover { background:#28304a; }
QPushButton:disabled { color:#64748b; }
QLineEdit { background:#1e2230; color:#e2e8f0; border:1px solid #334155;
            border-radius:6px; padding:6px; }
QComboBox { background:#1e2230; color:#e2e8f0; border:1px solid #334155;
            border-radius:6px; padding:4px; }
QProgressBar { background:#1e2230; border:1px solid #334155; border-radius:6px;
               text-align:center; }
QProgressBar::chunk { background:#22d3ee; border-radius:5px; }
"""


class MicCheckPage(QWizardPage):
    """Step 1: device list + live level + privacy shortcut."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Step 1 - Microphone & live level check")
        self.setSubTitle("Pick the microphone VoxStream should listen to and "
                         "watch the live meter while you speak.")
        self._probe_stream = None
        self._latest_rms = 0.0

        layout = QVBoxLayout(self)
        self.device_box = QComboBox()
        self.device_box.currentIndexChanged.connect(self._restart_probe)
        layout.addWidget(QLabel("Input device (WASAPI preferred):"))
        layout.addWidget(self.device_box)

        layout.addSpacing(8)
        self.meter = MeterWidget()
        self.meter.setFixedHeight(18)
        layout.addWidget(self.meter, alignment=Qt.AlignmentFlag.AlignLeft)
        self.level_hint = QLabel("Silence")
        self.level_hint.setStyleSheet("color:#94a3b8; font-size:11px;")
        layout.addWidget(self.level_hint)
        layout.addSpacing(8)

        row = QHBoxLayout()
        self.btn_recheck = QPushButton("Re-scan devices")
        self.btn_recheck.clicked.connect(self._fill_devices)
        row.addWidget(self.btn_recheck)
        self.btn_privacy = QPushButton("Open microphone privacy settings")
        self.btn_privacy.setToolTip("Windows Settings > Privacy > Microphone")
        self.btn_privacy.clicked.connect(self._open_privacy)
        row.addWidget(self.btn_privacy)
        row.addStretch(1)
        layout.addLayout(row)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("color:#94a3b8;")
        layout.addWidget(self.result_label)
        layout.addStretch(1)

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(60)
        self._ui_timer.timeout.connect(self._update_meter)

        self._fill_devices()

    # -- device listing / probing ---------------------------------------- #
    def _fill_devices(self) -> None:
        from backend.audio_capture import list_input_devices
        self.device_box.blockSignals(True)
        self.device_box.clear()
        devices = list_input_devices()
        if not devices:
            self.result_label.setText("No input devices found. Check Windows "
                                      "privacy settings or attach a microphone.")
            self.device_box.blockSignals(False)
            return
        configured = CONFIG.get("audio.input_device")
        default_idx = 0
        for i, d in enumerate(devices):
            label = f"{d['name']}  [{d['hostapi']}]"
            self.device_box.addItem(label, userData=d["index"])
            if isinstance(configured, int) and configured == d["index"]:
                default_idx = i
            elif configured is None and d["hostapi"].lower().startswith("wasapi") \
                    and default_idx == 0 and i > 0:
                default_idx = i
        self.device_box.setCurrentIndex(default_idx)
        self.device_box.blockSignals(False)
        self._restart_probe()

    def _selected_index(self):
        return self.device_box.currentData()

    def _restart_probe(self) -> None:
        self._stop_probe()
        if not _HAS_AUDIO or self._selected_index() is None:
            return
        try:
            self._probe_stream = sd.InputStream(
                samplerate=16000, channels=1, dtype="float32",
                device=int(self._selected_index()),
                blocksize=1024, callback=self._on_audio)
            self._probe_stream.start()
        except Exception as exc:
            self.result_label.setText(f"Could not open device: {exc}")
            return
        self._ui_timer.start()

    def _stop_probe(self) -> None:
        self._ui_timer.stop()
        if self._probe_stream is not None:
            try:
                self._probe_stream.stop()
                self._probe_stream.close()
            except Exception:
                pass
            self._probe_stream = None

    def _on_audio(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        import numpy as np
        if indata.size:
            self._latest_rms = float(np.sqrt(np.mean(np.square(indata))))

    def _update_meter(self) -> None:
        self.meter.set_level(self._latest_rms)
        if self._latest_rms > 0.005:
            self.level_hint.setText("Signal detected - you're good to go")
            self.level_hint.setStyleSheet("color:#22d3ee; font-size:11px;")
            self.result_label.setText("Microphone works. Continue to Step 2.")
            self.result_label.setStyleSheet("color:#22d3ee;")
        elif self._latest_rms > 0.0:
            self.level_hint.setText("Device open - speak to test")
            self.level_hint.setStyleSheet("color:#f59e0b; font-size:11px;")

    def _open_privacy(self) -> None:
        try:
            os.startfile("ms-settings:privacy-microphone")  # noqa: S606
        except Exception as exc:
            self.result_label.setText(f"Could not open settings: {exc}")

    # -- wizard lifecycle -------------------------------------------------- #
    def initializePage(self) -> None:  # noqa: N802
        self._fill_devices()

    def cleanupPage(self) -> None:  # noqa: N802
        self._stop_probe()

    def validatePage(self) -> bool:  # noqa: N802
        # Remember the chosen device for the app.
        idx = self._selected_index()
        if idx is not None:
            CONFIG.set("audio.input_device", int(idx))
        self._stop_probe()
        return True

    def isFinalPage(self) -> bool:  # noqa: N802
        return False


class EnrollPage(QWizardPage):
    """Step 2: 5-second voiceprint enrollment (or skip -> Anyone mode)."""

    recorded = Signal(str)              # wav path (GUI thread)
    enrollRequested = Signal(str, str)  # name, wav_path
    skipRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Step 2 - 'Only Me' voiceprint (5 seconds)")
        self.setSubTitle("Optional. Record a short sample so VoxLock can lock "
                         "dictation to your voice - or skip and transcribe "
                         "'Anyone'.")
        self._recording = False
        self._last_wav = None
        self._pending_name = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit("Me")
        form.addRow("Profile name:", self.name_edit)
        layout.addLayout(form)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        row = QHBoxLayout()
        self.btn_record = QPushButton("Record 5 seconds")
        self.btn_record.clicked.connect(self._start_record)
        row.addWidget(self.btn_record)
        self.btn_enroll = QPushButton("Enroll voiceprint")
        self.btn_enroll.setEnabled(False)
        self.btn_enroll.clicked.connect(self._enroll)
        row.addWidget(self.btn_enroll)
        self.btn_skip = QPushButton("Skip (use 'Anyone' mode)")
        self.btn_skip.clicked.connect(self._skip)
        row.addWidget(self.btn_skip)
        layout.addLayout(row)

        self.status = QLabel("Not enrolled yet.")
        self.status.setStyleSheet("color:#94a3b8;")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self.recorded.connect(self._on_recorded)

    # -- recording ---------------------------------------------------------- #
    def _start_record(self) -> None:
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
        device = resolve_device() if _HAS_AUDIO else None
        threading.Thread(target=self._record_thread, args=(device,),
                         daemon=True).start()

    def _tick(self) -> None:
        elapsed = time.time() - self._rec_start
        self.progress.setValue(min(100, int(100 * elapsed / 5.0)))

    def _record_thread(self, device) -> None:  # noqa: ANN001
        try:
            path = record_sample(5.0, device=device)
            self.recorded.emit(str(path))
        except Exception as exc:
            self.recorded.emit(f"__error__:{exc}")

    def _on_recorded(self, payload: str) -> None:
        self._recording = False
        self._timer.stop()
        if payload.startswith("__error__:"):
            self.progress.setValue(0)
            self.btn_record.setEnabled(True)
            self.status.setText(f"Recording failed: {payload.split(':', 1)[1]}")
            self.status.setStyleSheet("color:#ef4444;")
            return
        self._last_wav = payload
        self.progress.setValue(100)
        self.btn_record.setEnabled(True)
        self.btn_enroll.setEnabled(True)
        self.status.setText("Sample recorded - enroll it or go back to re-record.")
        self.status.setStyleSheet("color:#22d3ee;")

    # -- enroll / skip -------------------------------------------------------- #
    def _enroll(self) -> None:
        if not self._last_wav:
            return
        name = (self.name_edit.text() or "Me").strip() or "Me"
        self._pending_name = name
        self.btn_enroll.setEnabled(False)
        self.status.setText("Building voiceprint (offline)...")
        self.status.setStyleSheet("color:#94a3b8;")
        self.enrollRequested.emit(name, self._last_wav)

    def _skip(self) -> None:
        CONFIG.set("voxlock.isolation_mode", "off")
        CONFIG.set("voxlock.enabled", False)
        self.status.setText("Skipped - VoxStream will use 'Anyone' mode. "
                            "You can enroll later from the tray menu.")
        self.status.setStyleSheet("color:#f59e0b;")

    # -- controller callbacks ------------------------------------------------- #
    def on_enroll_result(self, result: dict) -> None:
        if result.get("ok"):
            meta = result.get("profile", {})
            self.status.setText(
                f"Enrolled '{meta.get('name')}' (voiceprint consistency "
                f"{meta.get('self_similarity', 0):.2f}). 'Only Me' isolation is active.")
            self.status.setStyleSheet("color:#22d3ee;")
        else:
            self.status.setText(str(result.get("message", "enrollment failed")))
            self.status.setStyleSheet("color:#ef4444;")
        self.btn_enroll.setEnabled(True)


class HotkeyTargetPage(QWizardPage):
    """Step 3: hotkey recap + lock-window vs active-cursor demo."""

    def __init__(self, on_pick_window: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self.setTitle("Step 3 - Hotkey & typing target")
        self.setSubTitle("How to control VoxStream and where your words land.")
        self._on_pick_window = on_pick_window

        hk = CONFIG.get("hotkey", {}) or {}
        mods = " + ".join(m.capitalize() for m in (hk.get("mods") or ["Ctrl", "Shift"]))
        key = str(hk.get("key") or "space").capitalize()

        layout = QVBoxLayout(self)
        hk_label = QLabel(f"<b>Global toggle:</b> {mods} + {key}")
        hk_label.setTextFormat(Qt.TextFormat.RichText)
        hk_label.setStyleSheet("font-size:14px; color:#22d3ee;")
        layout.addWidget(hk_label)
        hint = QLabel("Toggles listening from anywhere. Change it later in "
                      "Tray > Settings.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#94a3b8; font-size:11px;")
        layout.addWidget(hint)
        layout.addSpacing(8)

        layout.addWidget(QLabel("<b>Where does dictation land?</b>"))
        target_hint = QLabel(
            "- Active Cursor (default): text is typed into whatever window "
            "currently has focus - just click into Notepad/your editor and talk.\n"
            "- Locked Window (crosshair): pin dictation to one specific window "
            "so tab/app switches never interrupt it - VoxStream types into the "
            "background without stealing focus.")
        target_hint.setWordWrap(True)
        layout.addWidget(target_hint)

        row = QHBoxLayout()
        btn_lock = QPushButton("Try it: Lock a target window (crosshair)")
        btn_lock.clicked.connect(self._pick)
        row.addWidget(btn_lock)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)

        done = QLabel("Finish to start dictating. You can reopen this wizard "
                      "anytime from the tray menu ('Setup & Mic Check').")
        done.setWordWrap(True)
        done.setStyleSheet("color:#94a3b8;")
        layout.addWidget(done)

    def _pick(self) -> None:
        if self._on_pick_window:
            self._on_pick_window()

    def isFinalPage(self) -> bool:  # noqa: N802
        return True


class OnboardingWizard(QWizard):
    """Non-blocking 3-step first-run wizard."""

    def __init__(self, on_enroll: Optional[Callable] = None,
                 on_pick_window: Optional[Callable] = None,
                 on_finished: Optional[Callable] = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to VoxStream")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumSize(560, 420)
        self.setStyleSheet(WIZARD_STYLE)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)

        self._on_enroll = on_enroll
        self._on_finished = on_finished

        self.mic_page = MicCheckPage()
        self.enroll_page = EnrollPage()
        self.enroll_page.enrollRequested.connect(self._send_enroll)
        self.target_page = HotkeyTargetPage(on_pick_window=on_pick_window)
        self.addPage(self.mic_page)
        self.addPage(self.enroll_page)
        self.addPage(self.target_page)

    # ------------------------------------------------------------------ #
    def _send_enroll(self, name: str, wav_path: str) -> None:
        if self._on_enroll:
            self._on_enroll(name, wav_path)

    def on_enroll_result(self, result: dict) -> None:
        self.enroll_page.on_enroll_result(result)

    def accept(self) -> None:  # noqa: N802  (Finish pressed)
        CONFIG.set("first_run_completed", True)
        if self._on_finished:
            try:
                self._on_finished()
            except Exception:
                pass
        super().accept()
