"""Frameless, always-on-top, WS_EX_NOACTIVATE floating HUD.

A compact "pill" widget that never steals keyboard focus from the window
you are dictating into (WS_EX_NOACTIVATE), showing:

  * status LED        : starting / ready / listening / standby / error
  * live audio meter  : smoothed level bar fed by WS meter events
  * partial subtitle  : streaming Zipformer hypothesis in real time
  * final flash strip : the last finalized utterance (auto-fades)
  * buttons           : mic toggle, target-window picker, menu

The window is draggable and remembers its position in data/ui_state.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QPoint, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QMenu, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from backend import DATA_DIR

_UI_STATE = DATA_DIR / "ui_state.json"

# state -> (color, label)
_STATES = {
    "starting": ("#3b82f6", "Starting backend..."),
    "ready": ("#64748b", "Ready - press the mic or Ctrl+Shift+Space"),
    "listening": ("#22d3ee", "Listening..."),
    "standby": ("#f59e0b", "Standby (zero signal) - speak to wake"),
    "error": ("#ef4444", "Error"),
}

STYLE_BUTTON = """
QPushButton { background: transparent; border: none; border-radius: 14px; padding: 4px; }
QPushButton:hover { background: rgba(255,255,255,28); }
QPushButton:pressed { background: rgba(255,255,255,60); }
"""


def make_icon(kind: str, color: str = "#e2e8f0", size: int = 32) -> QPixmap:
    """Draw the HUD button glyphs (no external asset files needed)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    c = QColor(color)
    pen = QPen(c, max(2.0, size / 16.0))
    p.setPen(pen)
    p.setBrush(c)
    cx, cy = size / 2.0, size / 2.0
    if kind == "mic":
        w, h = size * 0.28, size * 0.52
        p.drawRoundedRect(QRectF(cx - w / 2, size * 0.12, w, h), w / 2, h / 2)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(QRectF(cx - size * 0.26, size * 0.34, size * 0.52, size * 0.42),
                  180 * 16, 180 * 16)
        p.drawLine(int(cx), int(size * 0.76), int(cx), int(size * 0.88))
    elif kind == "target":
        p.setBrush(Qt.BrushStyle.NoBrush)
        r = size * 0.3
        p.drawEllipse(QRectF(cx - r / 2, cy - r / 2, r, r))
        gap = size * 0.44
        arm = size * 0.2
        p.drawLine(int(cx - gap), int(cy - gap - arm), int(cx - gap), int(cy - gap))
        p.drawLine(int(cx - gap - arm), int(cy - gap), int(cx - gap), int(cy - gap))
        p.drawLine(int(cx + gap), int(cy - gap - arm), int(cx + gap), int(cy - gap))
        p.drawLine(int(cx + gap), int(cy - gap), int(cx + gap + arm), int(cy - gap))
        p.drawLine(int(cx - gap), int(cy + gap), int(cx - gap), int(cy + gap + arm))
        p.drawLine(int(cx - gap - arm), int(cy + gap), int(cx - gap), int(cy + gap))
        p.drawLine(int(cx + gap), int(cy + gap), int(cx + gap), int(cy + gap + arm))
        p.drawLine(int(cx + gap), int(cy + gap), int(cx + gap + arm), int(cy + gap))
    elif kind == "menu":
        for dy in (-0.24, 0.0, 0.24):
            p.drawLine(int(size * 0.2), int(cy + dy * size),
                       int(size * 0.8), int(cy + dy * size))
    p.end()
    return pm


class MeterWidget(QWidget):
    """Horizontal live level bar (log-scaled dB display)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(96, 14)
        self._level = 0.0
        self._color = QColor("#22d3ee")

    def set_level(self, rms: float) -> None:
        # map -60 dB..0 dB -> 0..1
        import math
        db = 20.0 * math.log10(max(rms, 1e-6))
        norm = max(0.0, min(1.0, (db + 60.0) / 60.0))
        self._level = 0.7 * self._level + 0.3 * norm
        self.update()

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        track = QRectF(0, 2.5, self.width(), self.height() - 5)
        path = QPainterPath()
        path.addRoundedRect(track, 5, 5)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 26))
        p.drawPath(path)
        if self._level > 0.01:
            fill = QRectF(0, 2.5, max(6.0, self._level * self.width()), self.height() - 5)
            fpath = QPainterPath()
            fpath.addRoundedRect(fill, 5, 5)
            p.setBrush(self._color)
            p.drawPath(fpath)
        p.end()


class FloatingHUD(QWidget):
    """The always-on-top pill + subtitle strip."""

    listenToggled = Signal()
    pickRequested = Signal()
    enrollRequested = Signal()
    translateToggled = Signal(bool)
    injectToggled = Signal(bool)
    isolationChanged = Signal(str)   # off | turn_titanet | overlap_voxlock
    llmToggled = Signal(bool)        # AI polish (local LLM) on/off
    quitRequested = Signal()

    # mode key -> (selector label, tooltip)  [plain-English labels]
    ISOLATION_MODES = (
        ("off", "Everyone",
         "Transcribes every speaker - good for interviews or group notes."),
        ("turn_titanet", "Only Me (Quiet Room)",
         "Ignores other people talking in the room using your 5-second voice sample."),
        ("overlap_voxlock", "Only Me (Noisy Room)",
         "Ignores other people talking in the room using your 5-second voice sample - "
         "even when they talk over you at the same time."),
    )

    TARGET_TYPE_ANYWHERE_TIP = (
        "Type Anywhere: types your speech into whatever app or browser box you "
        "click into (ChatGPT, Chrome, Word, etc.).\n"
        "Click \U0001F3AF to lock one app instead - VoxStream keeps typing there "
        "even while you browse Chrome.")
    TARGET_LOCKED_TIP = (
        "Lock App \U0001F3AF: VoxStream keeps typing into '{title}' even while "
        "you browse Chrome.\nClick \U0001F3AF to pick another app.")

    def __init__(self, config=None):
        super().__init__(None)
        self.config = config
        self._drag_pos: Optional[QPoint] = None
        self._accent = "#64748b"

        self.setWindowTitle("VoxStream")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
                            Qt.WindowType.Tool | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 8)
        root.setSpacing(2)

        # --- pill row -------------------------------------------------------
        pill_row = QHBoxLayout()
        pill_row.setSpacing(8)

        self.led = QLabel()
        self.led.setFixedSize(12, 12)
        pill_row.addWidget(self.led)

        self.meter = MeterWidget()
        pill_row.addWidget(self.meter)

        self.partial_label = QLabel("Starting backend...")
        self.partial_label.setStyleSheet(
            "color:#94a3b8; font-size:12px; background:transparent;")
        self.partial_label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.partial_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.partial_label.setWordWrap(False)
        pill_row.addWidget(self.partial_label, stretch=1)

        # 3-way speaker-isolation selector (plain-English labels)
        self.isolation_box = QComboBox()
        self.isolation_box.setObjectName("isolationBox")
        for key, label, tip in self.ISOLATION_MODES:
            self.isolation_box.addItem(label, userData=key)
            self.isolation_box.setItemData(self.isolation_box.count() - 1, tip,
                                           Qt.ItemDataRole.ToolTipRole)
        self.isolation_box.setToolTip(self.ISOLATION_MODES[2][2])
        self.isolation_box.setFixedWidth(150)
        self.isolation_box.setStyleSheet(
            "QComboBox { background:rgba(255,255,255,18); color:#e2e8f0;"
            " border:1px solid rgba(255,255,255,40); border-radius:10px;"
            " padding:2px 8px; font-size:11px; }"
            "QComboBox::drop-down { border:none; width:16px; }"
            "QComboBox QAbstractItemView { background:#14161b; color:#e2e8f0;"
            " selection-background-color:#1e3a4a; }")
        self.isolation_box.setCursor(Qt.CursorShape.PointingHandCursor)
        self.isolation_box.currentIndexChanged.connect(self._on_isolation_changed)
        pill_row.addWidget(self.isolation_box)

        self.btn_mic = QPushButton()
        self.btn_mic.setFixedSize(30, 30)
        self.btn_mic.setIcon(make_icon("mic"))
        self.btn_mic.setIconSize(self.btn_mic.size() * 0.6)
        self.btn_mic.setStyleSheet(STYLE_BUTTON)
        self.btn_mic.setToolTip("Start / stop listening (Ctrl+Shift+Space)")
        self.btn_mic.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mic.clicked.connect(self.listenToggled.emit)
        pill_row.addWidget(self.btn_mic)

        self.btn_target = QPushButton()
        self.btn_target.setFixedSize(30, 30)
        self.btn_target.setIcon(make_icon("target"))
        self.btn_target.setIconSize(self.btn_target.size() * 0.6)
        self.btn_target.setStyleSheet(STYLE_BUTTON)
        self.btn_target.setToolTip(self.TARGET_TYPE_ANYWHERE_TIP)
        self.btn_target.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_target.clicked.connect(self.pickRequested.emit)
        pill_row.addWidget(self.btn_target)

        self.btn_menu = QPushButton()
        self.btn_menu.setFixedSize(30, 30)
        self.btn_menu.setIcon(make_icon("menu"))
        self.btn_menu.setIconSize(self.btn_menu.size() * 0.6)
        self.btn_menu.setStyleSheet(STYLE_BUTTON)
        self.btn_menu.setToolTip("Menu")
        self.btn_menu.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_menu.clicked.connect(self._open_menu)
        pill_row.addWidget(self.btn_menu)

        root.addLayout(pill_row)

        # --- subtitle row -----------------------------------------------------
        self.subtitle_label = QLabel("")
        self.subtitle_label.setStyleSheet(
            "color:#64748b; font-style:italic; font-size:11px; background:transparent;")
        self.subtitle_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.subtitle_label.setWordWrap(False)
        root.addWidget(self.subtitle_label)

        self._subtitle_timer = QTimer(self)
        self._subtitle_timer.setSingleShot(True)
        self._subtitle_timer.timeout.connect(
            lambda: self.subtitle_label.setText(""))

        # --- "no text box" safety-net banner (auto-dismissing) ----------------
        self.banner = QWidget(self)
        banner_row = QHBoxLayout(self.banner)
        banner_row.setContentsMargins(0, 0, 0, 0)
        banner_row.setSpacing(6)
        self.banner_label = QLabel("No text box selected - your words were saved.")
        self.banner_label.setStyleSheet(
            "color:#fbbf24; font-size:11px; background:transparent;")
        banner_row.addWidget(self.banner_label, stretch=1)
        self.btn_copy_last = QPushButton("Copy Last Text")
        self.btn_copy_last.setFixedHeight(22)
        self.btn_copy_last.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_copy_last.setStyleSheet(
            "QPushButton { background:#f59e0b; color:#111827; border:none;"
            " border-radius:6px; padding:2px 10px; font-size:11px; font-weight:bold; }"
            "QPushButton:hover { background:#fbbf24; }")
        self.btn_copy_last.clicked.connect(self._copy_last_text)
        banner_row.addWidget(self.btn_copy_last)
        self.banner.setVisible(False)
        root.addWidget(self.banner)

        self._banner_timer = QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(lambda: self.banner.setVisible(False))
        self._last_final_text = ""

        self.resize(560, 92)
        self._restore_position()
        self._current_state = "starting"
        self._llm_checked = False
        self.set_state("starting")

    # ------------------------------------------------------------------ #
    # window behaviour
    # ------------------------------------------------------------------ #
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._apply_noactivate()

    def _apply_noactivate(self) -> None:
        """WS_EX_NOACTIVATE so the HUD never steals focus while dictating."""
        try:
            import ctypes
            hwnd = int(self.winId())
            if not hwnd:
                return
            GWL_EXSTYLE = -20
            WS_EX_NOACTIVATE = 0x08000000
            WS_EX_TOOLWINDOW = 0x00000080
            WS_EX_TOPMOST = 0x00000008
            user32 = ctypes.windll.user32
            GetWindowLongPtrW = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            SetWindowLongPtrW = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
            SetWindowLongPtrW(hwnd, GWL_EXSTYLE,
                              style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST)
        except Exception:
            pass

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_pos = None
        self._save_position()
        super().mouseReleaseEvent(event)

    def _save_position(self) -> None:
        try:
            state = {}
            if _UI_STATE.exists():
                state = json.loads(_UI_STATE.read_text(encoding="utf-8"))
            state.update({"hud_x": self.x(), "hud_y": self.y()})
            _UI_STATE.write_text(json.dumps(state), encoding="utf-8")
        except Exception:
            pass

    def _restore_position(self) -> None:
        try:
            if _UI_STATE.exists():
                state = json.loads(_UI_STATE.read_text(encoding="utf-8"))
                if state.get("hud_x") is not None:
                    self.move(int(state["hud_x"]), int(state["hud_y"]))
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # painting
    # ------------------------------------------------------------------ #
    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        path = QPainterPath()
        path.addRoundedRect(rect, 20, 20)
        p.fillPath(path, QColor(16, 18, 24, 235))
        glow = QColor(self._accent)
        glow.setAlpha(160)
        pen = QPen(glow, 1.4)
        p.setPen(pen)
        p.drawPath(path)
        p.end()

    # ------------------------------------------------------------------ #
    # state API (called from the controller on WS events)
    # ------------------------------------------------------------------ #
    def set_state(self, state: str, detail: str = "") -> None:
        color, label = _STATES.get(state, ("#64748b", state))
        self._current_state = state
        self._accent = color
        self._set_led(color)
        self.meter.set_color(color)
        if state in ("ready", "listening", "standby", "starting", "error"):
            self.partial_label.setStyleSheet(
                f"color:{color}; font-size:12px; background:transparent;")
            self.partial_label.setText(detail or label)
        self.setToolTip(f"VoxStream - {label}")
        self.update()

    def _set_led(self, color: str) -> None:
        self.led.setStyleSheet(
            f"background:{color}; border-radius:6px; min-width:12px; min-height:12px;")

    def set_meter(self, rms: float) -> None:
        self.meter.set_level(rms)

    def set_partial(self, text: str) -> None:
        if text:
            metrics = self.partial_label.fontMetrics()
            self.partial_label.setStyleSheet(
                "color:#e2e8f0; font-size:13px; background:transparent;")
            self.partial_label.setText(metrics.elidedText(text, Qt.TextElideMode.ElideRight, 330))
        else:
            # restore the status line for the current state
            state = getattr(self, "_current_state", "ready")
            color, label = _STATES.get(state, ("#64748b", state))
            self.partial_label.setStyleSheet(
                f"color:{color}; font-size:12px; background:transparent;")
            self.partial_label.setText(label)

    def set_final(self, text: str, translated: bool = False,
                  polished: bool = False) -> None:
        prefix = "[EN] " if translated else ""
        if polished:
            prefix += "[AI] "
        self._last_final_text = text
        metrics = self.subtitle_label.fontMetrics()
        self.subtitle_label.setStyleSheet(
            "color:#a5f3fc; font-style:italic; font-size:12px; background:transparent;")
        self.subtitle_label.setText(
            metrics.elidedText(prefix + text, Qt.TextElideMode.ElideRight, 520))
        self._subtitle_timer.start(8000)

    def show_no_text_box(self, text: str = "") -> None:
        """Safety net: spoken words had nowhere to land (desktop/taskbar)."""
        if text:
            self._last_final_text = text
        self.banner.setVisible(True)
        self.banner_label.setText("No text box selected - your words were saved. "
                                  "Click into a text box and talk again.")
        self._banner_timer.start(10000)

    def _copy_last_text(self) -> None:
        try:
            from PySide6.QtWidgets import QApplication
            QApplication.clipboard().setText(self._last_final_text or "")
            self.banner_label.setText("Copied! Paste it anywhere with Ctrl+V.")
            self.banner_label.setStyleSheet(
                "color:#22d3ee; font-size:11px; background:transparent;")
            self._banner_timer.start(6000)
        except Exception:
            self._banner_timer.start(2000)

    def set_rejected(self, similarity: Optional[float]) -> None:
        sim = f" (sim {similarity:.2f})" if similarity is not None else ""
        self.set_final(f"VoxLock filtered another speaker{sim}")

    def set_command(self, name: str) -> None:
        self.set_final(f"command: {name}")

    def set_target(self, title: Optional[str]) -> None:
        if title:
            self.btn_target.setToolTip(self.TARGET_LOCKED_TIP.format(title=title))
            self.subtitle_label.setText(f"Lock App \U0001F3AF: {title}")
            self._subtitle_timer.start(6000)
        else:
            self.btn_target.setToolTip(self.TARGET_TYPE_ANYWHERE_TIP)
            self.subtitle_label.setText("Type Anywhere mode - click into any text box and talk")
            self._subtitle_timer.start(6000)

    def set_toggle_icons(self, listening: bool) -> None:
        self.btn_mic.setToolTip("Stop listening (Ctrl+Shift+Space)" if listening
                                else "Start listening (Ctrl+Shift+Space)")

    # ------------------------------------------------------------------ #
    def _on_isolation_changed(self, index: int) -> None:
        key = self.isolation_box.itemData(index)
        if key:
            self.isolationChanged.emit(str(key))

    def set_isolation(self, mode: str) -> None:
        """Sync the selector from backend state (no signal loops)."""
        for i in range(self.isolation_box.count()):
            if self.isolation_box.itemData(i) == mode:
                self.isolation_box.blockSignals(True)
                self.isolation_box.setCurrentIndex(i)
                self.isolation_box.blockSignals(False)
                return

    def set_llm(self, enabled: bool) -> None:
        self._llm_checked = bool(enabled)

    def _open_menu(self) -> None:
        menu = QMenu(self)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        act_pick = menu.addAction(make_icon("target"), "Lock App \U0001F3AF (choose where to type)...")
        act_pick.triggered.connect(self.pickRequested.emit)
        act_enroll = menu.addAction(make_icon("mic"), "Set up 'Only Me' voice (5-second sample)...")
        act_enroll.triggered.connect(self.enrollRequested.emit)

        act_llm = menu.addAction("AI Polish (local LLM)")
        act_llm.setCheckable(True)
        act_llm.setChecked(bool(getattr(self, "_llm_checked", False)))
        act_llm.toggled.connect(self.llmToggled.emit)

        cfg = self.config
        translate_on = bool(cfg.get("translation.enabled", False)) if cfg else False
        act_translate = menu.addAction("Translate \u2192 English" if translate_on
                                       else "Same Language")
        act_translate.setCheckable(True)
        act_translate.setChecked(translate_on)
        act_translate.setToolTip("When checked, your speech is translated to English. "
                                 "When unchecked, we keep whatever language you speak.")
        act_translate.toggled.connect(
            lambda checked: (act_translate.setText("Translate \u2192 English" if checked
                                                  else "Same Language"),
                             self.translateToggled.emit(checked)))

        act_inject = menu.addAction("Type my speech automatically")
        act_inject.setCheckable(True)
        act_inject.setChecked(bool(cfg.get("injection.enabled", True)) if cfg else False)
        act_inject.toggled.connect(self.injectToggled.emit)

        menu.addSeparator()
        act_quit = menu.addAction("Quit VoxStream")
        act_quit.triggered.connect(self.quitRequested.emit)
        menu.popup(self.btn_menu.mapToGlobal(self.btn_menu.rect().topLeft()))
