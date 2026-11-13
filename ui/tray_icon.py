"""System tray icon + Win32 global hotkey (Ctrl+Shift+Space).

The hotkey runs in a dedicated thread with a Win32 message loop
(``RegisterHotKey`` + ``GetMessage``); when fired it marshals a Qt signal so
the toggle happens on the GUI thread.  The tray icon and its glyphs are
drawn programmatically -- no asset files required.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QBrush, QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

import win32api
import win32con
import win32gui

_MOD_MAP = {"ctrl": win32con.MOD_CONTROL, "shift": win32con.MOD_SHIFT,
            "alt": win32con.MOD_ALT, "win": win32con.MOD_WIN}


def _resolve_vk(key: str) -> int:
    """Map a hotkey config key name to its Win32 virtual-key code."""
    key = str(key or "space").lower().strip()
    named = {"space": 0x20, "return": 0x0D, "enter": 0x0D, "tab": 0x09,
             "esc": 0x1B, "escape": 0x1B, "backspace": 0x08,
             "pause": 0x13, "pageup": 0x21, "pagedown": 0x22,
             "home": 0x24, "end": 0x23, "insert": 0x2D, "delete": 0x2E}
    if key in named:
        return named[key]
    if len(key) == 1 and key.isalnum():
        return ord(key.upper())
    if key.startswith("f") and key[1:].isdigit() and 1 <= int(key[1:]) <= 12:
        return 0x70 + int(key[1:]) - 1  # F1..F12
    return 0x20


def make_app_icon(color: str = "#22d3ee", size: int = 64) -> QIcon:
    """Draw the VoxStream tray glyph: rounded dark badge + stylized mic."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(16, 18, 26, 245))
    p.drawRoundedRect(2, 2, size - 4, size - 4, size * 0.28, size * 0.28)
    accent = QColor(color)
    p.setBrush(accent)
    w = size * 0.20
    p.drawRoundedRect(size / 2 - w / 2, size * 0.18, w, size * 0.34,
                      w / 2, w / 2)
    p.setBrush(Qt.BrushStyle.NoBrush)
    pen = QPen(accent, size * 0.06)
    p.setPen(pen)
    p.drawArc(size * 0.24, size * 0.24, size * 0.52, size * 0.42,
              180 * 16, 180 * 16)
    p.drawLine(int(size / 2), int(size * 0.66), int(size / 2), int(size * 0.78))
    p.setBrush(accent)
    p.drawEllipse(int(size / 2 - 2), int(size * 0.78), 4, 4)
    p.end()
    return QIcon(pm)


class GlobalHotkey(QObject):
    """RegisterHotKey-based global shortcut with a Win32 message loop."""

    triggered = Signal()

    def __init__(self, mods=("ctrl", "shift"), key="space", hotkey_id=0xB00F):
        super().__init__()
        self._mods = mods
        self._key = key
        self._id = hotkey_id
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None

    def start(self) -> bool:
        if self._thread is not None:
            return True
        self._thread = threading.Thread(target=self._run, name="voxstream-hotkey",
                                        daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        import ctypes

        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        mod = 0
        for m in self._mods:
            mod |= _MOD_MAP.get(m.lower(), 0)
        vk = _resolve_vk(self._key)
        try:
            if not win32api.RegisterHotKey(None, self._id, mod, vk):
                print("[hotkey] Ctrl+Shift+Space is registered by another app")
                return
        except Exception as exc:
            print(f"[hotkey] RegisterHotKey failed: {exc}")
            return

        user32 = ctypes.windll.user32
        WM_HOTKEY = 0x0312
        WM_QUIT = 0x0012

        class MSG(ctypes.Structure):
            _fields_ = [("hwnd", ctypes.c_void_p), ("message", ctypes.c_uint),
                        ("wParam", ctypes.c_void_p), ("lParam", ctypes.c_void_p),
                        ("time", ctypes.c_uint), ("pt", ctypes.c_int * 2)]

        msg = MSG()
        try:
            while True:
                # GetMessageW returns 0 on WM_QUIT, -1 on error.
                ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if ret == 0 or ret == -1:
                    break
                if msg.message == WM_HOTKEY and msg.wParam == self._id:
                    self.triggered.emit()  # Qt signal: safe cross-thread queued emit
        finally:
            try:
                win32api.UnregisterHotKey(None, self._id)
            except Exception:
                pass

    def stop(self) -> None:
        if self._thread_id:
            try:
                import ctypes
                ctypes.windll.user32.PostThreadMessageW(
                    self._thread_id, win32con.WM_QUIT, 0, 0)
            except Exception:
                pass
            self._thread = None
            self._thread_id = None


class TrayIcon(QSystemTrayIcon):
    """System tray icon + context menu wired to controller callbacks."""

    def __init__(self, callbacks: dict, parent=None):
        super().__init__(make_app_icon("#64748b"), parent)
        self._callbacks = callbacks
        self._color_state = "#64748b"
        self.setToolTip("VoxStream - starting...")
        self._build_menu()
        self.setVisible(True)

    def _build_menu(self) -> None:
        menu = QMenu()
        cb = self._callbacks

        self.act_toggle_hud = QAction("Show / hide HUD")
        self.act_toggle_hud.triggered.connect(cb.get("toggle_hud", lambda: None))
        menu.addAction(self.act_toggle_hud)

        self.act_listen = QAction("Start listening")
        self.act_listen.triggered.connect(cb.get("toggle_listen", lambda: None))
        menu.addAction(self.act_listen)

        menu.addSeparator()
        act_pick = QAction("Pick target window...")
        act_pick.triggered.connect(cb.get("pick_window", lambda: None))
        menu.addAction(act_pick)

        act_onboarding = QAction("Setup & Mic Check...")
        act_onboarding.triggered.connect(cb.get("onboarding", lambda: None))
        menu.addAction(act_onboarding)

        act_settings = QAction("Settings...")
        act_settings.triggered.connect(cb.get("settings", lambda: None))
        menu.addAction(act_settings)

        self.act_startup = QAction("Start with Windows")
        self.act_startup.setCheckable(True)
        self.act_startup.setChecked(bool(cb.get("is_startup_enabled",
                                                lambda: False)()))
        self.act_startup.toggled.connect(cb.get("toggle_startup", lambda _v: None))
        menu.addAction(self.act_startup)

        act_enroll = QAction("Set up 'Only Me' voice (5-second sample)...")
        act_enroll.triggered.connect(cb.get("enroll", lambda: None))
        menu.addAction(act_enroll)

        self.act_translate = QAction("Same Language")
        self.act_translate.setCheckable(True)
        self.act_translate.setToolTip(
            "Checked: your speech is translated to English.\n"
            "Unchecked: we keep whatever language you speak.")
        self.act_translate.toggled.connect(
            lambda checked, act=None: self._on_translate_toggled(checked))
        menu.addAction(self.act_translate)

        self.act_inject = QAction("Type my speech automatically")
        self.act_inject.setCheckable(True)
        self.act_inject.setChecked(True)
        self.act_inject.setToolTip(
            "When checked, VoxStream types your words as you speak.\n"
            "When unchecked, your words are only saved in History.")
        self.act_inject.toggled.connect(cb.get("set_inject", lambda _v: None))
        menu.addAction(self.act_inject)

        # 3-way speaker-isolation selector (exclusive radio group, plain English)
        menu.addSeparator()
        iso_header = QAction("Whose voice should we type?")
        iso_header.setEnabled(False)
        menu.addAction(iso_header)
        self._iso_actions = {}
        iso_group = QActionGroup(menu)
        iso_group.setExclusive(True)
        iso_tips = {
            "off": "Transcribes every speaker - good for interviews or group notes.",
            "turn_titanet": "Ignores other people talking in the room using your "
                            "5-second voice sample.",
            "overlap_voxlock": "Ignores other people talking in the room using your "
                               "5-second voice sample - even when they talk over you.",
        }
        for key, label in (("off", "Everyone"),
                           ("turn_titanet", "Only Me (Quiet Room)"),
                           ("overlap_voxlock", "Only Me (Noisy Room)")):
            act = QAction(label, menu)
            act.setCheckable(True)
            act.setData(key)
            act.setToolTip(iso_tips[key])
            iso_group.addAction(act)
            act.toggled.connect(
                lambda checked, k=key: checked and cb.get("set_isolation", lambda _v: None)(k))
            self._iso_actions[key] = act
            menu.addAction(act)
        self._iso_actions["overlap_voxlock"].setChecked(True)

        menu.addSeparator()
        self.act_llm = QAction("AI Polish (local LLM)")
        self.act_llm.setCheckable(True)
        self.act_llm.toggled.connect(cb.get("set_llm", lambda _v: None))
        menu.addAction(self.act_llm)

        menu.addSeparator()
        act_quit = QAction("Quit VoxStream")
        act_quit.triggered.connect(cb.get("quit", lambda: None))
        menu.addAction(act_quit)

        self.setContextMenu(menu)

    # -- state ------------------------------------------------------------- #
    def set_state(self, state: str) -> None:
        colors = {"starting": "#3b82f6", "ready": "#64748b", "listening": "#22d3ee",
                  "standby": "#f59e0b", "error": "#ef4444"}
        self._color_state = colors.get(state, self._color_state)
        self.setIcon(make_app_icon(self._color_state))
        labels = {"starting": "VoxStream - starting backend...",
                  "ready": "VoxStream - ready (not listening)",
                  "listening": "VoxStream - listening",
                  "standby": "VoxStream - standby (zero signal)",
                  "error": "VoxStream - error"}
        self.setToolTip(labels.get(state, "VoxStream"))
        if hasattr(self, "act_listen"):
            self.act_listen.setText(
                "Stop listening" if state == "listening" else "Start listening")

    def _on_translate_toggled(self, checked: bool) -> None:
        """Relabel the language mode in plain English and forward the change."""
        self.act_translate.setText("Translate \u2192 English" if checked
                                   else "Same Language")
        cb = self._callbacks.get("set_translate")
        if cb:
            cb(bool(checked))

    def set_flags(self, translate: Optional[bool] = None,
                  inject: Optional[bool] = None) -> None:
        if translate is not None and hasattr(self, "act_translate"):
            self.act_translate.blockSignals(True)
            self.act_translate.setChecked(bool(translate))
            self.act_translate.setText("Translate \u2192 English" if translate
                                       else "Same Language")
            self.act_translate.blockSignals(False)
        if inject is not None and hasattr(self, "act_inject"):
            self.act_inject.blockSignals(True)
            self.act_inject.setChecked(bool(inject))
            self.act_inject.blockSignals(False)

    def set_isolation(self, mode: Optional[str]) -> None:
        if not mode or not hasattr(self, "_iso_actions"):
            return
        act = self._iso_actions.get(mode)
        if act is not None and not act.isChecked():
            act.blockSignals(True)
            act.setChecked(True)
            act.blockSignals(False)

    def set_llm(self, enabled: Optional[bool]) -> None:
        if hasattr(self, "act_llm"):
            self.act_llm.blockSignals(True)
            self.act_llm.setChecked(bool(enabled))
            self.act_llm.blockSignals(False)

    def notify(self, title: str, message: str, error: bool = False) -> None:
        icon = QSystemTrayIcon.Critical if error else QSystemTrayIcon.Information
        self.showMessage(title, message, icon, 5000)
