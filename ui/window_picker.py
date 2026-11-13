"""Crosshair click-to-lock target window picker.

A fullscreen overlay that is visually present but *transparent to mouse
hits* (WS_EX_TRANSPARENT + WS_EX_LAYERED), so the real windows below keep
receiving events.  A poll timer watches the cursor and the left mouse
button; the window under the cursor is highlighted live (title + process
name), and a click locks it.  Esc or right-click cancels.
"""

from __future__ import annotations

import ctypes
from typing import Optional

from PySide6.QtCore import QRect, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

try:
    import win32api
    import win32con
    import win32gui
    import win32process
    import psutil
    _HAS_WIN32 = True
except ImportError:  # pragma: no cover
    _HAS_WIN32 = False


def _get_ancestor_root(hwnd: int) -> int:
    """GA_ROOT with a ctypes fallback (older pywin32 lacks GetAncestor)."""
    try:
        return win32gui.GetAncestor(hwnd, 2) or hwnd  # GA_ROOT
    except Exception:
        try:
            import ctypes
            return int(ctypes.windll.user32.GetAncestor(hwnd, 2)) or hwnd
        except Exception:
            return hwnd


def window_info_at(x: int, y: int) -> Optional[dict]:
    """Return ``{hwnd, title, process}`` for the top window at a screen point."""
    if not _HAS_WIN32:
        return None
    try:
        hwnd = win32gui.WindowFromPoint((x, y))
        if not hwnd:
            return None
        root = _get_ancestor_root(hwnd)
        title = win32gui.GetWindowText(root)
        _, pid = win32process.GetWindowThreadProcessId(root)
        process = "?"
        try:
            process = psutil.Process(pid).name()
        except Exception:
            pass
        l, t, r, b = win32gui.GetWindowRect(root)
        return {"hwnd": int(root), "title": title or "(untitled)",
                "process": process, "rect": (int(l), int(t), int(r), int(b))}
    except Exception:
        return None


class WindowPicker(QWidget):
    """Fullscreen crosshair overlay; emits :attr:`picked` on click."""

    picked = Signal(int, str, str)   # hwnd, title, process name
    cancelled = Signal()

    def __init__(self):
        super().__init__(None)
        self.setWindowTitle("VoxStream window picker")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
                            Qt.WindowType.Tool | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._target: Optional[dict] = None
        self._lprev = False
        self._eprev = False
        self._rprev = False

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._poll)

    # ------------------------------------------------------------------ #
    def showEvent(self, event) -> None:  # noqa: N802
        screen = self.screen() or self.windowHandle().screen()
        self.setGeometry(screen.geometry())
        super().showEvent(event)
        self._make_hit_transparent()
        self._timer.start()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        super().closeEvent(event)

    def _make_hit_transparent(self) -> None:
        try:
            hwnd = int(self.winId())
            if not hwnd:
                return
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            user32 = ctypes.windll.user32
            getfn = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            setfn = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            style = getfn(hwnd, GWL_EXSTYLE)
            setfn(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _poll(self) -> None:
        if not _HAS_WIN32:
            self.cancel_and_close()
            return
        try:
            pt = win32gui.GetCursorPos()
            l_down = bool(win32api.GetAsyncKeyState(win32con.VK_LBUTTON) & 0x8000)
            r_down = bool(win32api.GetAsyncKeyState(win32con.VK_RBUTTON) & 0x8000)
            esc = bool(win32api.GetAsyncKeyState(win32con.VK_ESCAPE) & 0x8000)
        except Exception:
            self.cancel_and_close()
            return

        info = window_info_at(*pt)
        if info and info.get("hwnd") != int(self.winId() or 0):
            if info != self._target:
                self._target = info
                self.update()
        if l_down and not self._lprev:
            self._confirm()
            return
        if (r_down and not self._rprev) or (esc and not self._eprev):
            self.cancel_and_close()
            return
        self._lprev, self._rprev, self._eprev = l_down, r_down, esc

    def _confirm(self) -> None:
        info = self._target
        self.close()
        if info and info.get("hwnd"):
            self.picked.emit(int(info["hwnd"]), info["title"], info["process"])
        else:
            self.cancelled.emit()

    def cancel_and_close(self) -> None:
        self.close()
        self.cancelled.emit()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.cancel_and_close()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------ #
    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(10, 12, 18, 70))

        dpr = self.devicePixelRatioF() or 1.0
        if self._target:
            l, t, r, b = self._target["rect"]
            rect = QRect(int(l / dpr), int(t / dpr),
                         int((r - l) / dpr), int((b - t) / dpr))
            p.setPen(QPen(QColor(34, 211, 238), 3))
            p.setBrush(QColor(34, 211, 238, 28))
            p.drawRect(rect)
            label = f"{self._target['title']}  [{self._target['process']}]"
        else:
            label = "Move the crosshair over a window..."

        p.setPen(QPen(QColor(34, 211, 238), 1))
        try:
            pt = win32gui.GetCursorPos()
            cx, cy = int(pt[0] / dpr), int(pt[1] / dpr)
        except Exception:
            cx, cy = 40, 40
        p.drawLine(cx - 18, cy, cx - 6, cy)
        p.drawLine(cx + 6, cy, cx + 18, cy)
        p.drawLine(cx, cy - 18, cx, cy - 6)
        p.drawLine(cx, cy + 6, cx, cy + 18)

        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        p.setFont(font)
        tip = label + "   |   click = lock    Esc = cancel"
        rect = p.fontMetrics().boundingRect(tip).adjusted(-10, -6, 10, 6)
        rect.moveCenter(self.rect().center().translated(0, -int(self.height() * 0.4)))
        p.fillRect(rect, QColor(10, 12, 18, 220))
        p.setPen(QColor(226, 232, 240))
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, tip)
        p.end()
