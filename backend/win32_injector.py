"""Three-tier Win32 text injector.

Tier 1  ``wm_char``    PostMessage(WM_CHAR) into the target HWND -- types into
                       background windows (Notepad and many editors) without
                       stealing focus.
Tier 2  ``unicode``    SetForegroundWindow + SendInput KEYEVENTF_UNICODE --
                       exact Unicode input for focused windows (works in
                       Chromium/Electron editors that ignore posted WM_CHAR).
Tier 3  ``clipboard``  clipboard swap + Ctrl+V -- fastest bulk path; the
                       previous clipboard contents are restored afterwards
                       (privacy mode), so nothing lingers on the clipboard.

``auto`` mode picks the best tier: if the target already has keyboard focus,
tier 2 is used; otherwise tier 1 first, falling back to tier 2 then tier 3
on hard failures (invalid handle, focus lock, clipboard lock).
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import threading
import time
from typing import Optional

try:
    import win32api  # noqa: F401
    import win32clipboard
    import win32con
    import win32gui
    _HAS_PYWIN32 = True
except ImportError:  # pragma: no cover - Windows only dependency
    _HAS_PYWIN32 = False

_user32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None

WM_CHAR = 0x0102
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
VK_BACK = 0x08
VK_CONTROL = 0x11
VK_V = 0x56
SW_RESTORE = 9

KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002
INPUT_KEYBOARD = 1
GWL_EXSTYLE = -20


# ---------------------------------------------------------------------------
# ctypes SendInput plumbing
# ---------------------------------------------------------------------------
if _user32 is not None:
    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                    ("time", wt.DWORD), ("dwExtraInfo", ctypes.POINTER(wt.ULONG))]

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                    ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                    ("dwExtraInfo", ctypes.POINTER(wt.ULONG))]

    class _HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", wt.DWORD), ("wParamL", wt.WORD), ("wParamH", wt.WORD)]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT), ("hi", _HARDWAREINPUT)]

    class _INPUT(ctypes.Structure):
        _anonymous_ = ("u",)
        _fields_ = [("type", wt.DWORD), ("u", _INPUTUNION)]

    _user32.SendInput.argtypes = (wt.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)


def _kbd_event(vk: int = 0, scan: int = 0, flags: int = 0) -> "_INPUT":
    ev = _INPUT()
    ev.type = INPUT_KEYBOARD
    ev.ki.wVk = vk
    ev.ki.wScan = scan
    ev.ki.dwFlags = flags
    ev.ki.time = 0
    return ev


def send_inputs(events) -> bool:
    if not _user32 or not events:
        return False
    arr = (_INPUT * len(events))(*events)
    sent = _user32.SendInput(len(arr), arr, ctypes.sizeof(_INPUT))
    return int(sent) == len(events)


def unicode_key_events(text: str, char_delay: float = 0.0):
    """Build SendInput events that type ``text`` as unicode scan codes."""
    events = []
    for ch in text:
        code = ord(ch)
        if code > 0xFFFF:  # encode astral chars as UTF-16 surrogate pairs
            code -= 0x10000
            units = (0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF))
        else:
            units = (code,)
        for unit in units:
            events.append(_kbd_event(scan=unit, flags=KEYEVENTF_UNICODE))
            events.append(_kbd_event(scan=unit, flags=KEYEVENTF_UNICODE | 0x0002))
            if char_delay > 0:
                time.sleep(char_delay)
    return events


def press_key_vk(vk: int, delay: float = 0.01) -> bool:
    ok = send_inputs([_kbd_event(vk=vk)])
    if delay > 0:
        time.sleep(delay)
    ok = send_inputs([_kbd_event(vk=vk, flags=0x0002)]) and ok
    return ok


def press_ctrl_v() -> bool:
    seq = [_kbd_event(vk=VK_CONTROL), _kbd_event(vk=VK_V),
           _kbd_event(vk=VK_V, flags=0x0002), _kbd_event(vk=VK_CONTROL, flags=0x0002)]
    return send_inputs(seq)


def press_backspaces(n: int, delay: float = 0.01) -> bool:
    ok = True
    for _ in range(max(0, int(n))):
        ok = press_key_vk(VK_BACK, delay=delay) and ok
    return ok


# ---------------------------------------------------------------------------
# Clipboard helpers (with retry -- the clipboard is frequently locked)
# ---------------------------------------------------------------------------
def get_clipboard_text() -> Optional[str]:
    if not _HAS_PYWIN32:
        return None
    for _ in range(6):
        try:
            win32clipboard.OpenClipboard()
            try:
                if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                    return win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                return None
            finally:
                win32clipboard.CloseClipboard()
        except Exception:
            time.sleep(0.1)
    return None


def set_clipboard_text(text: str) -> bool:
    if not _HAS_PYWIN32:
        return False
    for _ in range(6):
        try:
            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
            finally:
                win32clipboard.CloseClipboard()
            return True
        except Exception:
            time.sleep(0.1)
    return False


def clear_clipboard() -> None:
    if not _HAS_PYWIN32:
        return
    try:
        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
        finally:
            win32clipboard.CloseClipboard()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Focus helpers
# ---------------------------------------------------------------------------
def set_foreground(hwnd: int) -> bool:
    """Bring ``hwnd`` to the foreground, working around the focus lock."""
    if not _HAS_PYWIN32 or not hwnd:
        return False
    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, SW_RESTORE)
            time.sleep(0.05)
    except Exception:
        pass
    try:
        win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception:
        pass
    # Alt-key trick: pressing Alt releases the foreground lock for one call.
    try:
        if _user32:
            _user32.keybd_event(0xA4, 0, 0, 0)
            try:
                win32gui.SetForegroundWindow(hwnd)
                return True
            finally:
                _user32.keybd_event(0xA4, 0, 0x0002, 0)
    except Exception:
        return False
    return False


def get_foreground_hwnd() -> Optional[int]:
    if not _HAS_PYWIN32:
        return None
    try:
        return win32gui.GetForegroundWindow()
    except Exception:
        return None


# Windows shell surfaces where no text input can ever exist.
SHELL_WINDOW_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd",
                        "Shell_SecondaryTrayWnd"}


def foreground_window_class() -> Optional[str]:
    if not _HAS_PYWIN32:
        return None
    try:
        hwnd = get_foreground_hwnd()
        if not hwnd:
            return None
        return win32gui.GetClassName(hwnd)
    except Exception:
        return None


def foreground_accepts_text() -> bool:
    """False when the user is focused on the desktop/taskbar (no text box)."""
    cls = foreground_window_class()
    if cls is None:
        return True  # unknown -> assume a normal app
    return cls not in SHELL_WINDOW_CLASSES


def root_window(hwnd: int) -> Optional[int]:
    if not _HAS_PYWIN32 or not hwnd:
        return None
    try:
        return win32gui.GetAncestor(hwnd, 2) or hwnd  # GA_ROOT
    except Exception:
        return hwnd


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------
class TextInjector:
    """Types text into a locked target window using the configured tier chain."""

    TIERS = ("wm_char", "unicode", "clipboard")

    def __init__(self, config=None):
        self.config = config
        self.target_hwnd: Optional[int] = None
        self.target_title: Optional[str] = None
        self._reload()

    def _reload(self) -> None:
        cfg = (self.config.get("injection", {}) if self.config else {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.tier = str(cfg.get("tier", "auto")).lower()
        self.max_chars = int(cfg.get("max_chars", 5000))
        self.restore_clipboard = bool(cfg.get("restore_clipboard", True))
        self.char_delay = float(cfg.get("char_delay_s", 0.002))

    # -- target ---------------------------------------------------------- #
    def set_target(self, hwnd: int, title: Optional[str] = None) -> None:
        self.target_hwnd = int(hwnd) if hwnd else None
        self.target_title = title

    def clear_target(self) -> None:
        self.target_hwnd = None
        self.target_title = None

    def is_target_valid(self) -> bool:
        if not self.target_hwnd or not _HAS_PYWIN32:
            return False
        try:
            return bool(win32gui.IsWindow(self.target_hwnd))
        except Exception:
            return False

    # -- tiers ------------------------------------------------------------ #
    def _tier_wm_char(self, text: str) -> bool:
        """Post WM_CHAR messages; types into many background windows."""
        if not _HAS_PYWIN32:
            return False
        hwnd = self.target_hwnd
        try:
            for ch in text:
                code = ord("\r") if ch in "\r\n" else ord(ch)
                if code > 0xFFFF:  # surrogate halves for astral characters
                    code -= 0x10000
                    for unit in (0xD800 + (code >> 10), 0xDC00 + (code & 0x3FF)):
                        win32gui.PostMessage(hwnd, WM_CHAR, unit, 0)
                else:
                    win32gui.PostMessage(hwnd, WM_CHAR, code, 0)
            return True
        except Exception:
            return False

    def _tier_unicode(self, text: str) -> bool:
        """Focus the window and send KEYEVENTF_UNICODE events."""
        if not _HAS_PYWIN32:
            return False
        if not set_foreground(self.target_hwnd):
            return False
        time.sleep(0.08)
        events = unicode_key_events(text, char_delay=self.char_delay)
        return send_inputs(events)

    def _tier_clipboard(self, text: str) -> bool:
        """Paste via the clipboard, then restore what was there before."""
        if not _HAS_PYWIN32:
            return False
        backup = get_clipboard_text() if self.restore_clipboard else None
        if not set_clipboard_text(text):
            return False
        focused = set_foreground(self.target_hwnd)
        time.sleep(0.08)
        ok = press_ctrl_v() if focused else False
        if self.restore_clipboard:
            # Restore after the paste has been processed by the target.
            def _restore():
                time.sleep(0.4)
                if backup is None:
                    clear_clipboard()
                else:
                    set_clipboard_text(backup)
            threading.Thread(target=_restore, daemon=True).start()
        return ok

    # -- public API -------------------------------------------------------- #
    def inject(self, text: str) -> Optional[str]:
        """Inject ``text``; returns the tier name used or ``None`` on failure."""
        self._reload()
        if not text or not self.enabled or not self.is_target_valid():
            return None
        text = text[: self.max_chars]
        text = text.replace("\r\n", "\r").replace("\n", "\r")

        if self.tier in self.TIERS:
            fn = {"wm_char": self._tier_wm_char,
                  "unicode": self._tier_unicode,
                  "clipboard": self._tier_clipboard}[self.tier]
            return self.tier if fn(text) else None

        # auto: focused -> unicode first (best fidelity), else background first
        focused = get_foreground_hwnd() == root_window(self.target_hwnd)
        order = ["unicode", "wm_char", "clipboard"] if focused else ["wm_char", "unicode", "clipboard"]
        for tier in order:
            fn = {"wm_char": self._tier_wm_char,
                  "unicode": self._tier_unicode,
                  "clipboard": self._tier_clipboard}[tier]
            try:
                if fn(text):
                    return tier
            except Exception:
                continue
        return None

    def inject_anywhere(self, text: str) -> Optional[str]:
        """Beginner-friendly "Type Anywhere" fallback (no locked target).

        Returns the tier used ("unicode"), the sentinel string
        ``"no_text_box"`` when the focused surface cannot accept text
        (Windows desktop / taskbar), or ``None`` on failure.
        """
        self._reload()
        if not text or not self.enabled:
            return None
        if self.is_target_valid():
            return self.inject(text)
        hwnd = get_foreground_hwnd()
        if not hwnd:
            return None
        try:
            cls = win32gui.GetClassName(hwnd)
        except Exception:
            cls = ""
        if cls in SHELL_WINDOW_CLASSES:
            return "no_text_box"   # nowhere to type - caller shows the banner
        text = text[: self.max_chars].replace("\r\n", "\r").replace("\n", "\r")
        # The window already has keyboard focus: send unicode directly.
        events = unicode_key_events(text, char_delay=self.char_delay)
        return "unicode" if send_inputs(events) else None

    def send_backspaces(self, n: int) -> bool:
        """Delete ``n`` characters in the target (focused: SendInput, else WM_KEYDOWN)."""
        if not self.is_target_valid() or n <= 0:
            return False
        focused = get_foreground_hwnd() == root_window(self.target_hwnd)
        if focused:
            return press_backspaces(n)
        # Background path: WM_KEYDOWN/WM_KEYUP VK_BACK.  lParam repeat count
        # must be 1 -- a zero repeat count makes some EDIT implementations
        # delete far more than one character.
        VK_BACK_SCAN = 0x0E
        ldown = 1 | (VK_BACK_SCAN << 16)
        lup = 1 | (VK_BACK_SCAN << 16) | (1 << 30) | (1 << 31)
        ok = True
        try:
            for _ in range(int(n)):
                win32gui.PostMessage(self.target_hwnd, WM_KEYDOWN, VK_BACK, ldown)
                win32gui.PostMessage(self.target_hwnd, WM_KEYUP, VK_BACK, lup)
        except Exception:
            ok = False
        return ok
