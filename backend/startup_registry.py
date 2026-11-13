"""Windows "Start with Windows" registry helper (HKCU Run key).

The toggle lives under ``HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run``:

* frozen release  -> ``"<path>\\VoxStream.exe"``
* source checkout -> ``"<path>\\start_voxstream.bat"`` (silent pythonw launcher)

``sub_key`` is injectable so tests can exercise the exact same code path
against a throwaway key (``Software\\VoxStreamTest\\Run``) without ever
touching the user's real startup entry.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:
    import winreg
except ImportError:  # pragma: no cover - non-Windows
    winreg = None

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
APP_NAME = "VoxStream"


def _root() -> int:
    return winreg.HKEY_CURRENT_USER


def startup_command() -> str:
    """The command line that should run at logon for this install."""
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}"'
    bat = Path(__file__).resolve().parent.parent / "start_voxstream.bat"
    return f'"{bat}"'


def is_enabled(sub_key: str = RUN_KEY) -> bool:
    if winreg is None:
        return False
    try:
        with winreg.OpenKey(_root(), sub_key, 0, winreg.KEY_READ) as key:
            winreg.QueryValueEx(key, APP_NAME)
            return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def get_command(sub_key: str = RUN_KEY) -> Optional[str]:
    if winreg is None:
        return None
    try:
        with winreg.OpenKey(_root(), sub_key, 0, winreg.KEY_READ) as key:
            value, _type = winreg.QueryValueEx(key, APP_NAME)
            return str(value)
    except OSError:
        return None


def enable(sub_key: str = RUN_KEY) -> bool:
    if winreg is None:
        return False
    try:
        with winreg.CreateKey(_root(), sub_key) as key:
            winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, startup_command())
        return True
    except OSError as exc:
        print(f"[startup] enable failed: {exc}")
        return False


def disable(sub_key: str = RUN_KEY) -> bool:
    if winreg is None:
        return False
    try:
        with winreg.OpenKey(_root(), sub_key, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, APP_NAME)
        return True
    except FileNotFoundError:
        return True  # already gone
    except OSError as exc:
        print(f"[startup] disable failed: {exc}")
        return False


def set_enabled(enabled: bool, sub_key: str = RUN_KEY) -> bool:
    return enable(sub_key) if enabled else disable(sub_key)
