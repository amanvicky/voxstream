"""VoxStream main entry point.

* Enforces the offline environment (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE)
  before anything else happens.
* Spawns ``backend.server`` as a sidecar process and waits for its /health.
* Owns the PySide6 UI: floating HUD, system tray + global hotkey
  (Ctrl+Shift+Space), crosshair window picker, enrollment dialog.
* Bridges WebSocket events between the backend and the UI.

Run:  python app.py     (models must exist: python setup_models.py once)
"""

from __future__ import annotations

import os

# --- Offline enforcement: before ANY model-capable import -------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import json  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"websockets.*")

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend import APP_VERSION, CONFIG, DATA_DIR, LOGS_DIR, ROOT  # noqa: E402
from backend import startup_registry  # noqa: E402

from PySide6.QtCore import QObject, Qt, QTimer, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import websockets  # noqa: E402
from websockets.sync.client import connect as ws_connect  # noqa: E402

from ui.enrollment_dialog import EnrollmentDialog  # noqa: E402
from ui.floating_hud import FloatingHUD  # noqa: E402
from ui.onboarding_wizard import OnboardingWizard  # noqa: E402
from ui.settings_dialog import SettingsDialog  # noqa: E402
from ui.tray_icon import GlobalHotkey, TrayIcon  # noqa: E402
from ui.window_picker import WindowPicker  # noqa: E402

UI_STATE = DATA_DIR / "ui_state.json"
CREATE_NO_WINDOW = 0x08000000


def _load_ui_state() -> dict:
    try:
        if UI_STATE.exists():
            return json.loads(UI_STATE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_ui_state(**kwargs) -> None:
    try:
        state = _load_ui_state()
        state.update(kwargs)
        UI_STATE.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
class WSClient(QObject):
    """Reconnecting synchronous WebSocket client running in a daemon thread."""

    connectedChanged = Signal(bool)
    messageReceived = Signal(dict)

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self._conn = None
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="voxstream-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        conn = self._conn
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

    def send(self, msg: dict) -> None:
        conn = self._conn
        if conn is None:
            return
        try:
            with self._send_lock:
                conn.send(json.dumps(msg, ensure_ascii=False))
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                conn = ws_connect(self.url, open_timeout=5, max_size=16 * 1024 * 1024)
            except Exception:
                time.sleep(1.0)
                continue
            self._conn = conn
            self.connected = True
            self.connectedChanged.emit(True)
            try:
                while not self._stop.is_set():
                    raw = conn.recv(timeout=1.0)
                    if raw is None:
                        continue
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    self.messageReceived.emit(msg)
            except Exception:
                pass
            finally:
                self._conn = None
                if self.connected:
                    self.connected = False
                    self.connectedChanged.emit(False)
                if self._stop.is_set():
                    break


# ---------------------------------------------------------------------------
class Controller(QObject):
    """Application orchestrator: sidecar + HUD + tray + hotkey + dialogs."""

    enrollResultReceived = Signal(dict)

    def __init__(self, app: QApplication):
        super().__init__()
        self.qt_app = app
        host = CONFIG.get("server.host") or "127.0.0.1"
        port = int(CONFIG.get("server.port") or 8765)
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"

        self.proc: subprocess.Popen | None = None
        self._restarts = 0
        self._shutting_down = False
        self._backend_stopping = False

        # UI
        self.hud = FloatingHUD(CONFIG)
        self.tray = TrayIcon({
            "toggle_hud": self.toggle_hud,
            "toggle_listen": self.toggle_listen,
            "pick_window": self.pick_window,
            "enroll": self.open_enrollment,
            "onboarding": self.open_onboarding,
            "settings": self.open_settings,
            "is_startup_enabled": lambda: startup_registry.is_enabled(),
            "toggle_startup": self.set_startup,
            "set_translate": self.set_translate,
            "set_inject": self.set_inject,
            "set_isolation": self.set_isolation,
            "set_llm": self.set_llm,
            "quit": self.shutdown,
        })
        self.tray.set_flags(
            translate=bool(CONFIG.get("translation.enabled", False)),
            inject=bool(CONFIG.get("injection.enabled", True)))
        self.tray.set_isolation(CONFIG.get("voxlock.isolation_mode") or "overlap_voxlock")
        self.tray.set_llm(bool(CONFIG.get("llm_polish_enabled", False)))

        self.ws = WSClient(self.ws_url)
        self.ws.messageReceived.connect(self.on_backend_message)
        self.ws.connectedChanged.connect(self.on_backend_connected)

        self.hotkey = GlobalHotkey(**self._hotkey_cfg())
        self.hotkey.triggered.connect(self.toggle_listen)

        self.enroll_dialog: EnrollmentDialog | None = None
        self.onboarding: OnboardingWizard | None = None
        self.settings_dialog: SettingsDialog | None = None
        self.picker: WindowPicker | None = None

        self._state = {}

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        # HUD -> controller wiring
        self.hud.listenToggled.connect(self.toggle_listen)
        self.hud.pickRequested.connect(self.pick_window)
        self.hud.enrollRequested.connect(self.open_enrollment)
        self.hud.translateToggled.connect(self.set_translate)
        self.hud.injectToggled.connect(self.set_inject)
        self.hud.isolationChanged.connect(self.set_isolation)
        self.hud.llmToggled.connect(self.set_llm)
        self.hud.quitRequested.connect(self.shutdown)

        self.hud.show()
        self.hud.set_state("starting")
        self.tray.notify("VoxStream", "Starting offline backend (loading models)...")
        self._spawn_backend()
        threading.Thread(target=self._health_loop, name="voxstream-health",
                         daemon=True).start()

        # Watchdog: restart the sidecar if it dies unexpectedly.
        self._proc_timer = QTimer(self)
        self._proc_timer.setInterval(2000)
        self._proc_timer.timeout.connect(self._check_proc)
        self._proc_timer.start()

        self.hotkey.start()
        self.ws.start()

        # First-run onboarding: let the UI settle, then offer the wizard.
        QTimer.singleShot(1500, self._maybe_first_run)

    def _hotkey_cfg(self) -> dict:
        hk = CONFIG.get("hotkey", {}) or {}
        return {"mods": tuple(hk.get("mods") or ["ctrl", "shift"]),
                "key": hk.get("key") or "space"}

    # ------------------------------------------------------------------ #
    # backend sidecar
    # ------------------------------------------------------------------ #
    def _spawn_backend(self) -> None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = open(LOGS_DIR / "backend_sidecar.log", "ab", buffering=0)
        if getattr(sys, "frozen", False):
            # Frozen bundle: the backend runs inside the same executable.
            cmd = [sys.executable, "--run-backend",
                   "--port", str(CONFIG.get("server.port") or 8765)]
        else:
            cmd = [sys.executable, "-m", "backend.server",
                   "--port", str(CONFIG.get("server.port") or 8765)]
        self.proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT,
            creationflags=CREATE_NO_WINDOW)
        print(f"[app] backend sidecar spawned (pid {self.proc.pid})")

    def _health_loop(self) -> None:
        deadline = time.time() + 180
        while time.time() < deadline:
            if self._shutting_down:
                return
            if self.proc is not None and self.proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=3) as r:
                    data = json.loads(r.read().decode("utf-8"))
                if data.get("ok"):
                    print("[app] backend healthy, connecting websocket")
                    self.ws.start()
                    return
            except Exception:
                pass
            time.sleep(0.5)
        self._on_backend_failed("backend did not become healthy in time")

    def _check_proc(self) -> None:
        if self._shutting_down or self._backend_stopping or self.proc is None:
            return
        if self.proc.poll() is not None:
            if self._restarts < 3:
                self._restarts += 1
                print(f"[app] backend died, restarting (attempt {self._restarts})")
                self.tray.notify("VoxStream", f"Backend restarted (attempt {self._restarts})")
                self._spawn_backend()
            else:
                self._on_backend_failed("backend process died repeatedly")

    def _on_backend_failed(self, reason: str) -> None:
        if self._shutting_down:
            return
        self.hud.set_state("error", f"Backend error: {reason}")
        self.tray.set_state("error")
        QTimer.singleShot(0, lambda: self._error_dialog(reason))

    def _error_dialog(self, reason: str) -> None:
        if getattr(self, "_error_shown", False):
            return
        self._error_shown = True
        QMessageBox.warning(
            None, "VoxStream backend error",
            f"The backend sidecar failed: {reason}\n\n"
            "Check logs/backend.log and make sure:\n"
            "  1. Dependencies are installed (pip install -r requirements.txt)\n"
            "  2. Models are downloaded (python setup_models.py)\n\n"
            "The UI stays available; the backend retries automatically.")

    # ------------------------------------------------------------------ #
    # WS -> UI routing
    # ------------------------------------------------------------------ #
    def on_backend_connected(self, connected: bool) -> None:
        if connected:
            print("[app] websocket connected")
        elif not self._shutting_down:
            self.hud.set_state("error", "Backend connection lost - retrying...")

    def on_backend_message(self, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == "state":
            self._state = msg
            if msg.get("load_error"):
                state = "error"
            elif msg.get("listening"):
                state = "standby" if msg.get("standby") else "listening"
            elif msg.get("models_loaded"):
                state = "ready"
            else:
                state = "starting"
            self.hud.set_state(state)
            self.tray.set_state(state)
            self.tray.set_flags(translate=msg.get("translate"), inject=msg.get("inject_enabled"))
            self.hud.set_toggle_icons(bool(msg.get("listening")))
            self.hud.set_target(msg.get("target_title"))
            # isolation selector + AI polish sync (no signal loops)
            self.hud.set_isolation(str(msg.get("isolation_mode") or "off"))
            self.tray.set_isolation(msg.get("isolation_effective") or msg.get("isolation_mode"))
            self.hud.set_llm(bool(msg.get("llm_polish", False)))
            self.tray.set_llm(bool(msg.get("llm_polish", False)))
        elif mtype == "meter":
            self.hud.set_meter(float(msg.get("rms") or 0.0))
        elif mtype == "partial":
            self.hud.set_partial(str(msg.get("text") or ""))
        elif mtype == "final":
            self.hud.set_partial("")
            self.hud.set_final(str(msg.get("text") or ""),
                               bool(msg.get("translated")),
                               bool(msg.get("polished")))
        elif mtype == "rejected":
            self.hud.set_partial("")
            self.hud.set_rejected(msg.get("similarity"))
        elif mtype == "command_executed":
            self.hud.set_command(str(msg.get("name") or ""))
        elif mtype == "request_pick_window":
            self.pick_window()
        elif mtype == "enroll_result":
            self.enrollResultReceived.emit(msg)
            if self.enroll_dialog is not None:
                self.enroll_dialog.on_enroll_result(msg)
            if self.onboarding is not None:
                self.onboarding.on_enroll_result(msg)
        elif mtype == "error":
            self.hud.set_final(f"error: {msg.get('message', '')}")
            self.tray.notify("VoxStream", str(msg.get("message", "")), error=True)
        elif mtype == "info":
            self.tray.notify("VoxStream", str(msg.get("message", "")))
        elif mtype == "bye":
            # backend shut down intentionally (Exit / explicit shutdown command):
            # let the watchdog know not to restart it
            self._backend_stopping = True

    # ------------------------------------------------------------------ #
    # UI actions -> WS
    # ------------------------------------------------------------------ #
    def toggle_hud(self) -> None:
        self.hud.setVisible(not self.hud.isVisible())

    def toggle_listen(self) -> None:
        mtype = "stop" if self._state.get("listening") else "start"
        self.ws.send({"type": mtype})

    def start_listen(self) -> None:
        self.ws.send({"type": "start"})

    def set_translate(self, enabled: bool) -> None:
        self.ws.send({"type": "set_translate", "enabled": bool(enabled)})

    def set_inject(self, enabled: bool) -> None:
        self.ws.send({"type": "set_inject", "enabled": bool(enabled)})

    def set_isolation(self, mode: str) -> None:
        self.ws.send({"type": "set_isolation", "mode": str(mode)})

    def set_llm(self, enabled: bool) -> None:
        self.ws.send({"type": "set_llm", "enabled": bool(enabled)})

    # -- window picking ---------------------------------------------------- #
    def pick_window(self) -> None:
        if self.picker is not None and self.picker.isVisible():
            return
        self.picker = WindowPicker()
        self.picker.picked.connect(self._on_picked)
        self.picker.cancelled.connect(lambda: None)
        self.picker.show()

    def _on_picked(self, hwnd: int, title: str, process: str) -> None:
        print(f"[app] locked target: hwnd={hwnd} title={title!r} proc={process}")
        _save_ui_state(target_hwnd=hwnd, target_title=title)
        self.ws.send({"type": "set_target", "hwnd": int(hwnd), "title": title})

    # -- enrollment ----------------------------------------------------------- #
    def open_enrollment(self) -> None:
        if self.enroll_dialog is None:
            self.enroll_dialog = EnrollmentDialog()
            self.enroll_dialog.enrollRequested.connect(self._on_enroll_request)
            self.enroll_dialog.setProfileRequested.connect(
                lambda name: self.ws.send({"type": "set_profile", "name": name}))
            self.enroll_dialog.deleteProfileRequested.connect(
                lambda name: self.ws.send({"type": "delete_profile", "name": name}))
        self.enroll_dialog.refresh_profiles()
        self.enroll_dialog.show()
        self.enroll_dialog.raise_()
        self.enroll_dialog.activateWindow()

    def _on_enroll_request(self, name: str, wav_path: str) -> None:
        self.ws.send({"type": "enroll", "name": name, "wav_path": wav_path})

    # -- onboarding wizard ------------------------------------------------------ #
    def _maybe_first_run(self) -> None:
        if not bool(CONFIG.get("first_run_completed", False)):
            self.open_onboarding()

    def open_onboarding(self) -> None:
        if self.onboarding is not None and self.onboarding.isVisible():
            self.onboarding.raise_()
            return
        self.onboarding = OnboardingWizard(
            on_enroll=self._on_enroll_request,
            on_pick_window=self.pick_window,
            on_finished=self._on_onboarding_finished)
        self.onboarding.show()

    def _on_onboarding_finished(self) -> None:
        # Push wizard-made settings (device, isolation, ...) to the backend.
        self.ws.send({"type": "refresh_settings"})

    # -- settings dialog ------------------------------------------------------------ #
    def open_settings(self) -> None:
        if self.settings_dialog is not None and self.settings_dialog.isVisible():
            self.settings_dialog.raise_()
            return
        CONFIG.reload()  # pick up any backend-side writes first
        self.settings_dialog = SettingsDialog(on_saved=self._on_settings_saved)
        self.settings_dialog.show()

    def _on_settings_saved(self) -> None:
        """Re-register the global hotkey and let the backend re-apply config."""
        try:
            self.hotkey.stop()
        except Exception:
            pass
        self.hotkey = GlobalHotkey(**self._hotkey_cfg())
        self.hotkey.triggered.connect(self.toggle_listen)
        self.hotkey.start()
        self.ws.send({"type": "refresh_settings"})
        self.tray.notify("VoxStream", "Settings applied.")

    # -- "Start with Windows" ------------------------------------------------------- #
    def set_startup(self, enabled: bool) -> None:
        ok = startup_registry.set_enabled(bool(enabled))
        self.tray.act_startup.blockSignals(True)
        self.tray.act_startup.setChecked(startup_registry.is_enabled())
        self.tray.act_startup.blockSignals(False)
        if ok:
            self.tray.notify("VoxStream",
                             "Will start with Windows." if enabled
                             else "Removed from Windows startup.")
        else:
            self.tray.notify("VoxStream", "Could not update the startup entry.",
                             error=True)

    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        print("[app] shutting down")
        try:
            self.hotkey.stop()
        except Exception:
            pass
        try:
            self.ws.send({"type": "shutdown"})
        except Exception:
            pass
        time.sleep(0.2)
        try:
            self.ws.stop()
        except Exception:
            pass
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=8)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.tray.hide()
        self.hud.close()
        self.qt_app.quit()


# ---------------------------------------------------------------------------
def main() -> int:
    # Frozen windowed builds can have no real stdout/stderr.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")

    # Frozen sidecar mode: "VoxStream.exe --run-backend [--port N]" runs the
    # backend server in this process (spawned by the UI process) and exits.
    if "--run-backend" in sys.argv:
        argv = [a for a in sys.argv[1:] if a != "--run-backend"]
        from backend.server import main as server_main
        return server_main(argv)

    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl+C in console
    app = QApplication(sys.argv)
    app.setApplicationName("VoxStream")
    app.setApplicationVersion(APP_VERSION)
    app.setQuitOnLastWindowClosed(False)

    controller = Controller(app)
    controller.start()
    rc = app.exec()

    if not controller._shutting_down:
        controller.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
