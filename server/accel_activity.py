"""Device-busy duty-cycle file for the Hailo-8L (activity file, schema 1).

HailoRT 4.24 on Windows exposes no busy counter and `hailortcli monitor` is
unsupported there, so a dashboard cannot ask the device how busy it is. The
process running inference is the only thing that knows, so it publishes the
answer: every device call is bracketed by `device_call()` (or the
`timed_infer()` proxy around an InferVStreams pipe), and a daemon thread
rewrites a small JSON file every 500 ms that a reader turns into a percentage
by differencing `busy_ms` against `updated_ms`.

Contract (schema 1) — the reader is written against exactly this:

  path     $NVPAIR_ACCEL_ACTIVITY_DIR, else %ProgramData%\\nvpair\\accel-activity
           on Windows, else /run/nvpair/accel-activity with a fallback to
           $XDG_RUNTIME_DIR/nvpair/accel-activity; file name hailo.json
  content  {"schema":1,"device":"hailo-8l","pid":..,"started_ms":..,
            "updated_ms":..,"busy_ms":..,"inflight":..}  UTF-8, no BOM
  busy_ms  wall time during which >= 1 device call was in flight (the UNION of
           the call intervals, not their sum), including the in-flight portion
           up to updated_ms; measured on a monotonic clock
  disable  HAILO_ACTIVITY_DISABLE=1 -> no thread, no file

The writer thread starts lazily on the FIRST tracked call, not when the
VDevice opens: it is the one hook shared by every device path (the runtime's
InferVStreams pipes and whisper's separate InferModel VDevice), it keeps the
runtime's VDevice code untouched, and a process that never inferred writes no
file, so a reader never sees a file for a process that has not used the device.

Nothing here may raise into an inference call or slow it: the inference path
only updates two counters under a lock (plus a one-time thread start). All I/O,
including directory resolution and creation, happens on the writer thread or at
exit, and every failure there is swallowed and retried on the next tick.
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

SCHEMA = 1
DEVICE = "hailo-8l"
FILE_NAME = "hailo.json"
INTERVAL_S = 0.5

_NS_PER_MS = 1_000_000


def _mono_ns() -> int:
    # perf_counter is monotonic and high-resolution on every platform, where
    # time.monotonic() on Windows before Python 3.13 ticks at ~15.6 ms — too
    # coarse for device calls that take a few milliseconds each.
    return time.perf_counter_ns()


def _wall_ms() -> int:
    return int(time.time() * 1000)


def resolve_dir() -> Path:
    """The activity directory per the contract. Creates nothing except when
    probing /run on Linux (which needs the mkdir to know it is writable)."""
    env = os.environ.get("NVPAIR_ACCEL_ACTIVITY_DIR")
    if env:
        return Path(env)
    if os.name == "nt":
        base = os.environ.get("ProgramData") or r"C:\ProgramData"
        return Path(base) / "nvpair" / "accel-activity"
    run = Path("/run/nvpair/accel-activity")
    try:
        run.mkdir(parents=True, exist_ok=True)
        if os.access(run, os.W_OK):
            return run
    except OSError:
        pass
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "nvpair" / "accel-activity"
    return run


class ActivityTracker:
    """Union-of-intervals busy accounting plus the file writer.

    `mono_ns`, `wall_ms` and `dir_resolver` are injectable so the accounting
    can be tested deterministically without a device or a real clock."""

    def __init__(
        self,
        *,
        mono_ns: Callable[[], int] = _mono_ns,
        wall_ms: Callable[[], int] = _wall_ms,
        dir_resolver: Callable[[], Path] = resolve_dir,
        interval_s: float = INTERVAL_S,
    ) -> None:
        self._mono_ns = mono_ns
        self._wall_ms = wall_ms
        self._dir_resolver = dir_resolver
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._inflight = 0
        self._busy_ns = 0  # closed busy periods
        self._busy_since_ns = 0  # start of the open busy period (inflight >= 1)
        self.started_ms = wall_ms()
        self._pid = os.getpid()
        self._dir: Path | None = None
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False

    # -- accounting (the only part on the inference path) -----------------

    def enter(self) -> None:
        with self._lock:
            if self._inflight == 0:
                self._busy_since_ns = self._mono_ns()
            self._inflight += 1

    def exit(self) -> None:
        with self._lock:
            if self._inflight <= 0:
                return
            self._inflight -= 1
            if self._inflight == 0:
                self._busy_ns += self._mono_ns() - self._busy_since_ns

    def snapshot(self) -> tuple[int, int]:
        """(busy_ms, inflight), busy including the open period up to now."""
        with self._lock:
            busy = self._busy_ns
            if self._inflight > 0:
                busy += self._mono_ns() - self._busy_since_ns
            return busy // _NS_PER_MS, self._inflight

    # -- writer ------------------------------------------------------------

    def payload(self, *, final: bool = False) -> dict[str, Any]:
        busy_ms, inflight = self.snapshot()
        return {
            "schema": SCHEMA,
            "device": DEVICE,
            "pid": self._pid,
            "started_ms": self.started_ms,
            "updated_ms": self._wall_ms(),
            "busy_ms": busy_ms,
            "inflight": 0 if final else inflight,
        }

    def write_once(self, *, final: bool = False) -> bool:
        """One atomic write (temp file in the same dir, then os.replace).
        Returns False on any failure — never raises. A PermissionError from
        os.replace (a Windows reader holding the file open) just skips this
        tick; the next tick writes the newer state anyway."""
        with self._write_lock:
            try:
                if self._dir is None:
                    d = self._dir_resolver()
                    try:
                        d.mkdir(parents=True, exist_ok=True)
                    except OSError:
                        pass  # best effort; the write below reports the outcome
                    self._dir = d
                data = json.dumps(self.payload(final=final), separators=(",", ":")).encode("utf-8")
                tmp = self._dir / f".{FILE_NAME}.{self._pid}.tmp"
                with open(tmp, "wb") as f:
                    f.write(data)
                try:
                    os.replace(tmp, self._dir / FILE_NAME)
                except PermissionError:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    return False
                return True
            except Exception:
                return False

    def _run(self) -> None:
        while True:
            self.write_once()
            if self._stop.wait(self._interval_s):
                return

    def start(self) -> None:
        """Start the writer thread and register the exit write. Idempotent;
        never raises."""
        if self._thread is not None:
            return
        try:
            t = threading.Thread(target=self._run, name="hailo-activity", daemon=True)
            t.start()
            self._thread = t
            atexit.register(self.close)
        except Exception:
            pass

    def close(self) -> None:
        """Stop the thread and write the final state with inflight 0."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        self.write_once(final=True)


class _DeviceCall:
    """Context manager bracketing one device call. The interval always ends,
    also when the call raises; the exception propagates unchanged."""

    __slots__ = ("_t",)

    def __init__(self, tracker: ActivityTracker | None) -> None:
        self._t = tracker

    def __enter__(self) -> None:
        if self._t is not None:
            self._t.enter()

    def __exit__(self, *exc: object) -> bool:
        if self._t is not None:
            self._t.exit()
        return False


_tracker: ActivityTracker | None = None
_resolved = False
_init_lock = threading.Lock()


def get_tracker() -> ActivityTracker | None:
    """The process-wide tracker, created (and its thread started) on first
    use; None when HAILO_ACTIVITY_DISABLE=1."""
    global _tracker, _resolved
    if _resolved:
        return _tracker
    with _init_lock:
        if not _resolved:
            try:
                if os.environ.get("HAILO_ACTIVITY_DISABLE", "").strip() != "1":
                    t = ActivityTracker()
                    t.start()
                    _tracker = t
            except Exception:
                _tracker = None
            _resolved = True
    return _tracker


def device_call() -> _DeviceCall:
    """`with device_call(): <one device call>` — times it into the tracker."""
    return _DeviceCall(get_tracker())


class _TimedPipe:
    """Proxy for an entered InferVStreams pipe: .infer is timed, everything
    else passes through."""

    __slots__ = ("_pipe",)

    def __init__(self, pipe: Any) -> None:
        self._pipe = pipe

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        with device_call():
            return self._pipe.infer(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pipe, name)


class timed_infer:
    """`with timed_infer(hpf.InferVStreams(...)) as pipe:` — the pipe's
    .infer calls count as device-busy time; entering/exiting the pipe is
    forwarded unchanged."""

    __slots__ = ("_cm",)

    def __init__(self, cm: Any) -> None:
        self._cm = cm

    def __enter__(self) -> _TimedPipe:
        return _TimedPipe(self._cm.__enter__())

    def __exit__(self, *exc: Any) -> Any:
        return self._cm.__exit__(*exc)
