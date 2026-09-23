"""Activity-file (schema 1) tests — device-free, the tracker is pure Python.

Covers the contract the dashboard reader is written against: union-of-intervals
busy accounting across overlapping calls, the in-flight portion counted at write
time, the exact file schema (keys, int types, no BOM), the directory override,
the disable switch, a PermissionError on os.replace being swallowed and retried,
the final exit write with inflight 0, and an exception inside a timed call still
ending its interval. A static check pins that every device call site in
hailo_runtime.py / whisper_npu.py goes through the tracker.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

_SERVER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SERVER))

import accel_activity  # noqa: E402
from accel_activity import ActivityTracker  # noqa: E402

MS = 1_000_000  # ns per ms
KEYS = {"schema", "device", "pid", "started_ms", "updated_ms", "busy_ms", "inflight"}


class FakeClock:
    """Monotonic ns clock the test advances by hand."""

    def __init__(self) -> None:
        self.ns = 0

    def __call__(self) -> int:
        return self.ns

    def at_ms(self, ms: int) -> None:
        self.ns = ms * MS


def _tracker(clock: FakeClock, d: Path, **kw) -> ActivityTracker:
    return ActivityTracker(mono_ns=clock, wall_ms=lambda: 1_700_000_000_000,
                           dir_resolver=lambda: d, **kw)


def _read(d: Path) -> tuple[bytes, dict]:
    raw = (d / "hailo.json").read_bytes()
    return raw, json.loads(raw.decode("utf-8"))


class _TmpDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)
        self.clock = FakeClock()

    def tearDown(self) -> None:
        self._td.cleanup()


class UnionAccounting(_TmpDirCase):
    def test_overlapping_calls_on_two_threads_count_wall_time_once(self) -> None:
        t = _tracker(self.clock, self.dir)
        go = {k: threading.Event() for k in ("a_in", "b_in", "a_out", "b_out")}
        done = {k: threading.Event() for k in go}

        def worker(enter_key: str, exit_key: str) -> None:
            go[enter_key].wait(5)
            t.enter()
            done[enter_key].set()
            go[exit_key].wait(5)
            t.exit()
            done[exit_key].set()

        ta = threading.Thread(target=worker, args=("a_in", "a_out"))
        tb = threading.Thread(target=worker, args=("b_in", "b_out"))
        ta.start()
        tb.start()

        def step(ms: int, key: str) -> None:
            self.clock.at_ms(ms)
            go[key].set()
            self.assertTrue(done[key].wait(5), key)

        # A: [0, 30)   B: [10, 50)   -> union 50 ms, sum would be 70 ms
        step(0, "a_in")
        step(10, "b_in")
        self.assertEqual(t.snapshot(), (10, 2))
        step(30, "a_out")
        self.clock.at_ms(40)
        self.assertEqual(t.snapshot(), (40, 1))
        step(50, "b_out")
        ta.join(5)
        tb.join(5)
        self.assertEqual(t.snapshot(), (50, 0))

        # idle gap is not busy; a later disjoint call adds only its own span
        self.clock.at_ms(100)
        t.enter()
        self.clock.at_ms(120)
        t.exit()
        self.assertEqual(t.snapshot(), (70, 0))

    def test_busy_includes_inflight_portion_at_write_time(self) -> None:
        t = _tracker(self.clock, self.dir)
        t.enter()
        self.clock.at_ms(250)
        self.assertTrue(t.write_once())
        _, doc = _read(self.dir)
        self.assertEqual((doc["busy_ms"], doc["inflight"]), (250, 1))
        self.clock.at_ms(300)
        t.exit()
        self.clock.at_ms(900)
        self.assertTrue(t.write_once())
        _, doc = _read(self.dir)
        self.assertEqual((doc["busy_ms"], doc["inflight"]), (300, 0))

    def test_exception_inside_timed_call_still_ends_interval(self) -> None:
        t = _tracker(self.clock, self.dir)
        with self.assertRaises(RuntimeError):
            with accel_activity._DeviceCall(t):
                self.clock.at_ms(7)
                raise RuntimeError("device timeout")
        self.clock.at_ms(100)
        self.assertEqual(t.snapshot(), (7, 0))


class FileContract(_TmpDirCase):
    def test_write_produces_exact_schema_without_bom(self) -> None:
        t = _tracker(self.clock, self.dir)
        self.assertTrue(t.write_once())
        raw, doc = _read(self.dir)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), "BOM written")
        self.assertEqual(set(doc), KEYS)
        for k, v in doc.items():
            if k == "device":
                continue
            self.assertIs(type(v), int, k)
        self.assertEqual(doc["schema"], 1)
        self.assertEqual(doc["device"], "hailo-8l")
        self.assertEqual(doc["pid"], os.getpid())
        self.assertEqual(doc["started_ms"], 1_700_000_000_000)
        self.assertEqual(os.listdir(self.dir), ["hailo.json"], "temp file left behind")

    def test_env_override_dir_is_honored_and_created(self) -> None:
        target = self.dir / "nested" / "activity"
        with mock.patch.dict(os.environ, {"NVPAIR_ACCEL_ACTIVITY_DIR": str(target)}):
            self.assertEqual(accel_activity.resolve_dir(), target)
            t = ActivityTracker()
            self.assertTrue(t.write_once())
        self.assertEqual(_read(target)[1]["schema"], 1)

    @unittest.skipUnless(os.name == "nt", "Windows default location")
    def test_windows_default_is_programdata(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "NVPAIR_ACCEL_ACTIVITY_DIR"}
        env["ProgramData"] = str(self.dir)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(accel_activity.resolve_dir(),
                             self.dir / "nvpair" / "accel-activity")

    def test_permission_error_on_replace_is_swallowed_and_retried(self) -> None:
        real_replace = os.replace
        fails = {"left": 2}

        def flaky_replace(src, dst):
            if fails["left"] > 0:
                fails["left"] -= 1
                raise PermissionError(13, "reader holds the file open")
            return real_replace(src, dst)

        t = _tracker(self.clock, self.dir, interval_s=0.01)
        with mock.patch.object(accel_activity.os, "replace", side_effect=flaky_replace):
            self.assertFalse(t.write_once())  # no raise, tick skipped
            self.assertFalse((self.dir / "hailo.json").exists())
            self.assertEqual(os.listdir(self.dir), [], "temp file left behind")
            t.start()  # the writer thread retries on its next tick
            deadline = time.monotonic() + 5
            while not (self.dir / "hailo.json").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            t.close()
        self.assertEqual(fails["left"], 0)
        self.assertEqual(_read(self.dir)[1]["schema"], 1)

    def test_close_writes_final_state_with_inflight_zero(self) -> None:
        t = _tracker(self.clock, self.dir)
        t.enter()  # a call still in flight when the process exits
        self.clock.at_ms(40)
        t.close()
        _, doc = _read(self.dir)
        self.assertEqual((doc["busy_ms"], doc["inflight"]), (40, 0))


class ProxyAndSingleton(_TmpDirCase):
    def setUp(self) -> None:
        super().setUp()
        self._saved = (accel_activity._tracker, accel_activity._resolved)
        self.t = _tracker(self.clock, self.dir)
        accel_activity._tracker, accel_activity._resolved = self.t, True

    def tearDown(self) -> None:
        accel_activity._tracker, accel_activity._resolved = self._saved
        super().tearDown()

    def test_timed_infer_times_infer_and_forwards_everything_else(self) -> None:
        clock = self.clock
        events: list[str] = []

        class Pipe:
            name = "pipe"

            def infer(self, feed):
                events.append(f"infer inflight={accel_activity._tracker._inflight}")
                clock.ns += 7 * MS
                if feed == "boom":
                    raise ValueError("boom")
                return {"out": feed}

        class FakeInferVStreams:
            def __enter__(self):
                events.append("enter")
                return Pipe()

            def __exit__(self, *exc):
                events.append(f"exit {exc[0].__name__ if exc[0] else None}")
                return False

        with accel_activity.timed_infer(FakeInferVStreams()) as pipe:
            self.assertEqual(pipe.name, "pipe")
            self.assertEqual(pipe.infer("x"), {"out": "x"})
            self.assertEqual(pipe.infer("y"), {"out": "y"})
        with self.assertRaises(ValueError):
            with accel_activity.timed_infer(FakeInferVStreams()) as pipe:
                pipe.infer("boom")
        self.assertEqual(events, ["enter", "infer inflight=1", "infer inflight=1", "exit None",
                                  "enter", "infer inflight=1", "exit ValueError"])
        self.assertEqual(self.t.snapshot(), (21, 0))


_CHILD = r"""
import sys, time, threading
sys.path.insert(0, sys.argv[1])
import accel_activity
with accel_activity.device_call():
    time.sleep(0.05)
accel_activity.get_tracker() and accel_activity.get_tracker().enter()  # left in flight at exit
time.sleep(0.1)
import os
print(accel_activity.get_tracker() is None,
      any(t.name == "hailo-activity" for t in threading.enumerate()), os.getpid())
"""


class ProcessLifecycle(_TmpDirCase):
    def _run_child(self, **env_extra: str) -> list[str]:
        env = dict(os.environ)
        env.pop("HAILO_ACTIVITY_DISABLE", None)
        env["NVPAIR_ACCEL_ACTIVITY_DIR"] = str(self.dir)
        env.update(env_extra)
        p =subprocess.Popen([sys.executable, "-c", _CHILD, str(_SERVER)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(timeout=60)
        self.assertEqual(p.returncode, 0, err)
        return out.split()

    def test_exit_write_has_inflight_zero_and_counts_busy(self) -> None:
        disabled, has_thread, pid = self._run_child()
        self.assertEqual((disabled, has_thread), ("False", "True"))
        _, doc = _read(self.dir)
        self.assertEqual(doc["pid"], int(pid))  # the child's own os.getpid()
        self.assertEqual(doc["inflight"], 0)
        # 50 ms timed call + >= 100 ms left in flight until exit
        self.assertGreaterEqual(doc["busy_ms"], 140)
        self.assertGreaterEqual(doc["updated_ms"], doc["started_ms"])

    def test_disable_switch_writes_nothing_and_starts_no_thread(self) -> None:
        disabled, has_thread, _ = self._run_child(HAILO_ACTIVITY_DISABLE="1")
        self.assertEqual((disabled, has_thread), ("True", "False"))
        self.assertEqual(os.listdir(self.dir), [])


class CallSitesAreTracked(unittest.TestCase):
    """Every device call in the runtime goes through the tracker — a new
    InferVStreams site that forgets the wrapper would under-report busy."""

    def test_every_infervstreams_is_wrapped(self) -> None:
        src = (_SERVER / "hailo_runtime.py").read_text(encoding="utf-8")
        sites = re.findall(r"^.*hpf\.InferVStreams\(.*$", src, flags=re.M)
        self.assertEqual(len(sites), 8)
        for line in sites:
            self.assertIn("timed_infer(hpf.InferVStreams(", line)

    def test_every_whisper_run_is_timed(self) -> None:
        lines = (_SERVER / "whisper_npu.py").read_text(encoding="utf-8").splitlines()
        runs = [i for i, ln in enumerate(lines) if re.search(r"_cfg\.run\(", ln)]
        self.assertEqual(len(runs), 2)
        for i in runs:
            self.assertEqual(lines[i - 1].strip(), "with device_call():")


if __name__ == "__main__":
    unittest.main()
