import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

import ACC_v1_3 as acc
from acc_task_state import CodexState as State, CurrentTask, DesktopSnapshot, handoff_mtime


class StateTests(unittest.TestCase):
    def task(self):
        return CurrentTask(project="test", task_start_ns=100, handoff_mtime_before=50)

    def test_handoff_timestamp_rules(self):
        task = self.task()
        for stamp in (None, 50, 75):
            self.assertFalse(task.new_handoff(stamp))
        for stamp in (100, 101):
            self.assertTrue(task.new_handoff(stamp))
        task.handoff_mtime_before = None
        self.assertTrue(task.new_handoff(101))

    def test_handoff_stronger_than_busy_ui(self):
        state, _ = self.task().observe(DesktopSnapshot(available=True, working=True), 101)
        self.assertEqual(state, State.COMPLETED)

    def test_unknown_or_idle_without_busy_never_completes(self):
        task = self.task()
        for _ in range(20):
            self.assertEqual(task.observe(DesktopSnapshot(), None)[0], State.WORKING)
            self.assertEqual(task.observe(DesktopSnapshot(available=True, input_ready=True), None)[0], State.WORKING)

    def test_two_idle_observations_after_working(self):
        task = self.task()
        busy = DesktopSnapshot(available=True, working=True)
        idle = DesktopSnapshot(available=True, input_ready=True)
        self.assertEqual(task.observe(busy, None)[0], State.WORKING)
        self.assertEqual(task.observe(idle, None)[0], State.WORKING)
        self.assertEqual(task.observe(DesktopSnapshot(), None)[0], State.WORKING)
        self.assertEqual(task.observe(idle, None)[0], State.WORKING)
        self.assertEqual(task.observe(idle, None)[0], State.COMPLETED)

    def test_stale_limit_ignored_and_busy_overrides_limit(self):
        task = self.task()
        task.baseline_messages = ("You've hit your usage limit",)
        self.assertEqual(task.observe(DesktopSnapshot(available=True, messages=task.baseline_messages), None)[0], State.WORKING)
        snapshot = DesktopSnapshot(available=True, working=True, messages=("usage limit reached",))
        self.assertEqual(task.observe(snapshot, None)[0], State.WORKING)

    def test_fresh_limit_combines_separate_reset_text(self):
        task = self.task()
        state, detail = task.observe(DesktopSnapshot(available=True, messages=(
            "You've hit your usage limit", "reset in 15 minutes")), None)
        self.assertEqual(state, State.WAITING)
        self.assertIsNotNone(acc.ACCApp.parse_reset_time(detail))

    def test_failure_and_ordinary_discussion(self):
        task = self.task()
        self.assertEqual(task.observe(DesktopSnapshot(available=True, messages=("Implemented usage limit detection",)), None)[0], State.WORKING)
        self.assertEqual(task.observe(DesktopSnapshot(available=True, messages=("Something went wrong",)), None)[0], State.ERROR)

    def test_journal_roundtrip(self):
        task = self.task()
        task.state = State.WORKING
        task.observed_working = True
        self.assertEqual(CurrentTask.from_dict(json.loads(json.dumps(task.to_dict()))), task)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        config = patch.object(acc, "CONFIG_FILE", self.project / "settings.json")
        config.start()
        self.addCleanup(config.stop)
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.app = acc.ACCApp(self.root)
        self.app.project_var.set(str(self.project))
        self.app.desktop_wait_var.set("0")
        self.app.find_desktop_codex_window = Mock(return_value=(True, "FOUND|123|456|Codex"))
        self.app.activate_and_send_to_desktop = Mock(return_value=(True, "SENT"))
        self.app.inspect_desktop_task = Mock(return_value=DesktopSnapshot())

    def handoff(self):
        path = self.project / acc.HANDOFF_FILE_NAME
        path.write_text("Finished current task", encoding="utf-8")
        stamp = max(self.app.current_task.task_start_ns + 1, path.stat().st_mtime_ns)
        os.utime(path, ns=(stamp, stamp))

    def stop_wait(self, seconds):
        self.app.stop_event.set()
        return True

    def test_send_does_not_complete_and_stop_resume_does_not_resend(self):
        self.app.monitor_wait = Mock(side_effect=self.stop_wait)
        result = self.app.run_desktop_codex_once(self.project)
        self.assertEqual(result[0], "stopped")
        self.assertTrue(self.app.current_task.pending)
        self.assertIsNotNone(self.app.current_task.prompt_sent_time)
        self.assertEqual(self.app.codex_state, State.STOPPED)
        self.app.stop_event.clear()
        self.handoff()
        self.assertEqual(self.app.run_desktop_codex_once(self.project)[0], "success")
        self.assertFalse(self.app.current_task.pending)
        self.app.activate_and_send_to_desktop.assert_called_once()

    def test_restart_restores_pending_task(self):
        self.app.monitor_wait = Mock(side_effect=self.stop_wait)
        self.app.run_desktop_codex_once(self.project)
        task = self.app.current_task
        self.app.current_task = None
        self.app.load_current_task()
        self.assertEqual(self.app.current_task, task)
        self.assertTrue(self.app.current_task.pending)

    def test_preexisting_working_task_is_monitored_without_sending(self):
        self.app.inspect_desktop_task.return_value = DesktopSnapshot(available=True, working=True)
        self.app.monitor_wait = Mock(side_effect=self.stop_wait)
        self.app.run_desktop_codex_once(self.project)
        self.app.activate_and_send_to_desktop.assert_not_called()
        self.assertTrue(self.app.current_task.observed_working)

    def test_ambiguous_delivery_never_retries_blindly(self):
        self.app.activate_and_send_to_desktop.return_value = (False, "helper timeout")
        self.app.monitor_wait = Mock(side_effect=self.stop_wait)
        self.assertEqual(self.app.run_desktop_codex_once(self.project)[0], "stopped")
        self.assertTrue(self.app.current_task.pending)
        self.assertIsNone(self.app.current_task.prompt_sent_time)
        self.app.activate_and_send_to_desktop.assert_called_once()

    def test_handoff_completion_skips_ui_probe_and_loads_panel(self):
        self.app.current_task = CurrentTask(project=str(self.project))
        self.handoff()
        self.assertEqual(self.app.monitor_desktop_task(self.project)[0], "success")
        self.app.inspect_desktop_task.assert_not_called()
        while not self.app.ui_queue.empty():
            self.app.ui_queue.get_nowait()()
        self.assertEqual(self.app.last_handoff_text, "Finished current task")
        self.assertIn("DESKTOP TASK COMPLETED", (self.project / acc.HISTORY_FILE_NAME).read_text())

    def test_interval_only_after_observed_completion(self):
        self.app.inspect_desktop_task.side_effect = [
            DesktopSnapshot(available=True, input_ready=True),
            DesktopSnapshot(available=True, working=True),
            DesktopSnapshot(available=True, input_ready=True),
            DesktopSnapshot(available=True, input_ready=True)]
        self.app.monitor_wait = Mock(return_value=False)
        self.app.stop_event.wait = Mock(return_value=True)
        self.app.session_started_at = acc.datetime.now()
        self.app.controller_loop()
        self.app.activate_and_send_to_desktop.assert_called_once()
        self.assertEqual(self.app.monitor_wait.call_count, 2)
        self.app.stop_event.wait.assert_called_once_with(3600)
        self.assertEqual(self.app.codex_state, State.COMPLETED)
        self.assertIsNotNone(self.app.current_task.next_attempt_time)

    def test_limit_schedules_reset_plus_safety(self):
        self.app.inspect_desktop_task.side_effect = [DesktopSnapshot(), DesktopSnapshot(
            available=True, messages=("You've hit your usage limit", "reset in 15 minutes"))]
        self.app.stop_event.wait = Mock(return_value=True)
        self.app.session_started_at = acc.datetime.now()
        self.app.controller_loop()
        delay = self.app.stop_event.wait.call_args.args[0]
        self.assertTrue(918 < delay <= 920, delay)
        self.assertEqual(self.app.codex_state, State.WAITING)
        self.assertFalse(self.app.current_task.pending)

    def test_error_remains_pending_and_no_automatic_retry(self):
        self.app.inspect_desktop_task.side_effect = [DesktopSnapshot(), DesktopSnapshot(
            available=True, messages=("Task failed: connection lost",))]
        self.app.session_started_at = acc.datetime.now()
        self.app.controller_loop()
        self.assertEqual(self.app.codex_state, State.ERROR)
        self.assertTrue(self.app.current_task.pending)
        self.app.activate_and_send_to_desktop.assert_called_once()

    def test_missing_probe_and_deadline_do_not_send_duplicates(self):
        self.app.inspect_desktop_task.return_value = DesktopSnapshot(detail="pywinauto missing")
        self.app.monitor_wait = Mock(side_effect=self.stop_wait)
        self.app.run_desktop_codex_once(self.project)
        self.app.activate_and_send_to_desktop.assert_called_once()
        self.app.stop_event.clear()
        self.app.deadline = acc.datetime.now()
        self.assertEqual(self.app.run_desktop_codex_once(self.project)[0], "stopped")
        self.app.activate_and_send_to_desktop.assert_called_once()

    def test_poll_validation_and_old_settings_default(self):
        self.assertEqual(self.app.poll_seconds_var.get(), "30")
        for value in ("nan", "inf", "0", "-1", "3601", "x"):
            self.app.poll_seconds_var.set(value)
            with self.assertRaises(ValueError):
                self.app.validate_inputs()

    def test_worker_monitoring_does_not_touch_tk(self):
        config = self.app.capture_config()
        self.app.monitor_wait = Mock(side_effect=self.stop_wait)
        results = []
        worker = threading.Thread(target=self.app.run_background,
            args=(lambda: results.append(self.app.run_desktop_codex_once(self.project)), config))
        worker.start()
        worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results[0][0], "stopped")

    def test_stopped_guard_prevents_submission(self):
        self.app.stop_event.set()
        self.assertEqual(self.app.run_desktop_codex_once(self.project)[0], "stopped")
        self.app.activate_and_send_to_desktop.assert_not_called()

    def test_cancellable_helper_timeout(self):
        with self.assertRaises(TimeoutError):
            self.app.run_cancellable_process([sys.executable, "-c", "import time; time.sleep(30)"], "", 0.15)

    def test_corrupt_journal_fails_closed(self):
        self.app.task_file.write_text("bad json", encoding="utf-8")
        self.app.load_current_task()
        self.assertIn("status is unknown", self.app.task_load_error)


if __name__ == "__main__":
    unittest.main()
