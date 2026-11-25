import copy
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from acc_github_bridge import BridgeError, StateStore
from acc_codex_inbox import Inbox, MARKER, codex_environment, load_targets, parse_message, redact

SESSION = "11111111-1111-4111-8111-111111111111"


def message(number, action="task", target="test", **values):
    job = {"runner": "test-runner", "action": action, "target": target}
    if action == "task":
        job["prompt"] = "Do one test"
    job.update(values)
    return {"id": number, "user": {"login": "owner"},
            "body": MARKER + "\n```json\n" + json.dumps(job) + "\n```"}


class Client:
    repo = "owner/test"
    _token = "fake-secret-token"

    def __init__(self):
        self.comments = []
        self.reads = 0

    def request(self, *args, **kwargs):
        return {"state": "open"}

    def inbox_comments(self, number):
        self.reads += 1
        return copy.deepcopy(self.comments)

    def api(self, method, path):
        key = int(path.rsplit("/", 1)[1])
        return copy.deepcopy(next(c for c in self.comments if c["id"] == key))

    def comment(self, issue, body):
        self.comments.append({"id": 10000 + len(self.comments), "body": body, "user": {"login": "owner"}})


class Executor:
    def __init__(self):
        self.calls = []
        self.finish = threading.Event()
        self.entered = threading.Event()

    def execute(self, target, session, prompt, cancel, notify):
        self.calls.append((target, session, prompt))
        self.entered.set()
        notify("session", SESSION)
        while not self.finish.wait(0.01):
            if cancel.is_set():
                return {"state": "CANCELLED", "session_id": SESSION, "result": "Cancelled"}
        return {"state": "COMPLETED", "session_id": SESSION, "result": "Test result"}


class InboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = StateStore(self.root / "journal.json")
        self.addCleanup(self.store.close)
        self.client = Client()
        self.executor = Executor()
        self.inbox = Inbox(self.client, "owner", "test-runner", 1,
                           {"test": {"project": str(self.root)}}, self.store, self.executor, True, lambda _: None)
        self.addCleanup(self.finish)

    def finish(self):
        self.executor.finish.set()
        self.inbox.cancel.set()
        if self.inbox.thread:
            self.inbox.thread.join(2)
        self.inbox.drain()
        if self.inbox.project_lock:
            self.inbox.project_lock.close()

    def test_idle_polling_never_invokes_codex(self):
        for _ in range(20):
            self.inbox.tick()
        self.assertEqual(self.client.reads, 20)
        self.assertEqual(len(self.executor.calls), 0)

    def test_changed_alias_binding_is_rejected(self):
        with self.assertRaises(BridgeError):
            Inbox(self.client, "owner", "test-runner", 1,
                  {"test": {"project": str(self.root), "session_id": SESSION}},
                  self.store, self.executor)

    def test_unknown_blocks_another_alias_for_same_project(self):
        self.inbox.targets["second"] = {"project": str(self.root)}
        self.client.comments = [message(1), message(2, target="second")]
        self.inbox.ingest(self.client.comments)
        self.inbox.record("1", "UNKNOWN", "Interrupted")
        self.inbox.dispatch()
        self.assertEqual(self.executor.calls, [])

    def test_local_stop_signal_prevents_poll_and_execution(self):
        stop_path = self.root / "receiver.stop"
        stop_path.write_text("stop")
        stop = threading.Event()
        self.inbox.serve(stop, stop_path=stop_path)
        self.assertTrue(stop.is_set())
        self.assertFalse(stop_path.exists())
        self.assertEqual(self.client.reads, 0)
        self.assertEqual(self.executor.calls, [])

    def test_busy_queue_status_and_session_resume(self):
        self.client.comments = [message(1)]
        self.inbox.tick()
        self.assertTrue(self.executor.entered.wait(1))
        self.client.comments += [message(2), message(3, "status")]
        for _ in range(5):
            self.inbox.tick()
        self.assertEqual(len(self.executor.calls), 1)
        self.assertEqual(self.inbox.messages["2"]["state"], "QUEUED")
        self.assertIn("no model call", self.inbox.messages["3"]["detail"])
        self.assertIn("running=['1']", self.inbox.messages["3"]["detail"])
        self.executor.finish.set()
        self.inbox.thread.join(1)
        self.inbox.tick()
        self.inbox.thread.join(1)
        self.inbox.tick()
        self.assertEqual(len(self.executor.calls), 2)
        self.assertEqual(self.executor.calls[1][1], SESSION)
        self.assertEqual(self.inbox.messages["2"]["state"], "COMPLETED")

    def test_explicit_cancel_active_and_queued(self):
        self.client.comments = [message(1)]
        self.inbox.tick()
        self.executor.entered.wait(1)
        self.client.comments += [message(2), message(3, "cancel")]
        self.inbox.tick()
        self.inbox.thread.join(1)
        self.inbox.tick()
        self.assertEqual(self.inbox.messages["1"]["state"], "CANCELLED")
        self.assertEqual(self.inbox.messages["2"]["state"], "CANCELLED")
        self.assertEqual(len(self.executor.calls), 1)

    def test_changed_queued_message_is_not_executed(self):
        self.inbox.auto = False
        self.client.comments = [message(1)]
        self.inbox.tick()
        self.inbox.approve("1", self.inbox.messages["1"]["digest"])
        self.client.comments[0]["body"] += "\nEdited"
        self.inbox.tick()
        self.assertEqual(self.inbox.messages["1"]["state"], "CANCELLED")
        self.assertEqual(len(self.executor.calls), 0)

    def test_receive_only_and_unknown_target(self):
        self.inbox.auto = False
        self.client.comments = [message(1), message(2, target="unconfigured")]
        self.inbox.tick()
        self.assertEqual(self.inbox.messages["1"]["state"], "RECEIVED")
        self.assertEqual(self.inbox.messages["2"]["state"], "REJECTED")
        self.assertFalse(self.executor.calls)

    def test_restart_running_is_unknown_and_not_replayed(self):
        self.inbox.auto = False
        self.client.comments = [message(1)]
        self.inbox.tick()
        self.inbox.record("1", "RUNNING", "old process")
        self.inbox = Inbox(self.client, "owner", "test-runner", 1, self.inbox.targets,
                           self.store, self.executor, True, lambda _: None)
        self.client.comments.append(message(2))
        self.inbox.tick()
        self.assertEqual(self.inbox.messages["1"]["state"], "UNKNOWN")
        self.assertFalse(self.executor.calls)

    def test_remote_receipt_prevents_replay_after_journal_loss(self):
        self.client.comments = [message(1), {"id": 2, "user": {"login": "owner"},
            "body": "<!-- acc-receipt:test-runner:1:RUNNING -->\nReserved"}]
        self.inbox.tick()
        self.assertEqual(self.inbox.messages["1"]["state"], "UNKNOWN")
        self.assertFalse(self.executor.calls)

    def test_comment_failure_blocks_launch(self):
        self.client.comments = [message(1)]
        original = self.client.comment

        def fail_running(issue, body):
            if ":RUNNING -->" in body:
                raise BridgeError("Network write uncertain")
            return original(issue, body)

        self.client.comment = fail_running
        with self.assertRaises(BridgeError):
            self.inbox.tick()
        self.assertFalse(self.executor.calls)
        self.assertEqual(self.inbox.messages["1"]["state"], "UNKNOWN")

    def test_completed_message_not_repeated(self):
        self.executor.finish.set()
        self.client.comments = [message(1)]
        self.inbox.tick()
        self.inbox.thread.join(1)
        for _ in range(5):
            self.inbox.tick()
        self.assertEqual(len(self.executor.calls), 1)
        completed = [c for c in self.client.comments if ":COMPLETED -->" in c["body"]]
        self.assertEqual(len(completed), 1)

    def test_closed_issue_stops_new_dispatch(self):
        self.client.comments = [message(1)]
        self.client.request = lambda *a, **kw: {"state": "closed"}
        self.inbox.tick()
        self.assertFalse(self.executor.calls)


class ProtocolTests(unittest.TestCase):
    def test_owner_runner_and_reply_filters(self):
        m = message(1)
        self.assertIsNone(parse_message(m, "someone-else", "test-runner"))
        self.assertIsNone(parse_message(m, "owner", "different-pc"))
        self.assertIsNone(parse_message({"user": {"login": "owner"}, "body": "ACC: COMPLETED"}, "owner", "test-runner"))

    def test_paths_and_arbitrary_sessions_rejected(self):
        for extra in ({"project": "C:\\"}, {"session_id": SESSION}, {"command": "cmd.exe"}):
            with self.assertRaises(BridgeError):
                parse_message(message(1, **extra), "owner", "test-runner")

    def test_strip_credentials(self):
        with patch.dict(os.environ, {"GH_TOKEN": "secret", "ACC_GITHUB_TOKEN": "secret2",
                                     "GIT_CONFIG_VALUE_0": "AUTHORIZATION", "PATH": "keep"}):
            env = codex_environment()
        self.assertNotIn("GH_TOKEN", env)
        self.assertNotIn("GIT_CONFIG_VALUE_0", env)
        self.assertEqual(env["PATH"], "keep")

    def test_result_redaction(self):
        result = redact("fake-secret-token and ghp_abcdefghijk", ("fake-secret-token",))
        self.assertNotIn("fake-secret-token", result)
        self.assertNotIn("ghp_", result)

    def test_targets_require_real_project_and_uuid(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "targets.json"
            path.write_text(json.dumps({"test": {"project": folder, "session_id": SESSION}}))
            self.assertEqual(load_targets(path)["test"]["session_id"], SESSION)
            path.write_text(json.dumps({"test": {"project": folder, "session_id": "--last"}}))
            with self.assertRaises(BridgeError):
                load_targets(path)


if __name__ == "__main__":
    unittest.main()
