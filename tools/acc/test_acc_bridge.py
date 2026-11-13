import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import acc_github_bridge as bridge

SHA = "a" * 40


def issue(action="push", **extra):
    job = {"runner": "win11", "action": action}
    if action != "handshake":
        job.update(branch="main", sha=SHA)
    job.update(extra)
    return {"number": 12, "state": "open", "title": "ACC: Publish version", "user": {"login": "owner"},
            "body": bridge.MARKER + "\n```json\n" + json.dumps(job) + "\n```"}


class FakeClient:
    repo = "owner/repo"
    _token = "TEST_TOKEN_NOT_REAL"

    def __init__(self):
        self.issue = issue()
        self.comments = []
        self.calls = []
        self.responses = {}

    def paged(self, path):
        if path.startswith("/issues?"):
            return [copy.deepcopy(self.issue)]
        return copy.deepcopy(self.comments)

    def api(self, method, path, data=None):
        self.calls.append((method, path, data))
        if path == "/issues/12":
            return copy.deepcopy(self.issue)
        if path.startswith("/git/ref/heads/"):
            return {"object": {"sha": SHA}}
        return copy.deepcopy(self.responses.get((method, path), {}))

    def comment(self, number, text):
        self.comments.append({"user": {"login": "owner"}, "body": text})


class ProtocolTests(unittest.TestCase):
    def test_author_prefix_marker_and_pr_filter(self):
        for change in ({"user": {"login": "attacker"}}, {"state": "closed"},
                       {"title": "untrusted"}, {"body": "no marker"}, {"pull_request": {}}):
            candidate = issue()
            candidate.update(change)
            self.assertIsNone(bridge.parse_job(candidate, "owner", "win11"))

    def test_routing_and_legacy_handshake(self):
        self.assertIsNone(bridge.parse_job(issue(), "owner", "cloud"))
        candidate = issue("handshake")
        candidate.update(title="ACC: Connection handshake test", body=bridge.MARKER)
        self.assertEqual(bridge.parse_job(candidate, "owner", "cloud")["action"], "handshake")

    def test_reject_arbitrary_commands_and_injections(self):
        for candidate in (issue("shell"), issue(command="rm -rf /"), issue(branch="--force"),
                          issue(branch="main:other"), issue(branch="a/../main"),
                          issue(sha="HEAD"), issue("workflow", workflow="../deploy.yml"),
                          issue("workflow", workflow="deploy.yml", inputs={"x": []})):
            with self.assertRaises(bridge.BridgeError):
                bridge.parse_job(candidate, "owner", "win11")

    def test_multiple_json_blocks_rejected(self):
        candidate = issue()
        candidate["body"] += '\n```json\n{}\n```'
        with self.assertRaises(bridge.BridgeError):
            bridge.parse_job(candidate, "owner", "win11")


class BridgeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "state.json"
        self.store = bridge.StateStore(self.path)
        self.addCleanup(self.store.close)
        self.client = FakeClient()
        self.stop = threading.Event()
        self.publisher = Mock()
        self.publisher.execute.return_value = ("COMPLETED", "verified")
        self.bridge = bridge.Bridge(self.client, "owner", "win11", self.store,
                                   self.publisher, self.stop, lambda _: None)

    def approve(self):
        self.bridge.approve(12, self.store.data["12"]["digest"])

    def test_receive_only_then_approve_once(self):
        self.bridge.poll()
        self.bridge.poll()
        self.publisher.execute.assert_not_called()
        self.assertEqual(len(self.client.comments), 1)
        self.approve()
        self.assertEqual(self.store.data["12"]["status"], "COMPLETED")
        self.assertEqual(self.publisher.execute.call_count, 1)
        with self.assertRaises(bridge.BridgeError):
            self.approve()
        self.bridge.poll()
        self.assertEqual(self.publisher.execute.call_count, 1)

    def test_edited_issue_invalidates_approval(self):
        self.bridge.poll()
        old_digest = self.store.data["12"]["digest"]
        self.client.issue["body"] += "\nchanged"
        with self.assertRaises(bridge.BridgeError):
            self.bridge.approve(12, old_digest)
        self.bridge.poll()
        self.assertNotEqual(self.store.data["12"]["digest"], old_digest)
        self.publisher.execute.assert_not_called()

    def test_closed_issue_rejected_at_approval(self):
        self.bridge.poll()
        self.client.issue["state"] = "closed"
        with self.assertRaises(bridge.BridgeError):
            self.approve()
        self.publisher.execute.assert_not_called()

    def test_remote_execution_receipt_blocks_replay(self):
        self.client.comments = [{"user": {"login": "owner"}, "body": "ACC v1.3: RUNNING\nold session"}]
        self.bridge.poll()
        self.assertEqual(self.store.data["12"]["status"], "UNKNOWN")
        self.publisher.execute.assert_not_called()

    def test_failed_receipt_blocks_execution_and_replay(self):
        self.bridge.poll()
        self.client.comment = Mock(side_effect=bridge.BridgeError("network failure"))
        with self.assertRaises(bridge.BridgeError):
            self.approve()
        self.publisher.execute.assert_not_called()
        self.assertEqual(json.loads(self.path.read_text())["12"]["status"], "RUNNING")

    def test_failure_after_mutation_is_unknown_not_retried(self):
        self.bridge.poll()
        self.publisher.execute.side_effect = TimeoutError("secret response")
        self.approve()
        self.assertEqual(self.store.data["12"]["status"], "UNKNOWN")
        self.assertNotIn("secret", self.store.data["12"]["detail"])
        self.bridge.poll()
        self.assertEqual(self.publisher.execute.call_count, 1)

    def test_stop_and_handshake_do_not_execute(self):
        self.bridge.poll()
        self.stop.set()
        self.approve()
        self.publisher.execute.assert_not_called()

    def test_journal_lock_and_corruption(self):
        with self.assertRaises(bridge.BridgeError):
            bridge.StateStore(self.path)
        bad = self.path.with_name("bad.json")
        bad.write_text("broken", encoding="utf-8")
        with self.assertRaises(bridge.BridgeError):
            bridge.StateStore(bad)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.stop = threading.Event()
        self.publisher = bridge.Publisher(self.client, ".", self.stop, ["deploy.yml"])

    def job(self, action, **extra):
        return bridge.parse_job(issue(action, **extra), "owner", "win11")

    def test_push_exact_sha_without_force(self):
        self.publisher.git = Mock(side_effect=["https://github.com/owner/repo.git", "", "main", SHA, ""])
        status, _ = self.publisher.execute(self.job("push"))
        self.assertEqual(status, "COMPLETED")
        self.publisher.git.assert_called_with("push", "https://github.com/owner/repo.git", SHA + ":refs/heads/main")

    def test_dirty_or_wrong_repository_blocks_push(self):
        for results in (["https://github.com/other/repo"], ["https://github.com/owner/repo", " M file"]):
            self.publisher.git = Mock(side_effect=results)
            with self.assertRaises(bridge.BridgeError):
                self.publisher.execute(self.job("push"))
            self.assertFalse(any(c.args[0] == "push" for c in self.publisher.git.call_args_list))

    def test_changed_head_blocks_push(self):
        self.publisher.git = Mock(side_effect=["https://github.com/owner/repo", "", "main", "b" * 40])
        with self.assertRaises(bridge.BridgeError):
            self.publisher.execute(self.job("push"))
        self.assertEqual(self.publisher.git.call_count, 4)

    def test_workflow_allowlist_and_dispatch_is_not_deployment_success(self):
        status, _ = self.publisher.execute(self.job("workflow", workflow="deploy.yml"))
        self.assertEqual(status, "DISPATCHED")
        with self.assertRaises(bridge.BridgeError):
            self.publisher.execute(self.job("workflow", workflow="untrusted.yml"))

    def test_changed_remote_ref_blocks_deploy(self):
        self.publisher.remote_sha = Mock(return_value="b" * 40)
        with self.assertRaises(bridge.BridgeError):
            self.publisher.execute(self.job("pages"))
        self.assertFalse(self.client.calls)

    def test_pages_requires_matching_source_and_commit(self):
        self.client.responses[("GET", "/pages")] = {"build_type": "legacy", "source": {"branch": "main"}}
        self.client.responses[("POST", "/pages/builds")] = {"url": "https://api.github.com/repos/owner/repo/pages/builds/12"}
        self.client.responses[("GET", "/pages/builds/12")] = {"status": "built", "commit": SHA}
        self.stop.wait = Mock(return_value=False)
        self.assertEqual(self.publisher.execute(self.job("pages"))[0], "COMPLETED")
        self.client.responses[("GET", "/pages/builds/12")]["commit"] = "b" * 40
        with self.assertRaises(bridge.BridgeError):
            self.publisher.execute(self.job("pages"))

    def test_pages_latest_build_is_monitored(self):
        self.client.responses[("GET", "/pages")] = {"build_type": "legacy", "source": {"branch": "main"}}
        self.client.responses[("POST", "/pages/builds")] = {"url": "https://api.github.com/repos/owner/repo/pages/builds/latest"}
        self.client.responses[("GET", "/pages/builds/latest")] = {
            "status": "built", "commit": SHA,
            "created_at": (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
            "url": "https://api.github.com/repos/owner/repo/pages/builds/15"}
        self.stop.wait = Mock(return_value=False)
        self.assertEqual(self.publisher.execute(self.job("pages"))[0], "COMPLETED")

    def test_workflow_pages_configuration_not_changed(self):
        self.client.responses[("GET", "/pages")] = {"build_type": "workflow"}
        with self.assertRaises(bridge.BridgeError):
            self.publisher.execute(self.job("pages"))
        self.assertFalse(any(method == "POST" for method, _, _ in self.client.calls))

    def test_token_only_in_child_environment_not_command(self):
        process = Mock(returncode=0)
        process.communicate.return_value = ("", "")
        process.poll.return_value = 0
        with patch.object(bridge.subprocess, "Popen", return_value=process) as popen:
            self.publisher.git("push", "https://github.com/owner/repo.git", SHA + ":refs/heads/main")
        args, kwargs = popen.call_args
        self.assertNotIn(self.client._token, repr(args))
        self.assertIn("GIT_CONFIG_VALUE_0", kwargs["env"])
        self.assertNotIn("shell", kwargs)

    def test_stop_prevents_all_publication(self):
        self.stop.set()
        with self.assertRaises(bridge.BridgeError):
            self.publisher.execute(self.job("pages"))
        self.assertFalse(self.client.calls)


if __name__ == "__main__":
    unittest.main()
