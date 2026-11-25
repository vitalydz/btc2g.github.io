import unittest
from unittest.mock import patch
from acc_github_bridge import PrivateGitHubClient, GitHubClient, BridgeError
from acc_site_stats import SiteStats, SITES, COUNT_SQL
import test_acc_inbox as inbox_tests
message = inbox_tests.message


class StatsTests(unittest.TestCase):
    def test_visibility_change_blocks_private_comment(self):
        client = PrivateGitHubClient("owner/private", "fixture")
        with patch.object(client, "request", return_value={"private": False}), patch.object(GitHubClient, "comment") as send:
            with self.assertRaises(BridgeError):
                client.comment(1, "private counts")
            send.assert_not_called()

    def test_private_comment_rechecks_visibility(self):
        client = PrivateGitHubClient("owner/private", "fixture")
        with patch.object(client, "request", return_value={"private": True}) as check, patch.object(GitHubClient, "comment") as send:
            client.comment(1, "counts")
            client.comment(1, "next")
            self.assertEqual(check.call_count, 2)
            self.assertEqual(send.call_count, 2)

    def test_changed_deployment_never_queries_wrong_database(self):
        client = SiteStats("fixture")
        with patch.object(client, "request", return_value={"bindings": []}) as request:
            rows = client.collect()
        self.assertEqual(request.call_count, 6)
        self.assertTrue(all(r["status"] == "UNAVAILABLE" for r in rows))
        self.assertTrue(all("counts" not in r for r in rows))

    def test_verified_bindings_use_only_fixed_aggregate_query(self):
        client = SiteStats("fixture")
        calls = []
        def request(path, data=None):
            calls.append((path, data))
            if data is None:
                site, (_, _, db) = next((k, v) for k, v in SITES.items() if v[0] in path)
                return {"bindings": [{"type": "d1", "name": "DB", "id": db},
                                     {"name": "PUBLIC_SITE_ORIGIN", "text": "https://"+site}]}
            self.assertEqual(data, {"sql": COUNT_SQL})
            return [{"success": True, "results": [{"registered_accounts": 3, "verified_email": 1, "administrators": 1, "disabled_accounts": 0}]}]
        with patch.object(client, "request", side_effect=request):
            rows = client.collect()
        self.assertTrue(all(r["status"] == "OK" for r in rows))
        self.assertEqual(len(calls), 12)
        self.assertIn("findbbest.com", SITES)
        self.assertNotIn("findbest.com", SITES)


class StatsInboxTests(unittest.TestCase):
    setUp = inbox_tests.InboxTests.setUp
    finish = inbox_tests.InboxTests.finish
    def test_public_stats_rejected_without_model_or_query(self):
        self.client.comments = [message(1, "stats")]
        with patch("acc_site_stats.statistics_report") as read:
            self.inbox.tick()
        self.assertEqual(self.inbox.messages['1']['state'], 'REJECTED')
        self.assertEqual(self.executor.calls, [])
        read.assert_not_called()

    def test_private_stats_uses_collector_without_model(self):
        self.client.requires_private = True
        self.client.verify_private = lambda: None
        self.client.comments = [message(1, "stats")]
        with patch("acc_site_stats.statistics_report", return_value={"state": "COMPLETED", "result": "fixture aggregate"}) as read:
            self.inbox.tick()
            self.inbox.thread.join(2)
            self.inbox.tick()
        self.assertEqual(self.inbox.messages['1']['state'], 'COMPLETED')
        read.assert_called_once()
        self.assertEqual(self.executor.calls, [])


if __name__ == '__main__':
    unittest.main()
