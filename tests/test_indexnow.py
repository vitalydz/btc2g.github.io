import io
import json
from pathlib import Path
import tempfile
import unittest
from scripts.submit_indexnow import changed_urls, sitemap_urls, submit, wait_for_publication


class Response(io.BytesIO):
    status = 202


class IndexNowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'indexnow-key.txt').write_text('abcdef1234567890')
        (self.root / 'sitemap.xml').write_text('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://btc2g.com/</loc></url><url><loc>https://btc2g.com/scenario.html</loc></url></urlset>')

    def test_only_changed_public_pages_are_selected(self):
        urls = sitemap_urls(self.root)
        self.assertEqual(changed_urls(urls, ['scenario.html', 'cloudflare/auth/public/admin/index.html']), [urls[1]])
        self.assertEqual(changed_urls(urls, ['docs/review.md']), [])
        self.assertEqual(changed_urls(urls, ['assets/btc_signal.json']), urls)

    def test_external_and_private_urls_never_reach_submission(self):
        for url in ['https://evil.test/', 'https://btc2g.com/admin/', 'https://btc2g.com/?email=private']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                submit([url], self.root, opener=lambda *a, **kw: self.fail('Network must not be called'))

    def test_unpublished_key_blocks_notification(self):
        calls = []
        def opener(request, **kwargs):
            calls.append(request)
            return Response(b'wrong key')
        with self.assertRaisesRegex(RuntimeError, 'does not match'):
            submit(sitemap_urls(self.root), self.root, opener)
        self.assertEqual(len(calls), 1)

    def test_success_checks_published_key_and_posts_canonical_urls(self):
        calls = []
        def opener(request, **kwargs):
            calls.append(request)
            return Response(b'abcdef1234567890' if len(calls) == 1 else b'')
        urls = sitemap_urls(self.root)
        self.assertEqual(submit(urls, self.root, opener), 202)
        self.assertEqual(json.loads(calls[1].data)['urlList'], urls)
        self.assertEqual(calls[1].full_url, 'https://api.indexnow.org/indexnow')

    def test_rejected_status_is_not_reported_as_success(self):
        def opener(request, **kwargs):
            response = Response(b'abcdef1234567890')
            response.status = 200 if request.get_method() == 'GET' else 429
            return response
        with self.assertRaisesRegex(RuntimeError, '429'):
            submit(sitemap_urls(self.root), self.root, opener)

    def test_notifications_wait_for_successful_pages_publication(self):
        for conclusion in ['success', 'failure']:
            def opener(request, **kwargs):
                return Response(json.dumps({'workflow_runs': [{'name': 'pages build and deployment', 'status': 'completed', 'conclusion': conclusion}]}).encode())
            if conclusion == 'success':
                wait_for_publication('a' * 40, opener, lambda seconds: None)
            else:
                with self.assertRaisesRegex(RuntimeError, 'no URLs submitted'):
                    wait_for_publication('a' * 40, opener, lambda seconds: None)


if __name__ == '__main__':
    unittest.main()
