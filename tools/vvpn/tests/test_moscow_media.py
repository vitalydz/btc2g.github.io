"""Regression tests for VVPN Moscow CSS, image and JavaScript discovery."""
import unittest

from types import SimpleNamespace
from moscow_media import (MediaParser, safe_asset, verify_mime, redacted_url,
                          http_error_record, classify_browser_report, browser_exit_status)
from vvpn import SITES


class MediaParserTests(unittest.TestCase):
    def test_discovers_all_three_resource_types_and_deduplicates(self):
        base = "https://example.org/path/"
        p = MediaParser(base)
        p.feed("""<!DOCTYPE html>
        <link rel="stylesheet" href="/app.css">
        <link rel="preload" as="style" href="/extra.css">
        <link rel="modulepreload" as="script" href="/modern.js">
        <script src="/app.js"></script><script src="/app.js"></script>
        <img src="a.jpg" srcset="b.webp 1x, c.webp 2x">
        <source srcset="/d.webp 320w">
        <style>
        body {background: url("/bg.png")}
        @font-face {src: url("/font.woff2")}
        </style>
        <img data-src="/lazy.png">
        """)
        p.finish()
        self.assertEqual(p.stylesheets, ["https://example.org/app.css", "https://example.org/extra.css"])
        self.assertEqual(p.scripts, ["https://example.org/modern.js", "https://example.org/app.js"])
        self.assertEqual(p.images, [
            "https://example.org/path/a.jpg",
            "https://example.org/path/b.webp",
            "https://example.org/path/c.webp",
            "https://example.org/d.webp",
            "https://example.org/lazy.png",
            "https://example.org/bg.png",
        ])

    def test_rejects_sensitive_and_non_http_assets(self):
        self.assertIsNone(safe_asset("https://example.org/", "data:image/png;base64,abc"))
        self.assertIsNone(safe_asset("https://example.org/", "javascript:alert(1)"))
        self.assertIsNone(safe_asset("https://example.org/", "/photo?a=1&token=secret"))
        self.assertIsNone(safe_asset("https://example.org/", "https://bad:password@example.org/a"))
        self.assertEqual(safe_asset("https://example.org/", "/good.png?v=1"),
                         "https://example.org/good.png?v=1")

    def test_mime_validation(self):
        cases = [
            ("script", "application/javascript", "ok"),
            ("script", "text/html", "invalid_script_mime"),
            ("stylesheet", "text/css; charset=utf-8", "ok"),
            ("stylesheet", "text/html", "invalid_css_mime"),
            ("image", "image/webp", "ok"),
            ("image", "text/html", "invalid_image_mime"),
            ("script", "", "missing_mime"),
        ]
        for kind, mime, expected in cases:
            with self.subTest(kind=kind, mime=mime):
                row = {"state": "ok", "code": 200, "content_type": mime}
                self.assertEqual(verify_mime(row, kind)["state"], expected)
        self.assertEqual(
            verify_mime({"state": "network_error", "code": None}, "image")["state"],
            "network_error",
        )


class MoscowBrowserNetworkTests(unittest.TestCase):
    def test_redacted_url_removes_query_fragment_and_userinfo(self):
        self.assertEqual(
            redacted_url("https://user:password@bmychance.com/api/me?token=secret#private"),
            "https://bmychance.com/api/me",
        )

    def test_non_http_url_omitted(self):
        self.assertEqual(redacted_url("data:text/plain,secret"), "<non-http-url>")

    def test_xhr_401_is_attributed(self):
        response = SimpleNamespace(
            status=401,
            url="https://bmychance.com/api/current?auth=secret",
            request=SimpleNamespace(resource_type="fetch"),
        )
        self.assertEqual(
            http_error_record(response),
            {"type": "fetch", "url": "https://bmychance.com/api/current", "http_status": 401},
        )

    def test_http_200_not_an_error(self):
        response = SimpleNamespace(
            status=200,
            url="https://btc2g.com/",
            request=SimpleNamespace(resource_type="document"),
        )
        self.assertIsNone(http_error_record(response))

    def test_css_404_reports_resource_type(self):
        response = SimpleNamespace(
            status=404,
            url="https://btc2g.com/assets/site.css?v=1",
            request=SimpleNamespace(resource_type="stylesheet"),
        )
        self.assertEqual(
            http_error_record(response),
            {"type": "stylesheet", "url": "https://btc2g.com/assets/site.css", "http_status": 404},
        )


class BrowserResultClassificationTests(unittest.TestCase):
    def test_anonymous_background_401_is_reported_without_failing_browser(self):
        page = {"site": SITES[0], "state": "loaded", "http": 200,
                "http_errors": [{"type": "fetch", "http_status": 401}]}
        self.assertEqual(classify_browser_report(page), "loaded_with_warnings")
        rows = [{"site": site, "state": "loaded", "http": 200} for site in SITES]
        rows[0]["state"] = classify_browser_report(page)
        self.assertEqual(browser_exit_status(rows), 0)

    def test_critical_asset_failure_and_js_exception_fail(self):
        for error in ({"critical_failed_requests": [{"type": "stylesheet"}]},
                      {"failed_resource_statuses": [{"type": "script", "status": 404}]},
                      {"javascript_exceptions": ["ReferenceError"]},
                      {"broken_image_count": 1}):
            with self.subTest(error=error):
                row = {"state": "loaded", "http": 200, **error}
                self.assertEqual(classify_browser_report(row), "loaded_with_errors")
        rows = [{"site": site, "state": "loaded", "http": 200} for site in SITES]
        rows[1]["state"] = "loaded_with_errors"
        self.assertEqual(browser_exit_status(rows), 1)

    def test_missing_or_duplicated_site_cannot_pass(self):
        rows = [{"site": site, "state": "loaded", "http": 200} for site in SITES]
        self.assertEqual(browser_exit_status(rows[:-1]), 1)
        rows[-1]["site"] = SITES[0]
        self.assertEqual(browser_exit_status(rows), 1)

    def test_http_or_browser_error_cannot_pass(self):
        for state in ("http_error", "browser_error"):
            self.assertEqual(classify_browser_report({"state": state, "http": 500}), state)
        self.assertEqual(classify_browser_report({"state": "loaded", "http": 500}), "http_error")

if __name__ == "__main__":
    unittest.main()
