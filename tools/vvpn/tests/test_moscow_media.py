"""Regression tests for VVPN Moscow CSS, image and JavaScript discovery."""
import unittest

from moscow_media import MediaParser, safe_asset, verify_mime


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


if __name__ == "__main__":
    unittest.main()
