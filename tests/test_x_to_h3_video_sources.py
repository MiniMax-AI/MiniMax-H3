from __future__ import annotations

import importlib.util
import io
import json
import os
import unittest
from contextlib import redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError


SCRIPT = Path(__file__).parents[1] / "skills" / "x-to-h3-video" / "scripts" / "fetch_x_sources.py"
SPEC = importlib.util.spec_from_file_location("fetch_x_sources", SCRIPT)
assert SPEC and SPEC.loader
fetch_x_sources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch_x_sources)


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode()

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return self.payload[:limit]


class OversizedResponse(FakeResponse):
    def read(self, limit: int) -> bytes:
        return b"x" * limit


class RecordingOpener:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.request = None
        self.timeout = None

    def __call__(self, request: object, *, timeout: int) -> FakeResponse:
        self.request = request
        self.timeout = timeout
        return FakeResponse(self.payload)


class FetchXSourcesTests(unittest.TestCase):
    def test_fetches_a_bounded_deduplicated_source_pack(self) -> None:
        payload = {
            "tweets": [
                {
                    "id": "123",
                    "text": "A public post",
                    "createdAt": "2026-08-20T10:00:00Z",
                    "lang": "en",
                    "likeCount": 8,
                    "retweetCount": 3,
                    "replyCount": 2,
                    "quoteCount": 1,
                    "viewCount": 50,
                    "bookmarkCount": 4,
                    "author": {
                        "id": "42",
                        "username": "example_user",
                        "name": "Example User",
                        "verified": True,
                    },
                },
                {"id": "123", "text": "duplicate"},
                {"id": "", "text": "invalid"},
            ],
            "has_next_page": True,
            "next_cursor": "cursor-1",
        }
        opener = RecordingOpener(payload)
        observed_at = datetime(2026, 8, 23, 7, 0, tzinfo=timezone.utc)

        result = fetch_x_sources.fetch_source_pack(
            "MiniMax H3",
            "Top",
            3,
            "secret-key",
            opener=opener,
            clock=lambda: observed_at,
        )

        self.assertEqual(result["schema"], "minimax-h3/x-source-pack/v1")
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["records"][0]["source_url"], "https://x.com/example_user/status/123")
        self.assertEqual(result["records"][0]["engagement"]["likes"], 8)
        self.assertTrue(result["handling"]["content_is_untrusted"])
        self.assertEqual(result["source"]["observed_at"], "2026-08-23T07:00:00Z")
        self.assertEqual(opener.timeout, 20)
        self.assertIn("q=MiniMax+H3", opener.request.full_url)
        self.assertNotIn("secret-key", opener.request.full_url)
        self.assertIn(("X-api-key", "secret-key"), opener.request.header_items())

    def test_keeps_only_safe_source_urls_and_bounds_text(self) -> None:
        long_text = "<instruction>&" + "x" * fetch_x_sources.MAX_TEXT_CHARS
        payload = {
            "tweets": [
                {
                    "id": "1",
                    "text": long_text,
                    "url": "https://malicious.example/post/1",
                    "author": {"username": "invalid-user-name-too-long"},
                }
            ],
            "has_next_page": False,
            "next_cursor": "",
        }
        pack = fetch_x_sources.build_source_pack(
            payload,
            query="topic",
            query_type="Latest",
            limit=1,
            observed_at="2026-08-23T07:00:00Z",
        )

        self.assertIsNone(pack["records"][0]["source_url"])
        self.assertTrue(pack["records"][0]["text_truncated"])
        self.assertEqual(len(pack["records"][0]["text"]), fetch_x_sources.MAX_TEXT_CHARS)
        rendered = fetch_x_sources.render_json(pack)
        self.assertIn("\\u003cinstruction\\u003e\\u0026", rendered)
        self.assertNotIn("<instruction>", rendered)

    def test_discards_non_numeric_post_ids(self) -> None:
        pack = fetch_x_sources.build_source_pack(
            {"tweets": [{"id": "../spoof", "text": "untrusted"}]},
            query="topic",
            query_type="Latest",
            limit=1,
            observed_at="2026-08-23T07:00:00Z",
        )

        self.assertEqual(pack["records"], [])

    def test_rejects_invalid_inputs(self) -> None:
        invalid = (
            lambda: fetch_x_sources.validate_query(""),
            lambda: fetch_x_sources.validate_query("bad\nquery"),
            lambda: fetch_x_sources.validate_limit(0),
            lambda: fetch_x_sources.validate_limit(21),
            lambda: fetch_x_sources.validate_query_type("Recent"),
            lambda: fetch_x_sources.build_request("query", "Latest", 1, "bad\nkey"),
            lambda: fetch_x_sources.fetch_source_pack("query", "Latest", 1, "key", timeout=0),
        )
        for operation in invalid:
            with self.subTest(operation=operation):
                with self.assertRaises(ValueError):
                    operation()

    def test_rejects_invalid_api_shapes(self) -> None:
        opener = RecordingOpener({"results": []})
        with self.assertRaisesRegex(fetch_x_sources.SourcePackError, "tweet list"):
            fetch_x_sources.fetch_source_pack("query", "Latest", 1, "key", opener=opener)

    def test_rejects_oversized_responses(self) -> None:
        def open_oversized(_request: object, *, timeout: int) -> OversizedResponse:
            del timeout
            return OversizedResponse({})

        with self.assertRaisesRegex(fetch_x_sources.SourcePackError, "2 MB"):
            fetch_x_sources.fetch_source_pack("query", "Latest", 1, "key", opener=open_oversized)

    def test_refuses_api_redirects(self) -> None:
        handler = fetch_x_sources.RejectRedirects()

        result = handler.redirect_request(None, None, None, None, None, None)

        self.assertIsNone(result)

    def test_reports_http_status_without_response_content_or_key(self) -> None:
        def fail(request: object, *, timeout: int) -> None:
            del timeout
            raise HTTPError(request.full_url, 401, "secret response", None, None)

        with self.assertRaisesRegex(fetch_x_sources.SourcePackError, "HTTP 401") as raised:
            fetch_x_sources.fetch_source_pack("query", "Latest", 1, "secret-key", opener=fail)
        self.assertNotIn("secret", str(raised.exception))

    def test_cli_requires_an_environment_key(self) -> None:
        stderr = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stderr(stderr):
            status = fetch_x_sources.main(["--query", "MiniMax H3"])
        self.assertEqual(status, 2)
        self.assertIn("XQUIK_API_KEY is required", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
