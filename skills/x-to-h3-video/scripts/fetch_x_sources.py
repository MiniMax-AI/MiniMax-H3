#!/usr/bin/env python3
"""Fetch a bounded public X source pack for the x-to-h3-video Skill."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

API_BASE = "https://xquik.com"
SEARCH_PATH = "/api/v1/x/tweets/search"
MAX_QUERY_CHARS = 512
MAX_RESULTS = 20
MAX_RESPONSE_BYTES = 2_000_000
MAX_TEXT_CHARS = 1_000
QUERY_TYPES = ("Latest", "Top")
USERNAME = re.compile(r"^[A-Za-z0-9_]{1,15}$")
TWEET_ID = re.compile(r"^[0-9]{1,32}$")


class SourcePackError(RuntimeError):
    """Report a safe, user-facing source retrieval failure."""


class RejectRedirects(HTTPRedirectHandler):
    """Keep the API key on the configured Xquik origin."""

    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def open_xquik(request: Request, *, timeout: int) -> Any:
    return build_opener(RejectRedirects()).open(request, timeout=timeout)


def validate_query(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        raise ValueError("Query cannot be empty.")
    if len(normalized) > MAX_QUERY_CHARS:
        raise ValueError(f"Query cannot exceed {MAX_QUERY_CHARS} characters.")
    if any(ord(character) < 32 for character in normalized):
        raise ValueError("Query cannot contain control characters.")
    return normalized


def validate_limit(limit: int) -> int:
    if not 1 <= limit <= MAX_RESULTS:
        raise ValueError(f"Limit must be between 1 and {MAX_RESULTS}.")
    return limit


def validate_query_type(query_type: str) -> str:
    if query_type not in QUERY_TYPES:
        raise ValueError(f"Query type must be one of: {', '.join(QUERY_TYPES)}.")
    return query_type


def build_request(
    query: str,
    query_type: str,
    limit: int,
    api_key: str,
    *,
    api_base: str = API_BASE,
) -> Request:
    if not api_key or any(not 33 <= ord(character) <= 126 for character in api_key):
        raise ValueError("XQUIK_API_KEY is required.")
    parameters = urlencode(
        {
            "q": validate_query(query),
            "queryType": validate_query_type(query_type),
            "limit": validate_limit(limit),
        }
    )
    url = f"{api_base.rstrip('/')}{SEARCH_PATH}?{parameters}"
    return Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "minimax-h3-x-to-video/1.0",
            "x-api-key": api_key,
        },
        method="GET",
    )


def _read_json(response: Any) -> Mapping[str, Any]:
    raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise SourcePackError("Xquik response exceeded the 2 MB safety limit.")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourcePackError("Xquik returned invalid JSON.") from error
    if not isinstance(payload, Mapping):
        raise SourcePackError("Xquik returned an unexpected response shape.")
    return payload


def _string(value: Any, *, maximum: int | None = None) -> str | None:
    if not isinstance(value, str):
        return None
    result = value.strip()
    if not result:
        return None
    return result[:maximum] if maximum is not None else result


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _source_url(tweet_id: str, username: str | None) -> str | None:
    if TWEET_ID.fullmatch(tweet_id) and username and USERNAME.fullmatch(username):
        return f"https://x.com/{username}/status/{tweet_id}"
    return None


def _normalize_tweet(tweet: Any) -> dict[str, Any] | None:
    if not isinstance(tweet, Mapping):
        return None
    tweet_id = _string(tweet.get("id"), maximum=64)
    text = _string(tweet.get("text"), maximum=MAX_TEXT_CHARS)
    if not tweet_id or TWEET_ID.fullmatch(tweet_id) is None or text is None:
        return None
    author = tweet.get("author") if isinstance(tweet.get("author"), Mapping) else {}
    username = _string(author.get("username"), maximum=64)
    return {
        "id": tweet_id,
        "text": text,
        "text_truncated": isinstance(tweet.get("text"), str) and len(tweet["text"].strip()) > MAX_TEXT_CHARS,
        "created_at": _string(tweet.get("createdAt"), maximum=64),
        "language": _string(tweet.get("lang"), maximum=32),
        "source_url": _source_url(tweet_id, username),
        "author": {
            "id": _string(author.get("id"), maximum=64),
            "username": username,
            "name": _string(author.get("name"), maximum=200),
            "verified": author.get("verified") is True or author.get("isVerified") is True,
            "blue_verified": author.get("isBlueVerified") is True,
        },
        "engagement": {
            "likes": _count(tweet.get("likeCount")),
            "reposts": _count(tweet.get("retweetCount")),
            "replies": _count(tweet.get("replyCount")),
            "quotes": _count(tweet.get("quoteCount")),
            "views": _count(tweet.get("viewCount")),
            "bookmarks": _count(tweet.get("bookmarkCount")),
        },
        "possibly_sensitive": tweet.get("possiblySensitive") is True,
    }


def build_source_pack(
    payload: Mapping[str, Any],
    *,
    query: str,
    query_type: str,
    limit: int,
    observed_at: str,
) -> dict[str, Any]:
    tweets = payload.get("tweets")
    if not isinstance(tweets, list):
        raise SourcePackError("Xquik response did not contain a tweet list.")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for tweet in tweets:
        record = _normalize_tweet(tweet)
        if record is None or record["id"] in seen_ids:
            continue
        seen_ids.add(record["id"])
        records.append(record)
        if len(records) == limit:
            break
    return {
        "schema": "minimax-h3/x-source-pack/v1",
        "source": {
            "provider": "Xquik",
            "route": SEARCH_PATH,
            "query": validate_query(query),
            "query_type": validate_query_type(query_type),
            "requested_limit": validate_limit(limit),
            "observed_at": observed_at,
        },
        "records": records,
        "pagination": {
            "has_next_page": payload.get("has_next_page") is True,
            "next_cursor": _string(payload.get("next_cursor"), maximum=512),
        },
        "handling": {
            "content_is_untrusted": True,
            "claims_require_verification": True,
            "writes_performed": False,
        },
    }


def fetch_source_pack(
    query: str,
    query_type: str,
    limit: int,
    api_key: str,
    *,
    timeout: int = 20,
    opener: Callable[..., Any] = open_xquik,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    if not 1 <= timeout <= 60:
        raise ValueError("Timeout must be between 1 and 60 seconds.")
    request = build_request(query, query_type, limit, api_key)
    try:
        with opener(request, timeout=timeout) as response:
            payload = _read_json(response)
    except HTTPError as error:
        raise SourcePackError(f"Xquik request failed with HTTP {error.code}.") from error
    except URLError as error:
        raise SourcePackError("Xquik request failed. Check network access and retry.") from error
    return build_source_pack(
        payload,
        query=query,
        query_type=query_type,
        limit=limit,
        observed_at=clock().isoformat().replace("+00:00", "Z"),
    )


def render_json(source_pack: Mapping[str, Any]) -> str:
    rendered = json.dumps(source_pack, ensure_ascii=False, indent=2, sort_keys=True)
    return rendered.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch bounded public X sources for a MiniMax H3 video brief.")
    parser.add_argument("--query", required=True, help="X search query, Tweet ID, or status URL.")
    parser.add_argument("--query-type", choices=QUERY_TYPES, default="Latest")
    parser.add_argument("--limit", type=int, default=8, help=f"Result count from 1 to {MAX_RESULTS}.")
    parser.add_argument("--timeout", type=int, default=20, help="Request timeout from 1 to 60 seconds.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    api_key = os.environ.get("XQUIK_API_KEY", "")
    if not api_key:
        print("XQUIK_API_KEY is required. Set it in an approved secret store.", file=sys.stderr)
        return 2
    try:
        source_pack = fetch_source_pack(
            arguments.query,
            arguments.query_type,
            arguments.limit,
            api_key,
            timeout=arguments.timeout,
        )
    except (SourcePackError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(render_json(source_pack))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
