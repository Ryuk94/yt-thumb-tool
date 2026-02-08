from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests import HTTPError

YOUTUBE_SEARCH_LIST = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_CHANNELS_LIST = "https://www.googleapis.com/youtube/v3/channels"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "established_creators.json"

REGION_CATEGORY_QUERIES: dict[str, dict[str, list[str]]] = {
    "US": {
        "entertainment": ["US entertainment creator", "US vlogger", "US comedy channel"],
        "gaming": ["US gaming creator", "US gaming channel", "US esports creator"],
        "music": ["US music artist official", "US music channel", "US singer official channel"],
        "sports": ["US sports highlights channel", "US sports creator", "US sports media channel"],
    },
    "GB": {
        "entertainment": ["UK entertainment creator", "UK comedy channel", "British vlog channel"],
        "gaming": ["UK gaming creator", "UK gaming channel", "British gamer channel"],
        "music": ["UK music artist official", "British music channel", "UK singer official channel"],
        "sports": ["UK sports highlights channel", "Premier League channel", "UK sports media channel"],
    },
    "CA": {
        "entertainment": ["Canada entertainment creator", "Canadian vlog channel", "Canadian comedy channel"],
        "gaming": ["Canada gaming creator", "Canadian gaming channel", "Canada esports creator"],
        "music": ["Canadian music artist official", "Canada music channel", "Canadian singer official channel"],
        "sports": ["Canada sports highlights channel", "Canadian sports media channel", "NHL channel"],
    },
    "AU": {
        "entertainment": ["Australia entertainment creator", "Australian vlog channel", "Australian comedy channel"],
        "gaming": ["Australia gaming creator", "Australian gaming channel", "Aussie gamer channel"],
        "music": ["Australian music artist official", "Australia music channel", "Australian singer channel"],
        "sports": ["Australia sports highlights channel", "AFL channel", "NRL channel"],
    },
}


def youtube_get(api_key: str, url: str, params: dict[str, Any], timeout: int = 15) -> dict[str, Any]:
    merged = params.copy()
    merged["key"] = api_key
    response = requests.get(url, params=merged, timeout=timeout)
    try:
        response.raise_for_status()
    except HTTPError as exc:
        reason = ""
        message = response.text
        try:
            payload = response.json()
            error = payload.get("error") or {}
            errors = error.get("errors") or []
            if errors and isinstance(errors[0], dict):
                reason = str(errors[0].get("reason") or "")
            message = str(error.get("message") or message)
        except ValueError:
            pass

        if response.status_code == 403:
            raise RuntimeError(
                f"YouTube API rejected the request (403). reason={reason or 'unknown'} message={message}"
            ) from exc
        raise
    return response.json()


def validate_api_key(api_key: str) -> None:
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY is required")
    # Most Google API keys begin with AIza and are 39 characters long.
    if not api_key.startswith("AIza") or len(api_key) < 35:
        raise RuntimeError("YOUTUBE_API_KEY format looks invalid (expected prefix 'AIza').")


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def collect_channel_candidates(
    api_key: str,
    region: str,
    category_queries: dict[str, list[str]],
    max_pages_per_query: int = 2,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for category, queries in category_queries.items():
        for query in queries:
            page_token = None
            pages = 0
            while pages < max_pages_per_query:
                payload = youtube_get(
                    api_key,
                    YOUTUBE_SEARCH_LIST,
                    {
                        "part": "snippet",
                        "type": "channel",
                        "q": query,
                        "regionCode": region,
                        "maxResults": 50,
                        "order": "relevance",
                        **({"pageToken": page_token} if page_token else {}),
                    },
                )
                for item in payload.get("items", []):
                    channel_id = ((item.get("snippet") or {}).get("channelId")) or ((item.get("id") or {}).get("channelId"))
                    if not isinstance(channel_id, str) or not channel_id:
                        continue
                    record = candidates.setdefault(
                        channel_id,
                        {"identifier": channel_id, "categories": set(), "formats": {"videos", "shorts"}},
                    )
                    record["categories"].update({"all", category})

                page_token = payload.get("nextPageToken")
                pages += 1
                if not page_token:
                    break
                time.sleep(0.05)
    return candidates


def enrich_channel_scores(api_key: str, candidates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ids = list(candidates.keys())
    enriched: list[dict[str, Any]] = []
    for group in chunked(ids, 50):
        payload = youtube_get(
            api_key,
            YOUTUBE_CHANNELS_LIST,
            {"part": "statistics,snippet,contentDetails", "id": ",".join(group), "maxResults": 50},
        )
        for channel in payload.get("items", []):
            cid = channel.get("id")
            if cid not in candidates:
                continue
            stats = channel.get("statistics") or {}
            snippet = channel.get("snippet") or {}
            try:
                subscribers = int(stats.get("subscriberCount", 0) or 0)
            except (TypeError, ValueError):
                subscribers = 0
            try:
                views = int(stats.get("viewCount", 0) or 0)
            except (TypeError, ValueError):
                views = 0
            try:
                videos = int(stats.get("videoCount", 0) or 0)
            except (TypeError, ValueError):
                videos = 0
            score = (subscribers * 5) + views + (videos * 100)
            entry = candidates[cid]
            enriched.append(
                {
                    "identifier": entry["identifier"],
                    "categories": sorted(entry["categories"]),
                    "formats": sorted(entry["formats"]),
                    "title": snippet.get("title"),
                    "score": score,
                    "subscribers": subscribers,
                }
            )
        time.sleep(0.05)
    enriched.sort(key=lambda item: (int(item["score"]), int(item["subscribers"])), reverse=True)
    return enriched


def build_region_pool(api_key: str, region: str, min_count: int) -> list[dict[str, Any]]:
    query_map = REGION_CATEGORY_QUERIES[region]
    candidates = collect_channel_candidates(api_key, region, query_map, max_pages_per_query=2)
    ranked = enrich_channel_scores(api_key, candidates)
    selected = ranked[: max(min_count, 70)]
    return [
        {
            "identifier": item["identifier"],
            "categories": item["categories"],
            "formats": item["formats"],
        }
        for item in selected
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh established creator pool by region.")
    parser.add_argument("--min-per-region", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    validate_api_key(api_key)

    regions = ["US", "GB", "CA", "AU"]
    region_payload: dict[str, list[dict[str, Any]]] = {}
    for region in regions:
        region_payload[region] = build_region_pool(api_key, region, max(50, args.min_per_region))

    output = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "regions": region_payload,
        "global": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote established creator pool: {args.output}")
    for region, entries in region_payload.items():
        print(f"{region}: {len(entries)} creators")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
