import math
import os
import re
import time
import json
import uuid
import subprocess
import sys
import threading
import requests
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import StatisticsError, median
from typing import Any
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field
try:
    from backend.app.services.thumb_quality import analyze_thumbnail
except ModuleNotFoundError:
    from app.services.thumb_quality import analyze_thumbnail


# ---------------------------
# Helpers
# ---------------------------

def iso8601_duration_to_seconds(duration: str) -> int:
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def is_verticalish(thumbnails: dict) -> bool:
    """
    Heuristic only. Most YouTube thumbnails are 16:9 even for Shorts.
    This will filter aggressively; keep as optional toggle.
    """
    for key in ("maxres", "standard", "high", "medium", "default"):
        t = thumbnails.get(key)
        if t and "width" in t and "height" in t:
            return t["height"] >= t["width"]
    return False


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Region -> language hint (bias, not guarantee)
REGION_LANG = {
    "GB": "en",
    "US": "en",
    "CA": "en",
    "AU": "en",
}
TRENDING_REGIONS = ["AU", "CA", "GB", "US"]


def lang_for_region(region: str) -> str:
    return REGION_LANG.get(region.upper(), "en")


def matches_lang(snip: dict, lang: str) -> bool:
    """
    YouTube sometimes provides defaultAudioLanguage/defaultLanguage.
    We'll treat any value starting with 'en' as a match for 'en'.
    """
    dal = (snip.get("defaultAudioLanguage") or "").lower()
    dl = (snip.get("defaultLanguage") or "").lower()
    lang = lang.lower()
    return dal.startswith(lang) or dl.startswith(lang)


# ---------------------------
# Very simple in-memory cache
# ---------------------------

CACHE: dict[str, tuple[float, Any]] = {}
CACHE_TTL_SECONDS = 60 * 30  # 30 minutes
SHORTS_DETECT_CACHE: dict[str, tuple[float, bool]] = {}
SHORTS_DETECT_TTL_SECONDS = 60 * 60 * 6  # 6 hours
PATTERN_LIBRARY: dict[str, dict[str, Any]] = {}
PATTERN_LIBRARY_LOCK = threading.Lock()
PATTERN_LIBRARY_FILE = Path(
    os.getenv("PATTERN_LIBRARY_FILE")
    or (Path(__file__).resolve().parent / "data_runtime" / "pattern_library.json")
)

def cache_get(key: str):
    hit = CACHE.get(key)
    if not hit:
        return None
    expires_at, value = hit
    if time.time() > expires_at:
        CACHE.pop(key, None)
        return None
    return value

def cache_set(key: str, value: Any):
    CACHE[key] = (time.time() + CACHE_TTL_SECONDS, value)


def cache_set_custom(key: str, value: Any, ttl: int):
    CACHE[key] = (time.time() + ttl, value)


def load_pattern_library() -> None:
    with PATTERN_LIBRARY_LOCK:
        PATTERN_LIBRARY.clear()
        try:
            if not PATTERN_LIBRARY_FILE.exists():
                return
            raw = json.loads(PATTERN_LIBRARY_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if isinstance(key, str) and isinstance(value, dict):
                        PATTERN_LIBRARY[key] = value
        except (OSError, json.JSONDecodeError):
            PATTERN_LIBRARY.clear()


def persist_pattern_library(snapshot: dict[str, dict[str, Any]] | None = None) -> None:
    if snapshot is None:
        with PATTERN_LIBRARY_LOCK:
            snapshot = dict(PATTERN_LIBRARY)
    PATTERN_LIBRARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    PATTERN_LIBRARY_FILE.write_text(
        json.dumps(snapshot, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


class ResolveRequest(BaseModel):
    query: str


class PatternExtractItem(BaseModel):
    thumbnail_url: str
    video_id: str | None = None
    title: str | None = None
    channel_title: str | None = None


class PatternExtractRequest(BaseModel):
    items: list[PatternExtractItem] = Field(default_factory=list)


class PatternSaveRequest(BaseModel):
    name: str
    clusters: list[dict[str, Any]] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class YouTubeQuotaExceededError(Exception):
    pass


def get_client_ip(request: Request) -> str:
    forwarded = (request.headers.get("x-forwarded-for") or "").strip()
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def enforce_api_rate_limit(request: Request, scope: str = "youtube") -> None:
    now_ts = time.time()
    key = f"{scope}:{get_client_ip(request)}"
    bucket = API_RATE_LIMIT_BUCKETS.get(key)
    if bucket is None:
        bucket = deque()
        API_RATE_LIMIT_BUCKETS[key] = bucket

    cutoff = now_ts - API_RATE_LIMIT_WINDOW_SECONDS
    while bucket and bucket[0] < cutoff:
        bucket.popleft()

    if len(bucket) >= API_RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a minute and try again.",
        )

    bucket.append(now_ts)


def enforce_profile_rate_limit(request: Request) -> None:
    now_ts = time.time()
    client_ip = get_client_ip(request)
    bucket = PROFILE_RATE_LIMIT_BUCKETS.get(client_ip)
    if bucket is None:
        bucket = deque()
        PROFILE_RATE_LIMIT_BUCKETS[client_ip] = bucket

    cutoff = now_ts - PROFILE_RATE_LIMIT_WINDOW_SECONDS
    while bucket and bucket[0] < cutoff:
        bucket.popleft()

    if len(bucket) >= PROFILE_RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Too many profile requests. Please wait a minute and try again.",
        )

    bucket.append(now_ts)
    enforce_api_rate_limit(request, scope="profile")


def youtube_api_get(url: str, params: dict[str, Any], timeout: int = 15) -> dict[str, Any]:
    try:
        response = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException:
        raise HTTPException(
            status_code=502,
            detail="YouTube is temporarily unavailable. Please try again.",
        )

    if response.status_code == 200:
        return response.json()

    lowered = response.text.lower()
    if response.status_code in {403, 429} and (
        "quotaexceeded" in lowered or "quota exceeded" in lowered or "youtube.quota" in lowered
    ):
        raise YouTubeQuotaExceededError("YouTube API quota exceeded")

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Channel not found.")

    raise HTTPException(status_code=502, detail="Could not fetch YouTube data right now.")


def _parse_duration_text_to_seconds(value: str | None) -> int | None:
    if not value:
        return None
    parts = [p for p in value.strip().split(":") if p.isdigit()]
    if not parts:
        return None
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        return None
    return None


def _deep_find_video_renderers(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        vr = node.get("videoRenderer")
        if isinstance(vr, dict):
            found.append(vr)
        for value in node.values():
            found.extend(_deep_find_video_renderers(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_deep_find_video_renderers(value))
    return found


def _extract_text_from_renderer(value: dict[str, Any] | None) -> str | None:
    if not isinstance(value, dict):
        return None
    simple = value.get("simpleText")
    if isinstance(simple, str) and simple.strip():
        return simple.strip()
    runs = value.get("runs")
    if isinstance(runs, list):
        parts = []
        for run in runs:
            if isinstance(run, dict):
                text = run.get("text")
                if isinstance(text, str):
                    parts.append(text)
        joined = "".join(parts).strip()
        return joined or None
    return None


def _extract_channel_id_from_html(html: str) -> str | None:
    match = CHANNEL_ID_RE.search(html or "")
    if match:
        return match.group(1)
    return None


def _extract_yt_initial_data(html: str) -> dict[str, Any] | None:
    match = YT_INITIAL_DATA_RE.search(html or "")
    if not match:
        return None
    raw = match.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _filter_profile_items_by_type(items: list[dict[str, Any]], content_type: str) -> list[dict[str, Any]]:
    if content_type == "all":
        return items
    filtered = []
    for item in items:
        duration = item.get("duration")
        if not isinstance(duration, int):
            # In fallback mode, missing duration should not be hard filtered.
            filtered.append(item)
            continue
        if content_type == "shorts" and duration <= 60:
            filtered.append(item)
        if content_type == "videos" and duration > 60:
            filtered.append(item)
    return filtered


def fetch_uploads_playlist_safe(channel_id: str) -> str | None:
    payload = youtube_api_get(
        YOUTUBE_CHANNELS_LIST,
        {
            "part": "contentDetails",
            "id": channel_id,
            "key": YOUTUBE_API_KEY,
        },
    )
    items = payload.get("items", [])
    if not items:
        return None
    return (((items[0].get("contentDetails") or {}).get("relatedPlaylists") or {}).get("uploads"))


def fetch_playlist_video_ids_safe(playlist_id: str, max_items: int) -> list[str]:
    ids: list[str] = []
    page_token = None
    while len(ids) < max_items:
        params: dict[str, Any] = {
            "part": "contentDetails",
            "playlistId": playlist_id,
            "maxResults": min(50, max_items - len(ids)),
            "key": YOUTUBE_API_KEY,
        }
        if page_token:
            params["pageToken"] = page_token
        payload = youtube_api_get(YOUTUBE_PLAYLIST_ITEMS_LIST, params)

        for item in payload.get("items", []):
            video_id = (item.get("contentDetails") or {}).get("videoId")
            if video_id:
                ids.append(video_id)
            if len(ids) >= max_items:
                break

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    return ids[:max_items]


def hydrate_video_metadata_safe(video_ids: list[str]) -> list[dict[str, Any]]:
    hydrated: list[dict[str, Any]] = []
    for batch in chunked(video_ids, 50):
        payload = youtube_api_get(
            YOUTUBE_VIDEOS_LIST,
            {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(batch),
                "key": YOUTUBE_API_KEY,
            },
        )
        hydrated.extend(payload.get("items", []))
    return hydrated


def build_profile_items_from_videos(videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for video in videos:
        snippet = video.get("snippet", {})
        statistics = video.get("statistics", {})
        details = video.get("contentDetails", {})
        thumb_obj = best_thumbnail_object(snippet.get("thumbnails", {}) or {}) or {}
        duration_seconds = iso8601_duration_to_seconds(details.get("duration", ""))

        items.append(
            {
                "id": video.get("id"),
                "title": snippet.get("title"),
                "channelTitle": snippet.get("channelTitle"),
                "publishedAt": snippet.get("publishedAt"),
                "views": int(statistics.get("viewCount", 0) or 0),
                "duration": duration_seconds,
                "thumbnail": thumb_obj.get("url"),
                "thumbWidth": thumb_obj.get("width"),
                "thumbHeight": thumb_obj.get("height"),
            }
        )
    return items


def sort_profile_items(items: list[dict[str, Any]], sort_mode: str) -> list[dict[str, Any]]:
    if sort_mode == "popular":
        return sorted(items, key=lambda item: int(item.get("views", 0) or 0), reverse=True)
    return sorted(items, key=lambda item: (item.get("publishedAt") or ""), reverse=True)


def fetch_profile_grid_tier_a(channel_id: str, limit: int, sort_mode: str) -> list[dict[str, Any]]:
    uploads_playlist = fetch_uploads_playlist_safe(channel_id)
    if not uploads_playlist:
        return []

    # Pull a larger deterministic candidate pool so content-type filtering has enough rows.
    candidate_limit = max(50, min(200, limit * 4))
    ids = fetch_playlist_video_ids_safe(uploads_playlist, candidate_limit)
    if not ids:
        return []

    hydrated = hydrate_video_metadata_safe(ids)
    items = build_profile_items_from_videos(hydrated)
    return sort_profile_items(items, sort_mode)


def resolve_channel_id_by_channel_html_hint(hint_value: str) -> str | None:
    path = hint_value.strip("/")
    if not path:
        return None
    try:
        response = requests.get(f"https://www.youtube.com/c/{path}/videos", timeout=15)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    return _extract_channel_id_from_html(response.text)


def resolve_channel_id_last_resort_search(query: str) -> str | None:
    payload = youtube_api_get(
        YOUTUBE_SEARCH_LIST,
        {
            "part": "snippet",
            "type": "channel",
            "q": query,
            "maxResults": 1,
            "key": YOUTUBE_API_KEY,
        },
    )
    items = payload.get("items", [])
    if not items:
        return None
    return (items[0].get("id") or {}).get("channelId")


def resolve_channel_id_safely(profile_url: str) -> str:
    hint_type, hint_value = extract_channel_hint(profile_url)

    if hint_type == "channel_id":
        return hint_value

    if hint_type == "handle":
        for handle_value in (hint_value, f"@{hint_value}"):
            payload = youtube_api_get(
                YOUTUBE_CHANNELS_LIST,
                {
                    "part": "id",
                    "forHandle": handle_value,
                    "maxResults": 1,
                    "key": YOUTUBE_API_KEY,
                },
            )
            items = payload.get("items", [])
            if items and items[0].get("id"):
                return items[0]["id"]

    if hint_type == "username":
        payload = youtube_api_get(
            YOUTUBE_CHANNELS_LIST,
            {
                "part": "id",
                "forUsername": hint_value,
                "maxResults": 1,
                "key": YOUTUBE_API_KEY,
            },
        )
        items = payload.get("items", [])
        if items and items[0].get("id"):
            return items[0]["id"]

    if hint_type == "query":
        resolved_from_html = resolve_channel_id_by_channel_html_hint(hint_value)
        if resolved_from_html:
            return resolved_from_html
        resolved_from_search = resolve_channel_id_last_resort_search(hint_value)
        if resolved_from_search:
            return resolved_from_search

    raise HTTPException(
        status_code=404,
        detail="Could not resolve that channel. Use a channel URL, @handle, or channel ID.",
    )


def fetch_profile_grid_tier_b(channel_id: str, limit: int, sort_mode: str) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            YOUTUBE_PUBLIC_CHANNEL_VIDEOS_URL.format(channel_id=channel_id),
            timeout=15,
        )
    except requests.RequestException:
        raise HTTPException(
            status_code=429,
            detail="YouTube quota is exhausted and fallback data is unavailable right now.",
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=429,
            detail="YouTube quota is exhausted and fallback data is unavailable right now.",
        )

    initial_data = _extract_yt_initial_data(response.text)
    if initial_data is None:
        fallback_ids = WATCH_URL_RE.findall(response.text)
        deduped = []
        seen = set()
        for video_id in fallback_ids:
            if video_id in seen:
                continue
            seen.add(video_id)
            deduped.append(
                {
                    "id": video_id,
                    "title": "YouTube video",
                    "channelTitle": None,
                    "publishedAt": None,
                    "views": None,
                    "duration": None,
                    "thumbnail": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                    "thumbWidth": None,
                    "thumbHeight": None,
                }
            )
            if len(deduped) >= limit:
                break
        return deduped

    renderers = _deep_find_video_renderers(initial_data)
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for renderer in renderers:
        video_id = renderer.get("videoId")
        if not isinstance(video_id, str) or video_id in seen_ids:
            continue
        seen_ids.add(video_id)

        title = _extract_text_from_renderer(renderer.get("title")) or "YouTube video"
        published_text = _extract_text_from_renderer(renderer.get("publishedTimeText"))
        thumbs = ((renderer.get("thumbnail") or {}).get("thumbnails") or [])
        thumb = thumbs[-1] if thumbs else {}
        duration_text = _extract_text_from_renderer(renderer.get("lengthText"))
        duration_seconds = _parse_duration_text_to_seconds(duration_text)

        items.append(
            {
                "id": video_id,
                "title": title,
                "channelTitle": None,
                "publishedAt": published_text,
                "views": None,
                "duration": duration_seconds,
                "thumbnail": thumb.get("url") or f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                "thumbWidth": thumb.get("width"),
                "thumbHeight": thumb.get("height"),
            }
        )

    if sort_mode == "popular":
        # Fallback has no reliable stats, keep source order to avoid fake ranking.
        return items[:limit]
    return items[:limit]

CHANNEL_UPLOADS_CACHE: dict[str, tuple[float, str]] = {}
CHANNEL_UPLOADS_TTL_SECONDS = 30 * 60
WINNERS_CACHE_TTL_SECONDS = 45 * 60
WINNERS_SOURCE_CACHE_TTL_SECONDS = 6 * 60 * 60
SHORTS_ASPECT_RATIO_MAX = 0.8
VIDEOS_ASPECT_RATIO_MIN = 1.2
FILTER_CACHE_VERSION = "v2"
ESTABLISHED_CHANNEL_RESOLVE_CACHE: dict[str, tuple[float, str]] = {}
ESTABLISHED_CHANNEL_RESOLVE_TTL_SECONDS = 7 * 24 * 60 * 60
ESTABLISHED_RECENT_IDS_CACHE: dict[str, tuple[float, list[str]]] = {}
ESTABLISHED_RECENT_IDS_TTL_SECONDS = 6 * 60 * 60
PROFILE_GRID_CACHE_TTL_SECONDS = 30 * 60
PROFILE_RATE_LIMIT_WINDOW_SECONDS = 60
PROFILE_RATE_LIMIT_MAX_REQUESTS = 30
PROFILE_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = {}
API_RATE_LIMIT_WINDOW_SECONDS = 60
API_RATE_LIMIT_MAX_REQUESTS = 30
API_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = {}
YOUTUBE_PUBLIC_CHANNEL_VIDEOS_URL = "https://www.youtube.com/channel/{channel_id}/videos"
CHANNEL_ID_RE = re.compile(r'"channelId":"(UC[a-zA-Z0-9_-]{20,})"')
YT_INITIAL_DATA_RE = re.compile(r"var ytInitialData = (\{.*?\});</script>", re.DOTALL)
WATCH_URL_RE = re.compile(r'"url":"\\/watch\\?v=([A-Za-z0-9_-]{11})')


def parse_iso8601_datetime(value: str):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def best_thumbnail_object(thumbnails: dict) -> dict | None:
    for key in ("maxres", "standard", "high", "medium", "default"):
        t = thumbnails.get(key)
        if t and "url" in t:
            return t
    return None


def best_thumbnail_url(thumbnails: dict) -> str | None:
    thumb_obj = best_thumbnail_object(thumbnails)
    if thumb_obj is None:
        return None
    return thumb_obj.get("url")


def thumbnail_aspect_ratio_from_dims(width: int | float | None, height: int | float | None) -> float | None:
    try:
        w = float(width or 0)
        h = float(height or 0)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return w / h


def pick_aspect_ratio(candidate: dict) -> float | None:
    insights = candidate.get("thumb_insights") or {}
    insight_ratio = insights.get("aspect_ratio")
    if isinstance(insight_ratio, (int, float)) and insight_ratio > 0:
        return float(insight_ratio)
    metadata_ratio = candidate.get("thumbnail_aspect_ratio")
    if isinstance(metadata_ratio, (int, float)) and metadata_ratio > 0:
        return float(metadata_ratio)
    return None


def matches_winners_type(candidate: dict, selected_type: str) -> bool:
    if selected_type == "all":
        return True

    aspect_ratio = pick_aspect_ratio(candidate)

    def get_short_confirmed() -> bool:
        cached = candidate.get("short_confirmed")
        if isinstance(cached, bool):
            return cached
        video_id = candidate.get("video_id") or candidate.get("id")
        if not video_id:
            candidate["short_confirmed"] = False
            return False
        confirmed = is_confirmed_short(str(video_id))
        candidate["short_confirmed"] = confirmed
        return confirmed

    if selected_type == "shorts":
        if aspect_ratio is not None:
            if aspect_ratio <= SHORTS_ASPECT_RATIO_MAX:
                return True
        if bool(candidate.get("is_short")):
            return True
        return get_short_confirmed()

    # videos
    if aspect_ratio is not None and aspect_ratio <= SHORTS_ASPECT_RATIO_MAX:
        return False
    if bool(candidate.get("is_short")):
        return False
    if get_short_confirmed():
        return False
    return True


def fetch_uploads_playlist(channel_id: str) -> str | None:
    cache_key = f"uploads_playlist:{channel_id}"
    hit = CHANNEL_UPLOADS_CACHE.get(cache_key)
    if hit:
        expires_at, value = hit
        if time.time() <= expires_at:
            return value
        CHANNEL_UPLOADS_CACHE.pop(cache_key, None)

    payload = youtube_api_get(
        YOUTUBE_CHANNELS_LIST,
        {
            "part": "contentDetails",
            "id": channel_id,
            "key": YOUTUBE_API_KEY,
        },
    )
    items = payload.get("items", [])
    if not items:
        return None

    playlist = (
        ((items[0].get("contentDetails") or {}).get("relatedPlaylists") or {}).get("uploads")
    )
    if playlist:
        CHANNEL_UPLOADS_CACHE[cache_key] = (time.time() + CHANNEL_UPLOADS_TTL_SECONDS, playlist)
    return playlist


def fetch_playlist_video_ids(playlist_id: str, max_items: int) -> list[str]:
    ids = []
    page_token = None
    while len(ids) < max_items:
        params = {
            "part": "contentDetails",
            "playlistId": playlist_id,
            "maxResults": min(50, max_items - len(ids)),
            "key": YOUTUBE_API_KEY,
        }
        if page_token:
            params["pageToken"] = page_token

        pdata = youtube_api_get(YOUTUBE_PLAYLIST_ITEMS_LIST, params)
        page_token = pdata.get("nextPageToken")

        for it in pdata.get("items", []):
            vid = (it.get("contentDetails") or {}).get("videoId")
            if vid:
                ids.append(vid)
                if len(ids) >= max_items:
                    break

        if not page_token:
            break

    return ids[:max_items]


def hydrate_video_metadata(video_ids: list[str]) -> list[dict]:
    hydrated = []
    for batch in chunked(video_ids, 50):
        payload = youtube_api_get(
            YOUTUBE_VIDEOS_LIST,
            {
                "part": "snippet,statistics,contentDetails",
                "id": ",".join(batch),
                "key": YOUTUBE_API_KEY,
            },
        )
        hydrated.extend(payload.get("items", []))
    return hydrated


def fetch_short_chart_videos(region: str, category_id: str | None, max_pages: int = 1) -> tuple[list[dict], int]:
    region = (region or "US").upper()
    if region == "GLOBAL":
        merged: list[dict] = []
        seen_ids: set[str] = set()
        for one_region in TRENDING_REGIONS:
            regional_items, _ = fetch_short_chart_videos(one_region, category_id, max_pages=1)
            for item in regional_items:
                vid = item.get("id")
                if not vid or vid in seen_ids:
                    continue
                seen_ids.add(vid)
                merged.append(item)
        return merged, 1

    collected = []
    page_token = None
    pages_loaded = 0

    while pages_loaded < max_pages:
        params = {
            "part": "snippet,statistics,contentDetails",
            "chart": "mostPopular",
            "regionCode": region,
            "maxResults": 50,
            "key": YOUTUBE_API_KEY,
        }
        if category_id:
            params["videoCategoryId"] = category_id
        if page_token:
            params["pageToken"] = page_token

        payload = youtube_api_get(YOUTUBE_VIDEOS_LIST, params)
        for item in payload.get("items", []):
            duration_seconds = iso8601_duration_to_seconds(
                (item.get("contentDetails") or {}).get("duration", "")
            )
            if duration_seconds <= 60:
                collected.append(item)

        page_token = payload.get("nextPageToken")
        pages_loaded += 1
        if not page_token:
            break

    return collected, pages_loaded


def build_candidates(
    videos: list[dict],
    now: datetime,
    window_days: int,
    min_age_hours: int,
    min_views: int,
) -> list[dict]:
    cutoff = now - timedelta(days=window_days)
    candidates = []

    for v in videos:
        snip = v.get("snippet", {})
        stats = v.get("statistics", {})
        details = v.get("contentDetails", {})
        published_at = parse_iso8601_datetime(snip.get("publishedAt") or "")
        if not published_at:
            continue

        if published_at < cutoff:
            continue

        age_seconds = (now - published_at).total_seconds()
        age_hours = age_seconds / 3600 if age_seconds > 0 else 0.0
        age_days = age_hours / 24
        if age_hours < min_age_hours:
            continue
        if age_days <= 0:
            continue

        view_count = int(stats.get("viewCount", 0) or 0)
        if view_count < min_views:
            continue

        duration_seconds = iso8601_duration_to_seconds(details.get("duration", ""))
        is_short = duration_seconds <= 60
        thumb_obj = best_thumbnail_object(snip.get("thumbnails", {}) or {}) or {}
        thumbnail_url = thumb_obj.get("url")
        if not thumbnail_url:
            continue
        thumbnail_width = int(thumb_obj.get("width", 0) or 0)
        thumbnail_height = int(thumb_obj.get("height", 0) or 0)
        thumbnail_aspect_ratio = thumbnail_aspect_ratio_from_dims(thumbnail_width, thumbnail_height)

        views_per_day = view_count / max(age_days, 1)

        candidates.append({
            "video_id": v.get("id"),
            "title": snip.get("title"),
            "published_at": published_at.isoformat(),
            "view_count": view_count,
            "thumbnail_url": thumbnail_url,
            "thumbnail_width": thumbnail_width,
            "thumbnail_height": thumbnail_height,
            "thumbnail_aspect_ratio": thumbnail_aspect_ratio,
            "is_short": is_short,
            "duration_seconds": duration_seconds,
            "age_days": age_days,
            "age_hours": age_hours,
            "views_per_day": views_per_day,
        })

    return candidates


def median_or_default(values: list[float], default: float = 1.0) -> float:
    if not values:
        return default
    try:
        return median(values)
    except StatisticsError:
        return default


def compute_outlier_scores(candidates: list[dict], baseline: float) -> list[dict]:
    scored = []
    denom = max(math.log1p(baseline), 0.0001)
    for c in candidates:
        score = math.log1p(c["views_per_day"]) / denom
        entry = c.copy()
        entry["outlier_score"] = score
        entry["url"] = f"https://www.youtube.com/watch?v={c['video_id']}"
        scored.append(entry)
    scored.sort(key=lambda x: x["outlier_score"], reverse=True)
    return scored


def resolve_winners_type(type_value: str | None, format_value: str | None) -> str:
    if not isinstance(type_value, str):
        type_value = None
    if not isinstance(format_value, str):
        format_value = None
    selected = (type_value or format_value or "videos").lower()
    if selected not in {"all", "videos", "shorts"}:
        raise HTTPException(status_code=400, detail="type must be all, videos, or shorts")
    return selected


def winner_primary_value(item: dict, sort_mode: str) -> float:
    if sort_mode == "total":
        return float(item.get("view_count", 0))
    if sort_mode == "hot":
        return float(item.get("hot_score", item.get("views_per_day", 0.0)))
    return float(item.get("outlier_score", 0.0))


def sort_winner_candidates(candidates: list[dict], sort_mode: str, tie_break_quality: bool = False) -> list[dict]:
    def key_fn(item: dict):
        primary = winner_primary_value(item, sort_mode)
        if not tie_break_quality:
            return (primary,)
        insights = item.get("thumb_insights") or {}
        quality = float(insights.get("quality_score", 0))
        return (primary, quality)

    return sorted(candidates, key=key_fn, reverse=True)


def apply_quality_filter(
    ranked: list[dict],
    selected_type: str,
    limit: int,
    min_quality: int,
    sort_mode: str,
) -> tuple[list[dict], int, int]:
    quality_threshold = max(0, min(100, int(min_quality)))
    analysis_cap = min(len(ranked), max(limit * 10, 300))
    quality_pool = ranked[:analysis_cap]
    enriched_pool = []

    for item in quality_pool:
        enriched = item.copy()
        insights = analyze_thumbnail(item.get("thumbnail_url"))
        enriched["thumb_insights"] = insights
        if matches_winners_type(enriched, selected_type):
            enriched_pool.append(enriched)

    filtered = [
        item for item in enriched_pool
        if int((item.get("thumb_insights") or {}).get("quality_score", 0)) >= quality_threshold
    ]
    return (
        sort_winner_candidates(filtered, sort_mode=sort_mode, tie_break_quality=True),
        len(quality_pool),
        quality_threshold,
    )


def fetch_global_winner_videos(region: str, category_id: str | None, max_pages: int = 1) -> tuple[list[dict], int]:
    region = (region or "US").upper()
    if region == "GLOBAL":
        merged: list[dict] = []
        seen_ids: set[str] = set()
        pages_loaded = 0
        for one_region in TRENDING_REGIONS:
            items, pages = fetch_global_winner_videos(one_region, category_id, max_pages=max_pages)
            pages_loaded = max(pages_loaded, pages)
            for item in items:
                vid = item.get("id")
                if not vid or vid in seen_ids:
                    continue
                seen_ids.add(vid)
                merged.append(item)
        return merged, pages_loaded or 1

    all_items: list[dict] = []
    page_token = None
    pages_loaded = 0

    while pages_loaded < max_pages:
        source_key = f"winners-source:{region}:{category_id or 'all'}:{page_token or 'first'}"
        cached = cache_get(source_key)
        if cached is None:
            params = {
                "part": "snippet,statistics,contentDetails",
                "chart": "mostPopular",
                "regionCode": region,
                "maxResults": 50,
                "key": YOUTUBE_API_KEY,
            }
            if category_id:
                params["videoCategoryId"] = category_id
            if page_token:
                params["pageToken"] = page_token

            payload = youtube_api_get(YOUTUBE_VIDEOS_LIST, params)
            page_items = payload.get("items", [])
            next_token = payload.get("nextPageToken")
            cached = {"items": page_items, "next_page_token": next_token}
            cache_set_custom(source_key, cached, WINNERS_SOURCE_CACHE_TTL_SECONDS)

        all_items.extend(cached.get("items", []))
        pages_loaded += 1
        page_token = cached.get("next_page_token")
        if not page_token:
            break

    return all_items, pages_loaded


def fetch_global_short_winner_videos(
    region: str,
    category_id: str | None,
    max_pages: int = 1,
) -> tuple[list[dict], int]:
    videos, pages = fetch_short_chart_videos(region, category_id, max_pages=max_pages)
    return videos, pages


def is_confirmed_short(video_id: str) -> bool:
    """
    Strong Shorts check:
    - True when /shorts/{id} resolves directly (HTTP 200)
    - False when it redirects to /watch (typical long-form behavior)
    """
    if not video_id:
        return False

    hit = SHORTS_DETECT_CACHE.get(video_id)
    if hit:
        expires_at, value = hit
        if time.time() <= expires_at:
            return value
        SHORTS_DETECT_CACHE.pop(video_id, None)

    is_short = False
    try:
        r = requests.get(
            f"https://www.youtube.com/shorts/{video_id}",
            allow_redirects=False,
            timeout=8,
        )
        if r.status_code == 200:
            is_short = True
        elif r.status_code in (301, 302, 303, 307, 308):
            location = (r.headers.get("Location") or "").lower()
            is_short = "/watch" not in location
    except requests.RequestException:
        # Conservative fallback: if uncertain, do not classify as Shorts.
        is_short = False

    SHORTS_DETECT_CACHE[video_id] = (time.time() + SHORTS_DETECT_TTL_SECONDS, is_short)
    return is_short


# ---------------------------
# App setup
# ---------------------------

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise RuntimeError("Missing YOUTUBE_API_KEY in backend/.env")


def parse_cors_origins() -> tuple[list[str], bool]:
    raw = (os.getenv("CORS_ALLOWED_ORIGINS") or "").strip()
    if not raw:
        return ["http://localhost:5173"], True
    if raw == "*":
        return ["*"], False
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if not origins:
        return ["http://localhost:5173"], True
    return origins, True

app = FastAPI()

cors_origins, cors_credentials = parse_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(YouTubeQuotaExceededError)
async def youtube_quota_exceeded_handler(_request: Request, _exc: YouTubeQuotaExceededError):
    return JSONResponse(
        status_code=429,
        content={
            "detail": "YouTube API quota is currently exhausted. Showing cached data where available.",
            "error_code": "youtube_quota_exhausted",
        },
    )


@app.on_event("startup")
def on_startup_refresh_established_pool():
    load_pattern_library()
    maybe_schedule_established_pool_refresh()

YOUTUBE_VIDEOS_LIST = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_SEARCH_LIST = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_CHANNELS_LIST = "https://www.googleapis.com/youtube/v3/channels"
YOUTUBE_PLAYLIST_ITEMS_LIST = "https://www.googleapis.com/youtube/v3/playlistItems"

WINNERS_CATEGORY_MAP = {
    "all": None,
    "gaming": "20",
    "music": "10",
    "entertainment": "24",
    "sports": "17",
}

ESTABLISHED_GLOBAL_CREATORS = [
    {"identifier": "MrBeast", "categories": {"all", "entertainment"}, "formats": {"videos", "shorts"}},
    {"identifier": "Dude Perfect", "categories": {"all", "entertainment", "sports"}, "formats": {"videos", "shorts"}},
    {"identifier": "Taylor Swift", "categories": {"all", "music"}, "formats": {"videos", "shorts"}},
    {"identifier": "Ed Sheeran", "categories": {"all", "music"}, "formats": {"videos", "shorts"}},
    {"identifier": "NBA", "categories": {"all", "sports"}, "formats": {"videos", "shorts"}},
    {"identifier": "UFC", "categories": {"all", "sports"}, "formats": {"videos", "shorts"}},
    {"identifier": "Markiplier", "categories": {"all", "gaming"}, "formats": {"videos", "shorts"}},
    {"identifier": "khaby.lame", "categories": {"all", "entertainment"}, "formats": {"shorts"}},
    {"identifier": "zachking", "categories": {"all", "entertainment"}, "formats": {"shorts"}},
]

ESTABLISHED_CREATORS_BY_REGION: dict[str, list[dict[str, set[str] | str]]] = {
    "US": [
        {"identifier": "MrBeast", "categories": {"all", "entertainment"}, "formats": {"videos", "shorts"}},
        {"identifier": "Dude Perfect", "categories": {"all", "entertainment", "sports"}, "formats": {"videos", "shorts"}},
        {"identifier": "Mark Rober", "categories": {"all", "entertainment"}, "formats": {"videos", "shorts"}},
        {"identifier": "MKBHD", "categories": {"all", "entertainment"}, "formats": {"videos", "shorts"}},
        {"identifier": "Ryan Trahan", "categories": {"all", "entertainment"}, "formats": {"videos", "shorts"}},
        {"identifier": "PewDiePie", "categories": {"all", "entertainment", "gaming"}, "formats": {"videos"}},
        {"identifier": "Markiplier", "categories": {"all", "gaming"}, "formats": {"videos", "shorts"}},
        {"identifier": "Game Theory", "categories": {"all", "gaming"}, "formats": {"videos"}},
        {"identifier": "Ninja", "categories": {"all", "gaming"}, "formats": {"videos", "shorts"}},
        {"identifier": "Taylor Swift", "categories": {"all", "music"}, "formats": {"videos", "shorts"}},
        {"identifier": "Billie Eilish", "categories": {"all", "music"}, "formats": {"videos", "shorts"}},
        {"identifier": "Ariana Grande", "categories": {"all", "music"}, "formats": {"videos"}},
        {"identifier": "EminemMusic", "categories": {"all", "music"}, "formats": {"videos"}},
        {"identifier": "NBA", "categories": {"all", "sports"}, "formats": {"videos", "shorts"}},
        {"identifier": "NFL", "categories": {"all", "sports"}, "formats": {"videos", "shorts"}},
        {"identifier": "ESPN", "categories": {"all", "sports"}, "formats": {"videos", "shorts"}},
        {"identifier": "UFC", "categories": {"all", "sports"}, "formats": {"videos", "shorts"}},
    ],
    "GB": [
        {"identifier": "Sidemen", "categories": {"all", "entertainment"}, "formats": {"videos", "shorts"}},
        {"identifier": "KSI", "categories": {"all", "entertainment", "music"}, "formats": {"videos", "shorts"}},
        {"identifier": "DanTDM", "categories": {"all", "gaming"}, "formats": {"videos", "shorts"}},
        {"identifier": "AliA", "categories": {"all", "gaming"}, "formats": {"videos", "shorts"}},
        {"identifier": "Memeulous", "categories": {"all", "entertainment"}, "formats": {"videos"}},
        {"identifier": "Coldplay", "categories": {"all", "music"}, "formats": {"videos", "shorts"}},
        {"identifier": "Ed Sheeran", "categories": {"all", "music"}, "formats": {"videos", "shorts"}},
        {"identifier": "Dua Lipa", "categories": {"all", "music"}, "formats": {"videos", "shorts"}},
        {"identifier": "Premier League", "categories": {"all", "sports"}, "formats": {"videos", "shorts"}},
        {"identifier": "Sky Sports", "categories": {"all", "sports"}, "formats": {"videos", "shorts"}},
        {"identifier": "BT Sport", "categories": {"all", "sports"}, "formats": {"videos"}},
    ],
    "CA": [
        {"identifier": "Linus Tech Tips", "categories": {"all", "entertainment"}, "formats": {"videos"}},
        {"identifier": "Nelk", "categories": {"all", "entertainment"}, "formats": {"videos"}},
        {"identifier": "VanossGaming", "categories": {"all", "gaming"}, "formats": {"videos"}},
        {"identifier": "Typical Gamer", "categories": {"all", "gaming"}, "formats": {"videos", "shorts"}},
        {"identifier": "Justin Bieber", "categories": {"all", "music"}, "formats": {"videos", "shorts"}},
        {"identifier": "The Weeknd", "categories": {"all", "music"}, "formats": {"videos", "shorts"}},
        {"identifier": "Drake", "categories": {"all", "music"}, "formats": {"videos"}},
        {"identifier": "Sportsnet", "categories": {"all", "sports"}, "formats": {"videos", "shorts"}},
        {"identifier": "TSN", "categories": {"all", "sports"}, "formats": {"videos"}},
    ],
    "AU": [
        {"identifier": "LazarBeam", "categories": {"all", "gaming", "entertainment"}, "formats": {"videos", "shorts"}},
        {"identifier": "Muselk", "categories": {"all", "gaming"}, "formats": {"videos", "shorts"}},
        {"identifier": "HowToBasic", "categories": {"all", "entertainment"}, "formats": {"videos", "shorts"}},
        {"identifier": "Yes Theory", "categories": {"all", "entertainment"}, "formats": {"videos"}},
        {"identifier": "Tones And I", "categories": {"all", "music"}, "formats": {"videos"}},
        {"identifier": "Sia", "categories": {"all", "music"}, "formats": {"videos", "shorts"}},
        {"identifier": "Flume", "categories": {"all", "music"}, "formats": {"videos"}},
        {"identifier": "AFL", "categories": {"all", "sports"}, "formats": {"videos", "shorts"}},
        {"identifier": "NRL", "categories": {"all", "sports"}, "formats": {"videos", "shorts"}},
    ],
}

ESTABLISHED_POOL_FILE = Path(__file__).resolve().parent / "data" / "established_creators.json"
ESTABLISHED_POOL_STALE_SECONDS = 7 * 24 * 60 * 60
ESTABLISHED_POOL_STARTUP_REFRESH_ENABLED = (
    os.getenv("ESTABLISHED_POOL_STARTUP_REFRESH", "0").strip().lower() in {"1", "true", "yes"}
)
_ESTABLISHED_REFRESH_STARTED = False


def _normalize_creator_entry(entry: dict[str, Any]) -> dict[str, set[str] | str] | None:
    identifier = str(entry.get("identifier") or "").strip()
    if not identifier:
        return None
    categories = entry.get("categories") or []
    formats = entry.get("formats") or []
    categories_set = {str(value).strip().lower() for value in categories if str(value).strip()}
    formats_set = {str(value).strip().lower() for value in formats if str(value).strip()}
    if not categories_set:
        categories_set = {"all"}
    if not formats_set:
        formats_set = {"videos", "shorts"}
    return {
        "identifier": identifier,
        "categories": categories_set,
        "formats": formats_set,
    }


def load_established_pool_from_file() -> bool:
    global ESTABLISHED_GLOBAL_CREATORS, ESTABLISHED_CREATORS_BY_REGION
    if not ESTABLISHED_POOL_FILE.exists():
        return False

    try:
        payload = json.loads(ESTABLISHED_POOL_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    regions_payload = payload.get("regions")
    if not isinstance(regions_payload, dict):
        return False

    merged_regions = dict(ESTABLISHED_CREATORS_BY_REGION)
    applied = False
    for region_key, entries in regions_payload.items():
        if not isinstance(region_key, str) or not isinstance(entries, list):
            continue
        normalized = []
        for raw in entries:
            if not isinstance(raw, dict):
                continue
            entry = _normalize_creator_entry(raw)
            if entry:
                normalized.append(entry)
        if normalized:
            merged_regions[region_key.upper()] = normalized
            applied = True

    global_entries = payload.get("global")
    if isinstance(global_entries, list):
        normalized_global = []
        for raw in global_entries:
            if not isinstance(raw, dict):
                continue
            entry = _normalize_creator_entry(raw)
            if entry:
                normalized_global.append(entry)
        if normalized_global:
            ESTABLISHED_GLOBAL_CREATORS = normalized_global

    if applied:
        ESTABLISHED_CREATORS_BY_REGION = merged_regions
    return applied


def established_pool_is_stale() -> bool:
    if not ESTABLISHED_POOL_FILE.exists():
        return True
    try:
        payload = json.loads(ESTABLISHED_POOL_FILE.read_text(encoding="utf-8"))
        updated_raw = payload.get("updated_at")
        if isinstance(updated_raw, str):
            updated = parse_iso8601_datetime(updated_raw)
            if updated is not None:
                age = (datetime.now(timezone.utc) - updated).total_seconds()
                return age > ESTABLISHED_POOL_STALE_SECONDS
    except (OSError, json.JSONDecodeError):
        pass

    try:
        mtime = ESTABLISHED_POOL_FILE.stat().st_mtime
    except OSError:
        return True
    return (time.time() - mtime) > ESTABLISHED_POOL_STALE_SECONDS


def refresh_established_pool_background(min_per_region: int = 50) -> None:
    script_path = Path(__file__).resolve().parent / "scripts" / "refresh_established_pool.py"
    if not script_path.exists():
        return

    env = os.environ.copy()
    env.setdefault("YOUTUBE_API_KEY", YOUTUBE_API_KEY)
    cmd = [
        sys.executable,
        str(script_path),
        "--min-per-region",
        str(max(50, min_per_region)),
        "--output",
        str(ESTABLISHED_POOL_FILE),
    ]
    try:
        subprocess.run(
            cmd,
            check=False,
            timeout=300,
            cwd=str(Path(__file__).resolve().parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        load_established_pool_from_file()
    except (OSError, subprocess.SubprocessError):
        return


def maybe_schedule_established_pool_refresh() -> None:
    global _ESTABLISHED_REFRESH_STARTED
    if not ESTABLISHED_POOL_STARTUP_REFRESH_ENABLED:
        return
    if _ESTABLISHED_REFRESH_STARTED:
        return
    if not established_pool_is_stale():
        return

    _ESTABLISHED_REFRESH_STARTED = True
    worker = threading.Thread(target=refresh_established_pool_background, daemon=True)
    worker.start()


load_established_pool_from_file()


def extract_channel_hint(profile_url: str) -> tuple[str, str]:
    """
    Parse common channel/profile URL styles.
    Returns: (hint_type, hint_value)
    hint_type: channel_id | handle | username | query
    """
    raw = (profile_url or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="profile_url is required")

    # Accept direct UC... channel ids as input.
    if raw.startswith("UC") and len(raw) >= 24:
        return "channel_id", raw

    # Accept plain @handle.
    if raw.startswith("@"):
        return "handle", raw[1:].strip()

    m = re.search(r"youtube\.com/channel/([A-Za-z0-9_-]+)", raw, flags=re.IGNORECASE)
    if m:
        return "channel_id", m.group(1)

    m = re.search(r"youtube\.com/@([A-Za-z0-9._-]+)", raw, flags=re.IGNORECASE)
    if m:
        return "handle", m.group(1)

    m = re.search(r"youtube\.com/user/([A-Za-z0-9._-]+)", raw, flags=re.IGNORECASE)
    if m:
        return "username", m.group(1)

    m = re.search(r"youtube\.com/c/([A-Za-z0-9._-]+)", raw, flags=re.IGNORECASE)
    if m:
        return "query", m.group(1)

    return "query", raw


def resolve_channel_id(profile_url: str) -> str:
    return resolve_channel_id_safely(profile_url)


def resolve_channel_id_cached(identifier: str) -> str:
    key = (identifier or "").strip().lower()
    hit = ESTABLISHED_CHANNEL_RESOLVE_CACHE.get(key)
    if hit:
        expires_at, value = hit
        if time.time() <= expires_at:
            return value
        ESTABLISHED_CHANNEL_RESOLVE_CACHE.pop(key, None)

    channel_id = resolve_channel_id(identifier)
    ESTABLISHED_CHANNEL_RESOLVE_CACHE[key] = (
        time.time() + ESTABLISHED_CHANNEL_RESOLVE_TTL_SECONDS,
        channel_id,
    )
    return channel_id


def resolve_channel_id_safely_cached(identifier: str) -> str:
    key = f"safe:{(identifier or '').strip().lower()}"
    hit = ESTABLISHED_CHANNEL_RESOLVE_CACHE.get(key)
    if hit:
        expires_at, value = hit
        if time.time() <= expires_at:
            return value
        ESTABLISHED_CHANNEL_RESOLVE_CACHE.pop(key, None)

    channel_id = resolve_channel_id_safely(identifier)
    ESTABLISHED_CHANNEL_RESOLVE_CACHE[key] = (
        time.time() + ESTABLISHED_CHANNEL_RESOLVE_TTL_SECONDS,
        channel_id,
    )
    return channel_id


def fetch_recent_playlist_video_ids_cached(playlist_id: str, max_items: int) -> list[str]:
    cache_key = f"recent-ids:{playlist_id}:{max_items}"
    hit = ESTABLISHED_RECENT_IDS_CACHE.get(cache_key)
    if hit:
        expires_at, value = hit
        if time.time() <= expires_at:
            return value
        ESTABLISHED_RECENT_IDS_CACHE.pop(cache_key, None)

    ids = fetch_playlist_video_ids(playlist_id, max_items)
    ESTABLISHED_RECENT_IDS_CACHE[cache_key] = (
        time.time() + ESTABLISHED_RECENT_IDS_TTL_SECONDS,
        ids,
    )
    return ids


def filter_established_creators(selected_type: str, category_key: str, region: str) -> list[dict]:
    region_key = (region or "").upper()
    regional = ESTABLISHED_CREATORS_BY_REGION.get(region_key) or []
    pools = regional + ESTABLISHED_GLOBAL_CREATORS

    filtered = []
    seen_identifiers: set[str] = set()
    for creator in pools:
        categories = creator.get("categories") or set()
        formats = creator.get("formats") or set()
        identifier = str(creator.get("identifier") or "").strip().lower()
        if not identifier or identifier in seen_identifiers:
            continue
        if category_key not in categories and "all" not in categories:
            continue
        if selected_type != "all" and selected_type not in formats:
            continue
        seen_identifiers.add(identifier)
        filtered.append(creator)
    return filtered


def fetch_established_creator_videos(
    selected_type: str,
    category_key: str,
    region: str,
    candidate_limit: int,
) -> tuple[list[dict], int]:
    cache_key = (
        f"winners-established:{FILTER_CACHE_VERSION}:{region}:{selected_type}:{category_key}:{candidate_limit}"
    )
    cached = cache_get(cache_key)
    if cached is not None:
        return cached.get("videos", []), cached.get("creator_count", 0)

    creators = filter_established_creators(selected_type, category_key, region)
    max_creators = max(16, min(64, candidate_limit // 3))
    creators = creators[:max_creators]
    creator_count = len(creators)
    quota_limited = False

    seen_ids: set[str] = set()
    video_ids: list[str] = []
    recent_per_creator = max(6, min(15, candidate_limit // max(1, creator_count // 2 or 1)))

    for creator in creators:
        identifier = str(creator.get("identifier") or "").strip()
        if not identifier:
            continue
        try:
            channel_id = resolve_channel_id_cached(identifier)
            playlist_id = fetch_uploads_playlist(channel_id)
        except (HTTPException, YouTubeQuotaExceededError) as exc:
            if is_quota_exceeded_error(exc):
                quota_limited = True
                break
            continue
        if not playlist_id:
            continue

        try:
            recent_ids = fetch_recent_playlist_video_ids_cached(playlist_id, recent_per_creator)
        except (HTTPException, YouTubeQuotaExceededError) as exc:
            if is_quota_exceeded_error(exc):
                quota_limited = True
                break
            continue

        for vid in recent_ids:
            if vid in seen_ids:
                continue
            seen_ids.add(vid)
            video_ids.append(vid)
            if len(video_ids) >= candidate_limit:
                break
        if len(video_ids) >= candidate_limit:
            break

    videos: list[dict] = []
    if video_ids:
        try:
            videos = hydrate_video_metadata(video_ids[:candidate_limit])
        except (HTTPException, YouTubeQuotaExceededError) as exc:
            if is_quota_exceeded_error(exc):
                quota_limited = True
                videos = []
            else:
                raise

    payload = {"videos": videos, "creator_count": creator_count}
    ttl = 15 * 60 if quota_limited else WINNERS_SOURCE_CACHE_TTL_SECONDS
    cache_set_custom(cache_key, payload, ttl)
    return videos, creator_count


def merge_weighted_winners(
    top_items: list[dict],
    established_items: list[dict],
    limit: int,
    top_weight: float = 0.5,
) -> tuple[list[dict], dict[str, int]]:
    top_quota = int(round(limit * top_weight))
    top_quota = max(0, min(limit, top_quota))
    established_quota = max(0, limit - top_quota)

    merged: list[dict] = []
    seen: set[str] = set()
    mix = {"top_performers": 0, "established_creators": 0}

    def take(source: list[dict], quota: int, label: str) -> None:
        for item in source:
            if quota <= 0:
                return
            vid = item.get("video_id")
            if not vid or vid in seen:
                continue
            entry = item.copy()
            entry["source_group"] = label
            merged.append(entry)
            seen.add(vid)
            mix["top_performers" if label == "top_performers" else "established_creators"] += 1
            quota -= 1

    take(top_items, top_quota, "top_performers")
    take(established_items, established_quota, "established_creators")

    if len(merged) < limit:
        overflow = [("top_performers", item) for item in top_items] + [
            ("established_creators", item) for item in established_items
        ]
        for label, item in overflow:
            if len(merged) >= limit:
                break
            vid = item.get("video_id")
            if not vid or vid in seen:
                continue
            entry = item.copy()
            entry["source_group"] = label
            merged.append(entry)
            seen.add(vid)
            mix["top_performers" if label == "top_performers" else "established_creators"] += 1

    return merged[:limit], mix


def is_quota_exceeded_error(exc: Exception) -> bool:
    detail = str(getattr(exc, "detail", exc))
    lowered = detail.lower()
    return (
        "quotaexceeded" in lowered
        or "quota exceeded" in lowered
        or "youtube.quota" in lowered
        or isinstance(exc, YouTubeQuotaExceededError)
    )


def _quality_band(score: int) -> str:
    if score >= 80:
        return "high"
    if score >= 60:
        return "good"
    if score >= 40:
        return "medium"
    return "low"


def _clutter_band(score: int) -> str:
    if score <= 30:
        return "clean"
    if score <= 55:
        return "moderate"
    return "busy"


def _extract_pattern_item(item: PatternExtractItem) -> dict[str, Any]:
    insights = analyze_thumbnail(item.thumbnail_url)
    quality_score = int(insights.get("quality_score") or 0)
    clutter_score = int(insights.get("clutter_score") or 0)
    aspect_orientation = str(insights.get("aspect_orientation") or "unknown")
    has_face = bool(insights.get("has_face"))
    text_present = bool(insights.get("text_present"))
    signature = (
        f"ar:{aspect_orientation}|face:{int(has_face)}|text:{int(text_present)}|"
        f"quality:{_quality_band(quality_score)}|clutter:{_clutter_band(clutter_score)}"
    )
    return {
        "video_id": item.video_id,
        "title": item.title,
        "channel_title": item.channel_title,
        "thumbnail_url": item.thumbnail_url,
        "features": {
            "has_face": has_face,
            "text_present": text_present,
            "aspect_orientation": aspect_orientation,
            "quality_score": quality_score,
            "quality_band": _quality_band(quality_score),
            "clutter_score": clutter_score,
            "clutter_band": _clutter_band(clutter_score),
            "dominant_color": "unknown",
            "composition": "auto",
        },
        "cluster_signature": signature,
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/resolve")
def resolve_channel(payload: ResolveRequest, request: Request):
    query = (payload.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    enforce_api_rate_limit(request, scope="resolve")
    channel_id = resolve_channel_id_safely_cached(query)
    return {"query": query, "channel_id": channel_id}


@app.post("/patterns/extract")
def extract_patterns(payload: PatternExtractRequest, request: Request):
    if not payload.items:
        raise HTTPException(status_code=400, detail="items must contain at least one thumbnail")
    enforce_api_rate_limit(request, scope="patterns_extract")

    enriched_items = [_extract_pattern_item(item) for item in payload.items]
    clusters_by_signature: dict[str, list[dict[str, Any]]] = {}
    for item in enriched_items:
        key = item["cluster_signature"]
        clusters_by_signature.setdefault(key, []).append(item)

    clusters: list[dict[str, Any]] = []
    for idx, (signature, members) in enumerate(clusters_by_signature.items(), start=1):
        cluster_id = f"cluster_{idx}"
        for member in members:
            member["cluster_id"] = cluster_id
        clusters.append(
            {
                "cluster_id": cluster_id,
                "signature": signature,
                "count": len(members),
            }
        )

    return {
        "items": enriched_items,
        "clusters": clusters,
        "meta": {
            "input_count": len(payload.items),
            "cluster_count": len(clusters),
        },
    }


@app.post("/patterns/save")
def save_pattern(payload: PatternSaveRequest, request: Request):
    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    enforce_api_rate_limit(request, scope="patterns_save")

    pattern_id = uuid.uuid4().hex[:12]
    entry = {
        "pattern_id": pattern_id,
        "name": name,
        "clusters": payload.clusters,
        "filters": payload.filters,
        "notes": payload.notes,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    with PATTERN_LIBRARY_LOCK:
        PATTERN_LIBRARY[pattern_id] = entry
        snapshot = dict(PATTERN_LIBRARY)
    persist_pattern_library(snapshot)
    return {
        "ok": True,
        "pattern_id": pattern_id,
        "pattern": entry,
    }


@app.get("/patterns")
def list_patterns(request: Request):
    enforce_api_rate_limit(request, scope="patterns_list")
    with PATTERN_LIBRARY_LOCK:
        patterns = list(PATTERN_LIBRARY.values())

    patterns.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    summarized = [
        {
            "pattern_id": pattern.get("pattern_id"),
            "name": pattern.get("name"),
            "created_at": pattern.get("created_at"),
            "cluster_count": len(pattern.get("clusters") or []),
        }
        for pattern in patterns
    ]
    return {"items": summarized}


@app.get("/patterns/apply/{pattern_id}")
def apply_pattern(pattern_id: str, request: Request):
    enforce_api_rate_limit(request, scope="patterns_apply")
    with PATTERN_LIBRARY_LOCK:
        pattern = PATTERN_LIBRARY.get(pattern_id)
    if not pattern:
        raise HTTPException(status_code=404, detail="Pattern not found")
    return {"pattern_id": pattern_id, "pattern": pattern}


@app.get("/top")
def top(
    request: Request,
    region: str = "GB",
    max_results: int = 24,
    video_category_id: str | None = None,
    type: str = "all",  # all | videos
    page_token: str | None = None,
):
    """
    Regional chart feed (videos.list chart=mostPopular) + pagination.
    Best for All/Videos. Not Shorts-focused.
    """
    max_results = max(1, min(max_results, 50))
    region = region.upper()
    hl = lang_for_region(region)
    type = (type or "all").lower()
    if type not in {"all", "videos"}:
        raise HTTPException(status_code=400, detail="type must be all or videos")
    enforce_api_rate_limit(request, scope="top")

    cache_key = f"top:{region}:{max_results}:{video_category_id}:{type}:{page_token}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    source_items: list[dict] = []
    next_token = None
    if region == "GLOBAL":
        for one_region in TRENDING_REGIONS:
            params = {
                "part": "snippet,statistics,contentDetails",
                "chart": "mostPopular",
                "regionCode": one_region,
                "hl": lang_for_region(one_region),
                "maxResults": max_results,
                "key": YOUTUBE_API_KEY,
            }
            if video_category_id:
                params["videoCategoryId"] = video_category_id
            source_items.extend(youtube_api_get(YOUTUBE_VIDEOS_LIST, params).get("items", []))
    else:
        params = {
            "part": "snippet,statistics,contentDetails",
            "chart": "mostPopular",
            "regionCode": region,
            "hl": hl,
            "maxResults": max_results,
            "key": YOUTUBE_API_KEY,
        }
        if video_category_id:
            params["videoCategoryId"] = video_category_id
        if page_token:
            params["pageToken"] = page_token

        data = youtube_api_get(YOUTUBE_VIDEOS_LIST, params)
        next_token = data.get("nextPageToken")
        source_items = data.get("items", [])

    items = []
    seen_ids: set[str] = set()
    for v in source_items:
        video_id = v.get("id")
        if video_id in seen_ids:
            continue
        seen_ids.add(video_id)
        snip = v.get("snippet", {})
        stats = v.get("statistics", {})
        details = v.get("contentDetails", {})
        thumbs = snip.get("thumbnails", {})

        duration_seconds = iso8601_duration_to_seconds(details.get("duration", ""))

        # If user selects "videos", enforce >60s (simple definition)
        if type == "videos" and duration_seconds <= 60:
            continue

        thumb_obj = (
            thumbs.get("maxres")
            or thumbs.get("standard")
            or thumbs.get("high")
            or thumbs.get("medium")
            or thumbs.get("default")
            or {}
        )

        items.append({
            "id": video_id,
            "title": snip.get("title"),
            "channelTitle": snip.get("channelTitle"),
            "publishedAt": snip.get("publishedAt"),
            "views": int(stats.get("viewCount", 0) or 0),
            "duration": duration_seconds,
            "thumbnail": thumb_obj.get("url"),
            "thumbWidth": thumb_obj.get("width"),
            "thumbHeight": thumb_obj.get("height"),
            "defaultAudioLanguage": snip.get("defaultAudioLanguage"),
            "defaultLanguage": snip.get("defaultLanguage"),
        })

    items.sort(key=lambda x: int(x.get("views", 0) or 0), reverse=True)
    resp = {"items": items[:max_results], "nextPageToken": next_token}
    cache_set(cache_key, resp)
    return resp


@app.get("/discover")
def discover(
    request: Request,
    region: str = "GB",
    max_results: int = 24,
    video_category_id: str | None = None,

    # Shorts toggles
    require_60s: bool = True,
    require_hash: bool = False,
    verticalish: bool = False,

    # recency window
    days: int = 7,

    # pagination
    page_token: str | None = None,

    # language enforcement (soft)
    enforce_lang: bool = True,
    strict_shorts: bool = True,
    aspect_filter: bool = True,
):
    """
    Shorts discovery feed:
    - videos.list chart feed filtered to short-duration candidates
    - hydrate includes duration + stats from the same payload
    - enforce <=60s (if require_60s)
    - soft language preference using defaultAudioLanguage/defaultLanguage
    - returns nextPageToken for "Load more"
    """
    max_results = max(1, min(max_results, 50))
    days = max(1, min(days, 30))
    region = region.upper()
    enforce_api_rate_limit(request, scope="discover")

    lang = lang_for_region(region)

    cache_key = (
        f"discover:{FILTER_CACHE_VERSION}:{region}:{max_results}:{video_category_id}:{require_60s}:"
        f"{require_hash}:{verticalish}:{days}:{page_token}:{enforce_lang}:{strict_shorts}:{aspect_filter}"
    )
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    short_videos, pages_used = fetch_short_chart_videos(region, video_category_id, max_pages=2)
    next_token = None if pages_used < 2 else "pages=" + str(pages_used)

    matched = []
    other = []

    for v in short_videos:
        snip = v.get("snippet", {})
        stats = v.get("statistics", {})
        details = v.get("contentDetails", {})
        thumbs = snip.get("thumbnails", {})

        title = (snip.get("title") or "").strip()
        duration_seconds = iso8601_duration_to_seconds(details.get("duration", ""))
        thumb_obj = best_thumbnail_object(thumbs) or {}
        thumb_ratio = thumbnail_aspect_ratio_from_dims(thumb_obj.get("width"), thumb_obj.get("height"))
        thumb_url = thumb_obj.get("url") or ""

        if aspect_filter:
            scanned_ratio = None
            if thumb_url:
                insights = analyze_thumbnail(thumb_url)
                scanned_value = insights.get("aspect_ratio")
                if isinstance(scanned_value, (int, float)) and scanned_value > 0:
                    scanned_ratio = float(scanned_value)

            ratio_for_filter = scanned_ratio if scanned_ratio is not None else thumb_ratio
            pass_aspect = ratio_for_filter is not None and ratio_for_filter <= SHORTS_ASPECT_RATIO_MAX
            pass_confirmed = strict_shorts and is_confirmed_short(v.get("id") or "")
            pass_duration_fallback = (
                strict_shorts and require_60s and ratio_for_filter is not None and ratio_for_filter <= 1.0
            )
            if not (pass_aspect or pass_confirmed or pass_duration_fallback):
                continue
        else:
            if require_60s and duration_seconds > 60:
                continue
            confirmed_short = strict_shorts and is_confirmed_short(v.get("id") or "")
            if strict_shorts and not (
                confirmed_short or (require_60s and thumb_ratio is not None and thumb_ratio <= 1.0)
            ):
                continue
            if require_hash and "#shorts" not in title.lower():
                continue
            if verticalish and not is_verticalish(thumbs):
                continue

        item = {
            "id": v.get("id"),
            "title": title,
            "channelTitle": snip.get("channelTitle"),
            "publishedAt": snip.get("publishedAt"),
            "views": int(stats.get("viewCount", 0) or 0),
            "duration": duration_seconds,
            "thumbnail": thumb_url,
            "thumbWidth": thumb_obj.get("width"),
            "thumbHeight": thumb_obj.get("height"),
            "defaultAudioLanguage": snip.get("defaultAudioLanguage"),
            "defaultLanguage": snip.get("defaultLanguage"),
        }

        if enforce_lang and matches_lang(snip, lang):
            matched.append(item)
        else:
            other.append(item)

    matched.sort(key=lambda x: x["views"], reverse=True)
    other.sort(key=lambda x: x["views"], reverse=True)

    items = matched
    if len(items) < max_results:
        items = items + other

    resp = {"items": items[:max_results], "nextPageToken": next_token}
    cache_set(cache_key, resp)
    return resp


@app.get("/profile")
def profile(
    request: Request,
    profile_url: str,
    content_type: str = "all",  # all | shorts | videos
    sort: str = "recent",  # recent | popular
    max_results: int = 24,
    page_token: str | None = None,
    strict_shorts: bool = True,
):
    max_results = max(1, min(max_results, 48))
    content_type = (content_type or "all").lower()
    if content_type not in {"all", "shorts", "videos"}:
        raise HTTPException(status_code=400, detail="content_type must be one of: all, shorts, videos")
    sort_mode = (sort or "recent").lower()
    if sort_mode not in {"recent", "popular"}:
        raise HTTPException(status_code=400, detail="sort must be one of: recent, popular")

    # Legacy arguments kept for API compatibility; deterministic profile grid no longer paginates.
    _ = page_token
    _ = strict_shorts

    enforce_profile_rate_limit(request)
    try:
        channel_id = resolve_channel_id_safely_cached(profile_url)
    except YouTubeQuotaExceededError:
        hint_type, hint_value = extract_channel_hint(profile_url)
        if hint_type != "channel_id":
            raise HTTPException(
                status_code=429,
                detail="YouTube API quota is currently exhausted. Please provide a direct channel ID or try again later.",
            )
        channel_id = hint_value

    cache_key = f"youtube:{channel_id}:{sort_mode}:{max_results}"
    cached = cache_get(cache_key)
    source = "api"
    if cached is None:
        try:
            base_items = fetch_profile_grid_tier_a(channel_id, max_results, sort_mode)
        except YouTubeQuotaExceededError:
            base_items = fetch_profile_grid_tier_b(channel_id, max_results, sort_mode)
            source = "fallback"
        if not base_items:
            raise HTTPException(status_code=404, detail="No public videos available for this channel.")

        payload = {"items": base_items, "source": source}
        cache_set_custom(cache_key, payload, PROFILE_GRID_CACHE_TTL_SECONDS)
        cached = payload
    else:
        source = cached.get("source") or "api"

    typed_items = _filter_profile_items_by_type(cached.get("items", []), content_type)
    if not typed_items:
        raise HTTPException(status_code=404, detail="No videos available for the selected filter.")

    return {
        "items": typed_items[:max_results],
        "nextPageToken": None,
        "meta": {
            "channel_id": channel_id,
            "sort": sort_mode,
            "content_type": content_type,
            "source": source,
            "cache_ttl_seconds": PROFILE_GRID_CACHE_TTL_SECONDS,
        },
    }


@app.get("/youtube/winners")
def winners(
    request: Request,
    type: str | None = Query(default=None),
    format: str | None = None,
    category: str = "all",
    region: str = "US",
    limit: int = 24,
    window_days: int = 180,
    sort: str = "outlier",
    quality: int = 0,
    min_quality: int = 60,
    min_age_hours: int = 48,
    min_views: int = 1000,
):
    selected_type = resolve_winners_type(type, format)
    sort_mode = (sort or "outlier").lower()
    if sort_mode not in {"outlier", "hot", "total"}:
        raise HTTPException(status_code=400, detail="sort must be outlier, hot, or total")
    if quality not in {0, 1}:
        raise HTTPException(status_code=400, detail="quality must be 0 or 1")
    quality_enabled = quality == 1

    category_key = (category or "all").lower()
    if category_key not in WINNERS_CATEGORY_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"category must be one of: {', '.join(WINNERS_CATEGORY_MAP.keys())}",
        )

    if limit not in {12, 24, 48}:
        raise HTTPException(status_code=400, detail="limit must be one of: 12, 24, 48")

    region = region.upper()
    window_days = max(1, min(window_days, 365))
    min_quality = max(0, min(min_quality, 100))
    min_age_hours = max(1, min(min_age_hours, 168))
    min_views = max(1, min(min_views, 1000000))
    enforce_api_rate_limit(request, scope="winners")

    category_id = WINNERS_CATEGORY_MAP[category_key]
    cache_key = (
        f"winners-global:{FILTER_CACHE_VERSION}:{selected_type}:{category_key}:{region}:{limit}:{window_days}:"
        f"{sort_mode}:{quality}:{min_quality}:{min_age_hours}:{min_views}"
    )
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    now = datetime.now(timezone.utc)
    params_seed = {
        "window_days": window_days,
        "min_age_hours": min_age_hours,
        "min_views": min_views,
    }
    relaxations: list[str] = []
    source_relaxations: list[str] = []
    scored_items: list[dict] = []
    candidate_count = 0
    analyzed_count = 0
    source_pages_loaded = 1
    effective_min_quality = min_quality
    source_mix = {"top_performers": 0, "established_creators": 0}
    candidate_count_top = 0
    candidate_count_established = 0
    analyzed_count_top = 0
    analyzed_count_established = 0

    relax_steps = [
        ("min_views", lambda v: max(100, v // 2), "min_views lowered"),
        ("window_days", lambda v: min(365, int(v * 1.5)), "window_days increased"),
        ("min_age_hours", lambda v: max(12, v // 2), "min_age_hours lowered"),
    ]

    established_candidate_limit = max(limit * 4, 96)
    established_videos, established_creator_count = fetch_established_creator_videos(
        selected_type=selected_type,
        category_key=category_key,
        region=region,
        candidate_limit=established_candidate_limit,
    )

    def score_source(current_params: dict, source_videos: list[dict]) -> tuple[list[dict], int, int, int]:
        candidates = build_candidates(
            source_videos,
            now,
            current_params["window_days"],
            current_params["min_age_hours"],
            current_params["min_views"],
        )
        typed_candidates = [c for c in candidates if matches_winners_type(c, selected_type)]
        candidate_total = len(typed_candidates)
        if not typed_candidates:
            return [], candidate_total, 0, min_quality

        rates = [c["views_per_day"] for c in typed_candidates]
        baseline = median_or_default(rates, default=1.0)
        if baseline <= 0 or len(typed_candidates) < 10:
            broader = [c["views_per_day"] for c in candidates]
            fallback = median_or_default(broader, default=1.0)
            baseline = fallback if fallback > 0 else 1.0

        scored = compute_outlier_scores(typed_candidates, baseline)
        for item in scored:
            item["hot_score"] = item.get("views_per_day", 0.0)

        ranked = sort_winner_candidates(scored, sort_mode=sort_mode, tie_break_quality=False)
        if not quality_enabled:
            return ranked, candidate_total, 0, min_quality

        filtered_sorted, analyzed_total, effective_threshold = apply_quality_filter(
            ranked=ranked,
            selected_type=selected_type,
            limit=limit,
            min_quality=min_quality,
            sort_mode=sort_mode,
        )
        return filtered_sorted, candidate_total, analyzed_total, effective_threshold

    max_pages = 3 if quality_enabled else 1
    for page_count in range(1, max_pages + 1):
        try:
            if selected_type == "shorts":
                videos, source_pages_loaded = fetch_global_short_winner_videos(
                    region,
                    category_id,
                    max_pages=page_count,
                )
            else:
                videos, source_pages_loaded = fetch_global_winner_videos(region, category_id, max_pages=page_count)
        except (HTTPException, YouTubeQuotaExceededError) as exc:
            if is_quota_exceeded_error(exc):
                videos = []
                source_pages_loaded = max(source_pages_loaded, page_count)
                source_relaxations.append("top source quota-limited")
            else:
                raise
        if page_count > 1:
            source_relaxations.append(f"source pages expanded: 1 -> {page_count}")

        params_state = params_seed.copy()
        attempts = 0
        while attempts <= len(relax_steps):
            ranked_top, cand_top, analyzed_top_local, effective_top = score_source(params_state, videos)
            ranked_established, cand_est, analyzed_est_local, effective_est = score_source(
                params_state,
                established_videos,
            )
            top_weight = 0.5
            if quality_enabled:
                # When strict quality floor starves top-performer supply, bias harder to established creators.
                if len(ranked_top) < (limit // 2):
                    top_weight = 0.25
                if len(ranked_top) < (limit // 4):
                    top_weight = 0.1

            scored_items, source_mix = merge_weighted_winners(
                ranked_top,
                ranked_established,
                limit=limit,
                top_weight=top_weight,
            )
            if quality_enabled and len(scored_items) < limit and ranked_established:
                scored_items, source_mix = merge_weighted_winners(
                    ranked_top,
                    ranked_established,
                    limit=limit,
                    top_weight=0.0,
                )

            candidate_count_top = cand_top
            candidate_count_established = cand_est
            analyzed_count_top = analyzed_top_local
            analyzed_count_established = analyzed_est_local
            candidate_count = cand_top + cand_est
            analyzed_count = analyzed_top_local + analyzed_est_local
            if quality_enabled:
                effective_min_quality = min(effective_top, effective_est)

            if len(scored_items) >= limit or attempts == len(relax_steps):
                break

            key, updater, label = relax_steps[attempts]
            old_value = params_state[key]
            new_value = updater(old_value)
            if new_value != old_value:
                params_state[key] = new_value
                relaxations.append(f"{label}: {old_value} -> {new_value}")
            attempts += 1

        if len(scored_items) >= limit:
            break

    results = scored_items[:limit]
    meta = {
        "type": selected_type,
        "sort": sort_mode,
        "quality": quality_enabled,
        "min_quality": min_quality,
        "effective_min_quality": effective_min_quality if quality_enabled else None,
        "quality_relaxed": bool(quality_enabled and effective_min_quality < min_quality),
        "category": category_key,
        "region": region,
        "candidate_count": candidate_count,
        "candidate_count_top": candidate_count_top,
        "candidate_count_established": candidate_count_established,
        "analyzed_count": analyzed_count,
        "analyzed_count_top": analyzed_count_top,
        "analyzed_count_established": analyzed_count_established,
        "source_mix": source_mix,
        "established_creators_considered": established_creator_count,
        "source_pages_loaded": source_pages_loaded,
        "relaxations": relaxations + source_relaxations,
        "quota_limited": any("quota-limited" in str(note).lower() for note in source_relaxations),
    }
    if len(results) < limit:
        meta["message"] = f"Only {len(results)} winners after applying filters"

    resp = {"items": results, "meta": meta}
    cache_set_custom(cache_key, resp, WINNERS_CACHE_TTL_SECONDS)
    return resp


@app.get("/youtube/channel/{channel_identifier}/winners")
def channel_winners(
    request: Request,
    channel_identifier: str,
    type: str | None = Query(default=None),
    format: str | None = None,
    limit: int = 24,
    window_days: int = 180,
    sort: str = "outlier",
    quality: int = 0,
    min_quality: int = 60,
    min_age_hours: int = 48,
    min_views: int = 1000,
    candidate_limit: int = 300,
):
    selected_type = resolve_winners_type(type, format)

    sort_mode = (sort or "outlier").lower()
    if sort_mode not in {"outlier", "hot", "total"}:
        raise HTTPException(status_code=400, detail="sort must be outlier, hot, or total")

    if quality not in {0, 1}:
        raise HTTPException(status_code=400, detail="quality must be 0 or 1")
    quality_enabled = quality == 1

    if limit not in {12, 24, 48}:
        raise HTTPException(status_code=400, detail="limit must be one of: 12, 24, 48")

    window_days = max(1, min(window_days, 365))
    min_quality = max(0, min(min_quality, 100))
    min_age_hours = max(1, min(min_age_hours, 168))
    min_views = max(1, min(min_views, 1000000))
    candidate_limit = max(limit, min(candidate_limit, 500))
    enforce_api_rate_limit(request, scope="channel_winners")

    resolved_id = resolve_channel_id(channel_identifier)

    cache_key = (
        f"winners:{FILTER_CACHE_VERSION}:{resolved_id}:{selected_type}:{sort_mode}:{window_days}:{limit}:"
        f"{quality}:{min_quality}:{min_age_hours}:{min_views}:{candidate_limit}"
    )
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    playlist_id = fetch_uploads_playlist(resolved_id)
    if not playlist_id:
        raise HTTPException(status_code=404, detail="Channel playlist not found")

    video_ids = fetch_playlist_video_ids(playlist_id, candidate_limit)
    if not video_ids:
        resp = {"items": [], "meta": {"message": "No uploaded videos"}}
        cache_set_custom(cache_key, resp, WINNERS_CACHE_TTL_SECONDS)
        return resp

    videos = hydrate_video_metadata(video_ids)
    now = datetime.now(timezone.utc)

    params = {
        "window_days": window_days,
        "min_age_hours": min_age_hours,
        "min_views": min_views,
    }
    relaxations = []
    scored_items = []
    candidate_count = 0
    analyzed_count = 0
    effective_min_quality = min_quality

    relax_steps = [
        ("min_views", lambda v: max(100, v // 2), "min_views lowered"),
        ("window_days", lambda v: min(365, int(v * 1.5)), "window_days increased"),
        ("min_age_hours", lambda v: max(12, v // 2), "min_age_hours lowered"),
    ]

    def score_for_format(current_params):
        candidates = build_candidates(
            videos,
            now,
            current_params["window_days"],
            current_params["min_age_hours"],
            current_params["min_views"],
        )
        typed_candidates = [c for c in candidates if matches_winners_type(c, selected_type)]

        nonlocal candidate_count, analyzed_count, effective_min_quality
        candidate_count = len(typed_candidates)
        analyzed_count = 0
        if not typed_candidates:
            return []

        base_rates = [c["views_per_day"] for c in typed_candidates]
        baseline = median_or_default(base_rates, default=1.0)
        if baseline <= 0 or len(typed_candidates) < 10:
            broader = [c["views_per_day"] for c in candidates]
            fallback = median_or_default(broader, default=1.0)
            baseline = fallback if fallback > 0 else 1.0

        scored = compute_outlier_scores(typed_candidates, baseline)
        for item in scored:
            item["hot_score"] = item.get("views_per_day", 0.0)

        ranked = sort_winner_candidates(scored, sort_mode=sort_mode, tie_break_quality=False)
        if not quality_enabled:
            return ranked

        filtered_sorted, analyzed_count_value, effective_threshold = apply_quality_filter(
            ranked=ranked,
            selected_type=selected_type,
            limit=limit,
            min_quality=min_quality,
            sort_mode=sort_mode,
        )
        analyzed_count = analyzed_count_value
        effective_min_quality = effective_threshold
        return filtered_sorted

    attempts = 0
    while attempts <= len(relax_steps):
        scored_items = score_for_format(params)
        if len(scored_items) >= limit or attempts == len(relax_steps):
            break

        key, updater, desc = relax_steps[attempts]
        old_value = params[key]
        new_value = updater(old_value)
        if new_value != old_value:
            params[key] = new_value
            relaxations.append(f"{desc}: {old_value} -> {new_value}")
        attempts += 1

    results = scored_items[:limit]
    meta = {
        "type": selected_type,
        "sort": sort_mode,
        "quality": quality_enabled,
        "min_quality": min_quality,
        "effective_min_quality": effective_min_quality if quality_enabled else None,
        "quality_relaxed": bool(quality_enabled and effective_min_quality < min_quality),
        "candidate_count": candidate_count,
        "analyzed_count": analyzed_count,
        "relaxations": relaxations,
    }
    if len(results) < limit:
        meta["message"] = f"Only {len(results)} winners after applying filters"

    resp = {"items": results, "meta": meta}
    cache_set_custom(cache_key, resp, WINNERS_CACHE_TTL_SECONDS)
    return resp


@app.get("/youtube/channel/{channel_identifier}/videos")
def channel_videos(
    request: Request,
    channel_identifier: str,
    content_type: str = "all",  # all | shorts | videos
    sort: str = "recent",  # recent | popular
    max_results: int = 24,
):
    max_results = max(1, min(max_results, 50))
    content_type = (content_type or "all").lower()
    if content_type not in {"all", "shorts", "videos"}:
        raise HTTPException(status_code=400, detail="content_type must be one of: all, shorts, videos")
    sort_mode = (sort or "recent").lower()
    if sort_mode not in {"recent", "popular"}:
        raise HTTPException(status_code=400, detail="sort must be one of: recent, popular")

    enforce_api_rate_limit(request, scope="channel_videos")

    resolved_id = resolve_channel_id(channel_identifier)
    cache_key = f"channel-videos:{FILTER_CACHE_VERSION}:{resolved_id}:{content_type}:{sort_mode}:{max_results}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    playlist_id = fetch_uploads_playlist(resolved_id)
    if not playlist_id:
        raise HTTPException(status_code=404, detail="Channel playlist not found")

    candidate_limit = max(50, min(250, max_results * 4))
    video_ids = fetch_playlist_video_ids(playlist_id, candidate_limit)
    if not video_ids:
        resp = {
            "items": [],
            "meta": {
                "channel_id": resolved_id,
                "content_type": content_type,
                "sort": sort_mode,
                "message": "No uploaded videos",
            },
        }
        cache_set_custom(cache_key, resp, PROFILE_GRID_CACHE_TTL_SECONDS)
        return resp

    hydrated = hydrate_video_metadata(video_ids)
    items = build_profile_items_from_videos(hydrated)
    items = sort_profile_items(items, sort_mode)
    typed_items = _filter_profile_items_by_type(items, content_type)

    resp = {
        "items": typed_items[:max_results],
        "meta": {
            "channel_id": resolved_id,
            "content_type": content_type,
            "sort": sort_mode,
            "candidate_count": len(items),
        },
    }
    cache_set_custom(cache_key, resp, PROFILE_GRID_CACHE_TTL_SECONDS)
    return resp
