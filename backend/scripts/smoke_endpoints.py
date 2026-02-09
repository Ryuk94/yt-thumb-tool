from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from starlette.requests import Request

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import backend.main as main_module


def make_request(ip: str = "127.0.0.1") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "client": (ip, 8000),
            "query_string": b"",
            "server": ("test", 80),
            "scheme": "http",
            "http_version": "1.1",
        }
    )


def make_video(video_id: str, days_ago: int, views: int, duration: str) -> dict:
    published_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")
    return {
        "id": video_id,
        "snippet": {
            "title": f"Video {video_id}",
            "channelTitle": "Smoke Channel",
            "publishedAt": published_at,
            "thumbnails": {
                "high": {"url": f"https://img/{video_id}.jpg", "width": 1280, "height": 720},
            },
            "defaultAudioLanguage": "en",
            "defaultLanguage": "en",
        },
        "statistics": {"viewCount": str(views)},
        "contentDetails": {"duration": duration},
    }


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def reset_state() -> None:
    main_module.CACHE.clear()
    main_module.API_RATE_LIMIT_BUCKETS.clear()
    main_module.PROFILE_RATE_LIMIT_BUCKETS.clear()
    main_module.SHORTS_DETECT_CACHE.clear()


def test_health() -> None:
    payload = main_module.health()
    assert_true(payload.get("ok") is True, "/health should return ok=true")


def test_top_cache() -> None:
    reset_state()
    request = make_request()
    call_count = {"youtube_api_get": 0}

    def fake_youtube_api_get(url: str, params: dict, timeout: int = 15) -> dict:
        _ = (url, params, timeout)
        call_count["youtube_api_get"] += 1
        return {
            "items": [make_video("top1", 3, 25000, "PT2M5S")],
            "nextPageToken": "NEXT_TOKEN",
        }

    with patch.object(main_module, "youtube_api_get", side_effect=fake_youtube_api_get):
        payload_1 = main_module.top(request, region="US", max_results=12, type="all")
        payload_2 = main_module.top(request, region="US", max_results=12, type="all")

    assert_true(payload_1 == payload_2, "/top cached response should be identical")
    assert_true(call_count["youtube_api_get"] == 1, "/top should hit source once then cache")


def test_discover_cache() -> None:
    reset_state()
    request = make_request()
    call_count = {"short_feed": 0}

    def fake_fetch_short_chart_videos(region: str, video_category_id: str | None, max_pages: int = 2):
        _ = (region, video_category_id, max_pages)
        call_count["short_feed"] += 1
        return ([make_video("short1", 2, 15000, "PT45S")], 1)

    with (
        patch.object(main_module, "fetch_short_chart_videos", side_effect=fake_fetch_short_chart_videos),
        patch.object(main_module, "analyze_thumbnail", return_value={"aspect_ratio": 0.56}),
        patch.object(main_module, "is_confirmed_short", return_value=False),
    ):
        payload_1 = main_module.discover(request, region="US", max_results=12)
        payload_2 = main_module.discover(request, region="US", max_results=12)

    assert_true(len(payload_1.get("items", [])) > 0, "/discover shorts fallback should keep <=60s items")
    assert_true(payload_1 == payload_2, "/discover cached response should be identical")
    assert_true(call_count["short_feed"] == 1, "/discover should hit source once then cache")


def test_profile_api_cache() -> None:
    reset_state()
    request = make_request()
    call_count = {"tier_a": 0}

    def fake_tier_a(channel_id: str, limit: int, sort_mode: str):
        _ = (channel_id, limit, sort_mode)
        call_count["tier_a"] += 1
        return [
            {
                "id": "prof_api_1",
                "title": "API profile video",
                "channelTitle": "Smoke Channel",
                "publishedAt": "2025-01-01T00:00:00Z",
                "views": 1234,
                "duration": 80,
                "thumbnail": "https://img/prof_api_1.jpg",
                "thumbWidth": 1280,
                "thumbHeight": 720,
            }
        ]

    with (
        patch.object(main_module, "resolve_channel_id_safely_cached", return_value="UC_PROFILE_API"),
        patch.object(main_module, "fetch_profile_grid_tier_a", side_effect=fake_tier_a),
    ):
        payload_1 = main_module.profile(request, profile_url="https://youtube.com/@smoke", max_results=24)
        payload_2 = main_module.profile(request, profile_url="https://youtube.com/@smoke", max_results=24)

    assert_true(call_count["tier_a"] == 1, "/profile api mode should hit tier A once then cache")
    assert_true(payload_1.get("meta", {}).get("source") == "api", "/profile api mode should report source=api")
    assert_true(payload_1 == payload_2, "/profile api mode cached response should be identical")


def test_profile_fallback_shape_and_cache() -> None:
    reset_state()
    request = make_request()
    call_count = {"tier_a": 0, "tier_b": 0}

    def fake_tier_a(channel_id: str, limit: int, sort_mode: str):
        _ = (channel_id, limit, sort_mode)
        call_count["tier_a"] += 1
        raise main_module.YouTubeQuotaExceededError("quota exceeded")

    def fake_tier_b(channel_id: str, limit: int, sort_mode: str):
        _ = (channel_id, limit, sort_mode)
        call_count["tier_b"] += 1
        return [
            {
                "id": "fallback1",
                "title": "YouTube video",
                "channelTitle": None,
                "publishedAt": None,
                "views": None,
                "duration": None,
                "thumbnail": "https://i.ytimg.com/vi/fallback1/hqdefault.jpg",
                "thumbWidth": None,
                "thumbHeight": None,
            }
        ]

    with (
        patch.object(main_module, "resolve_channel_id_safely_cached", return_value="UC_PROFILE_FALLBACK"),
        patch.object(main_module, "fetch_profile_grid_tier_a", side_effect=fake_tier_a),
        patch.object(main_module, "fetch_profile_grid_tier_b", side_effect=fake_tier_b),
    ):
        payload_1 = main_module.profile(request, profile_url="UC_PROFILE_FALLBACK", max_results=24)
        payload_2 = main_module.profile(request, profile_url="UC_PROFILE_FALLBACK", max_results=24)

    assert_true(payload_1.get("meta", {}).get("source") == "fallback", "/profile fallback should report source=fallback")
    assert_true(call_count["tier_a"] == 1, "/profile fallback should try tier A once")
    assert_true(call_count["tier_b"] == 1, "/profile fallback should hit tier B once then cache")
    assert_true(payload_1 == payload_2, "/profile fallback cached response should be identical")

    first_item = payload_1.get("items", [{}])[0]
    required_keys = {
        "id",
        "title",
        "channelTitle",
        "publishedAt",
        "views",
        "duration",
        "thumbnail",
        "thumbWidth",
        "thumbHeight",
    }
    assert_true(required_keys.issubset(set(first_item.keys())), "/profile fallback item shape is missing keys")


def test_global_winners_cache() -> None:
    reset_state()
    request = make_request()
    call_count = {"global_source": 0, "established_source": 0}
    fake_videos = [
        make_video("gw1", 6, 24000, "PT3M"),
        make_video("gw2", 7, 18000, "PT52S"),
        make_video("gw3", 8, 14000, "PT2M"),
    ]

    def fake_global(region: str, category_id: str | None, max_pages: int = 1):
        _ = (region, category_id, max_pages)
        call_count["global_source"] += 1
        return (fake_videos, 1)

    def fake_established(selected_type: str, category_key: str, region: str, candidate_limit: int):
        _ = (selected_type, category_key, region, candidate_limit)
        call_count["established_source"] += 1
        return (fake_videos, 5)

    with (
        patch.object(main_module, "fetch_global_winner_videos", side_effect=fake_global),
        patch.object(main_module, "fetch_established_creator_videos", side_effect=fake_established),
    ):
        payload_1 = main_module.winners(
            request,
            format="all",
            region="US",
            limit=12,
            min_age_hours=1,
            min_views=1,
        )
        payload_2 = main_module.winners(
            request,
            format="all",
            region="US",
            limit=12,
            min_age_hours=1,
            min_views=1,
        )

    assert_true(call_count["global_source"] == 1, "/youtube/winners should hit global source once then cache")
    assert_true(call_count["established_source"] == 1, "/youtube/winners should hit established source once then cache")
    assert_true(payload_1 == payload_2, "/youtube/winners cached response should be identical")


def test_channel_winners_cache() -> None:
    reset_state()
    request = make_request()
    call_count = {"hydrate": 0}
    fake_videos = [
        make_video("cw1", 6, 30000, "PT3M"),
        make_video("cw2", 9, 19000, "PT58S"),
        make_video("cw3", 10, 16000, "PT4M"),
    ]

    def fake_hydrate(video_ids: list[str]):
        _ = video_ids
        call_count["hydrate"] += 1
        return fake_videos

    with (
        patch.object(main_module, "resolve_channel_id", return_value="UC_CHANNEL_WINNERS"),
        patch.object(main_module, "fetch_uploads_playlist", return_value="PL_CHANNEL_WINNERS"),
        patch.object(main_module, "fetch_playlist_video_ids", return_value=["cw1", "cw2", "cw3"]),
        patch.object(main_module, "hydrate_video_metadata", side_effect=fake_hydrate),
    ):
        payload_1 = main_module.channel_winners(
            request,
            channel_identifier="UC_CHANNEL_WINNERS",
            format="all",
            limit=12,
            min_age_hours=1,
            min_views=1,
            candidate_limit=50,
        )
        payload_2 = main_module.channel_winners(
            request,
            channel_identifier="UC_CHANNEL_WINNERS",
            format="all",
            limit=12,
            min_age_hours=1,
            min_views=1,
            candidate_limit=50,
        )

    assert_true(call_count["hydrate"] == 1, "/youtube/channel/{id}/winners should hydrate once then cache")
    assert_true(payload_1 == payload_2, "/youtube/channel/{id}/winners cached response should be identical")


def run() -> int:
    checks = [
        ("health", test_health),
        ("top cache", test_top_cache),
        ("discover cache", test_discover_cache),
        ("profile api cache", test_profile_api_cache),
        ("profile fallback shape + cache", test_profile_fallback_shape_and_cache),
        ("global winners cache", test_global_winners_cache),
        ("channel winners cache", test_channel_winners_cache),
    ]
    failures = []

    for check_name, check_fn in checks:
        try:
            check_fn()
            print(f"[PASS] {check_name}")
        except Exception as exc:  # pragma: no cover - smoke script output path
            failures.append((check_name, str(exc)))
            print(f"[FAIL] {check_name}: {exc}")

    if failures:
        print(f"\nSmoke checks failed: {len(failures)}")
        for check_name, message in failures:
            print(f"- {check_name}: {message}")
        return 1

    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
