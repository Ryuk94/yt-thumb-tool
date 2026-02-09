from datetime import datetime, timedelta, timezone

import backend.main as main_module
from starlette.requests import Request
from backend.app.services.thumb_quality import compute_quality_score
from backend.main import (
    CACHE,
    build_candidates,
    channel_winners,
    compute_outlier_scores,
    iso8601_duration_to_seconds,
    matches_winners_type,
    median_or_default,
)


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


def make_video(video_id, days_ago, views, duration):
    published_at = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")
    return {
        "id": video_id,
        "snippet": {
            "title": f"Video {video_id}",
            "publishedAt": published_at,
            "thumbnails": {
                "high": {"url": f"https://img/{video_id}.jpg"},
            },
        },
        "statistics": {"viewCount": str(views)},
        "contentDetails": {"duration": duration},
    }


def test_iso8601_duration_to_seconds():
    assert iso8601_duration_to_seconds("PT1H2M3S") == 3723
    assert iso8601_duration_to_seconds("PT45S") == 45
    assert iso8601_duration_to_seconds("PT5M") == 300


def test_is_short_classification():
    now = datetime.now(timezone.utc)
    videos = [
        make_video("shorty", 5, 2000, "PT45S"),
        make_video("longy", 5, 2000, "PT2M5S"),
    ]
    candidates = build_candidates(videos, now, window_days=30, min_age_hours=24, min_views=1)
    assert any(v["is_short"] for v in candidates if v["video_id"] == "shorty")
    assert any(not v["is_short"] for v in candidates if v["video_id"] == "longy")


def test_outlier_scores_sorting():
    now = datetime.now(timezone.utc)
    fixture = [
        make_video("a", 10, 500, "PT3M"),
        make_video("b", 10, 1000, "PT3M"),
        make_video("c", 10, 2000, "PT3M"),
        make_video("d", 10, 4000, "PT3M"),
        make_video("e", 10, 50, "PT3M"),
    ]
    candidates = build_candidates(fixture, now, window_days=30, min_age_hours=24, min_views=1)
    base_rates = [c["views_per_day"] for c in candidates]
    baseline = median_or_default(base_rates)
    scored = compute_outlier_scores(candidates, baseline)
    assert scored[0]["views_per_day"] == max(base_rates)
    assert scored[0]["outlier_score"] > scored[-1]["outlier_score"]
    assert scored[0]["url"].endswith(scored[0]["video_id"])


def test_quality_score_heuristic():
    strong = compute_quality_score(
        has_face=True,
        face_area_ratio=0.2,
        text_present=True,
        text_area_ratio=0.05,
        ocr_words=3,
        clutter_score=20,
        contrast_score=60,
    )
    weak = compute_quality_score(
        has_face=False,
        face_area_ratio=0.0,
        text_present=False,
        text_area_ratio=0.0,
        ocr_words=0,
        clutter_score=80,
        contrast_score=10,
    )
    assert strong > weak
    assert 0 <= strong <= 100
    assert 0 <= weak <= 100


def test_matches_winners_type_prefers_aspect_ratio():
    original_confirm = main_module.is_confirmed_short
    main_module.is_confirmed_short = lambda video_id: video_id == "confirmed_short"

    short_like = {"is_short": False, "thumbnail_aspect_ratio": 0.56}
    video_like = {"is_short": True, "thumbnail_aspect_ratio": 1.78, "video_id": "video_like"}
    confirmed_short = {"is_short": False, "thumbnail_aspect_ratio": 1.78, "video_id": "confirmed_short"}
    unknown_ratio = {"is_short": False, "video_id": "video_like"}

    try:
        assert matches_winners_type(short_like, "shorts")
        assert not matches_winners_type(short_like, "videos")
        assert not matches_winners_type(video_like, "videos")
        assert matches_winners_type(video_like, "shorts")
        assert matches_winners_type(confirmed_short, "shorts")
        assert not matches_winners_type(confirmed_short, "videos")
        assert not matches_winners_type(unknown_ratio, "shorts")
    finally:
        main_module.is_confirmed_short = original_confirm


def test_channel_winners_quality_filter(monkeypatch):
    CACHE.clear()
    request = make_request()

    fake_videos = [
        make_video("a", 10, 5000, "PT3M"),
        make_video("b", 10, 4500, "PT50S"),
        make_video("c", 10, 4000, "PT3M"),
        make_video("d", 10, 2000, "PT3M"),
    ]

    monkeypatch.setattr(main_module, "resolve_channel_id", lambda _: "UC_TEST")
    monkeypatch.setattr(main_module, "fetch_uploads_playlist", lambda _: "PL_TEST")
    monkeypatch.setattr(main_module, "fetch_playlist_video_ids", lambda _a, _b: ["a", "b", "c", "d"])
    monkeypatch.setattr(main_module, "hydrate_video_metadata", lambda _ids: fake_videos)

    analyze_calls = {"count": 0}

    def fake_analyze(url: str):
        analyze_calls["count"] += 1
        if "a.jpg" in url:
            quality = 90
        elif "b.jpg" in url:
            quality = 40
        elif "c.jpg" in url:
            quality = 72
        else:
            quality = 55
        return {
            "has_face": False,
            "face_area_ratio": 0.0,
            "text_present": False,
            "text_area_ratio": 0.0,
            "ocr_words": 0,
            "clutter_score": 20,
            "contrast_score": 50,
            "quality_score": quality,
        }

    monkeypatch.setattr(main_module, "analyze_thumbnail", fake_analyze)

    without_quality = channel_winners(
        request=request,
        channel_identifier="UC_TEST",
        type="all",
        sort="hot",
        quality=0,
        min_quality=60,
        limit=12,
        window_days=180,
        min_age_hours=1,
        min_views=1,
        candidate_limit=50,
    )

    with_quality = channel_winners(
        request=request,
        channel_identifier="UC_TEST",
        type="all",
        sort="hot",
        quality=1,
        min_quality=60,
        limit=12,
        window_days=180,
        min_age_hours=1,
        min_views=1,
        candidate_limit=50,
    )

    assert len(with_quality["items"]) <= len(without_quality["items"])
    assert all("thumb_insights" not in item for item in without_quality["items"])
    assert all("thumb_insights" in item for item in with_quality["items"])
    effective_min = int(with_quality["meta"].get("effective_min_quality") or 0)
    assert effective_min == 60
    assert with_quality["meta"].get("quality_relaxed") is False
    assert all(item["thumb_insights"]["quality_score"] >= effective_min for item in with_quality["items"])
    assert analyze_calls["count"] <= 24


def test_global_winners_quality_filter(monkeypatch):
    CACHE.clear()
    request = make_request()

    fake_videos = [
        make_video("ga", 8, 6000, "PT3M"),
        make_video("gb", 8, 5500, "PT55S"),
        make_video("gc", 8, 2500, "PT3M"),
        make_video("gd", 8, 2200, "PT3M"),
    ]

    def fake_analyze(url: str):
        if "ga.jpg" in url:
            quality = 85
        elif "gb.jpg" in url:
            quality = 45
        elif "gc.jpg" in url:
            quality = 70
        else:
            quality = 50
        return {
            "has_face": False,
            "face_area_ratio": 0.0,
            "text_present": False,
            "text_area_ratio": 0.0,
            "ocr_words": 0,
            "clutter_score": 22,
            "contrast_score": 50,
            "quality_score": quality,
        }

    monkeypatch.setattr(main_module, "fetch_global_winner_videos", lambda *args, **kwargs: (fake_videos, 1))
    monkeypatch.setattr(main_module, "fetch_established_creator_videos", lambda *args, **kwargs: (fake_videos, 6))
    monkeypatch.setattr(main_module, "analyze_thumbnail", fake_analyze)

    without_quality = main_module.winners(
        request=request,
        format="all",
        category="all",
        region="US",
        limit=12,
        window_days=180,
        quality=0,
        min_quality=60,
        min_age_hours=1,
        min_views=1,
    )
    with_quality = main_module.winners(
        request=request,
        format="all",
        category="all",
        region="US",
        limit=12,
        window_days=180,
        quality=1,
        min_quality=60,
        min_age_hours=1,
        min_views=1,
    )

    assert len(with_quality["items"]) <= len(without_quality["items"])
    assert all("thumb_insights" not in item for item in without_quality["items"])
    assert all("thumb_insights" in item for item in with_quality["items"])
    effective_min = int(with_quality["meta"].get("effective_min_quality") or 0)
    assert effective_min == 60
    assert with_quality["meta"].get("quality_relaxed") is False
    assert all(item["thumb_insights"]["quality_score"] >= effective_min for item in with_quality["items"])


def test_resolve_endpoint(monkeypatch):
    monkeypatch.setattr(main_module, "resolve_channel_id_safely_cached", lambda _query: "UC_RESOLVED")
    payload = main_module.resolve_channel(main_module.ResolveRequest(query="@creator"), make_request())
    assert payload["channel_id"] == "UC_RESOLVED"


def test_patterns_extract_save_apply(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "analyze_thumbnail",
        lambda _url: {
            "has_face": True,
            "text_present": False,
            "aspect_orientation": "portrait",
            "quality_score": 82,
            "clutter_score": 20,
        },
    )

    extract_response = main_module.extract_patterns(
        main_module.PatternExtractRequest(
            items=[
                main_module.PatternExtractItem(thumbnail_url="https://img/a.jpg", video_id="a"),
                main_module.PatternExtractItem(thumbnail_url="https://img/b.jpg", video_id="b"),
            ]
        ),
        make_request(),
    )
    extracted = extract_response
    assert extracted["meta"]["input_count"] == 2
    assert extracted["meta"]["cluster_count"] >= 1
    assert extracted["items"][0]["cluster_id"].startswith("cluster_")

    save_response = main_module.save_pattern(
        main_module.PatternSaveRequest(
            name="portrait-face",
            clusters=extracted["clusters"],
            filters={"min_quality": 60},
        ),
        make_request(),
    )
    pattern_id = save_response["pattern_id"]

    applied = main_module.apply_pattern(pattern_id, make_request())
    assert applied["pattern"]["name"] == "portrait-face"
