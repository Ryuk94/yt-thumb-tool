import hashlib
import io
import time
from typing import Any

import requests

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None


THUMB_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
ANALYSIS_MAX_WIDTH = 320

FACE_RATIO_MIN = 0.08
FACE_RATIO_BONUS = 0.18
TEXT_RATIO_MIN = 0.03
CLUTTER_GOOD_MAX = 35
CONTRAST_GOOD_MIN = 45

THUMB_QUALITY_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_FACE_CASCADE = None
_OCR_CHECKED = False
_OCR_READY = False


def _empty_insights() -> dict[str, Any]:
    return {
        "has_face": False,
        "face_area_ratio": 0.0,
        "text_present": False,
        "text_area_ratio": 0.0,
        "ocr_words": 0,
        "aspect_ratio": 0.0,
        "aspect_orientation": "unknown",
        "clutter_score": 0,
        "contrast_score": 0,
        "quality_score": 0,
    }


def _cache_key(thumbnail_url: str) -> str:
    sig = hashlib.sha1(thumbnail_url.encode("utf-8", errors="ignore")).hexdigest()
    return f"thumbsig:{sig}"


def _cache_get(key: str) -> dict[str, Any] | None:
    hit = THUMB_QUALITY_CACHE.get(key)
    if not hit:
        return None
    expires_at, value = hit
    if time.time() > expires_at:
        THUMB_QUALITY_CACHE.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: dict[str, Any]) -> None:
    THUMB_QUALITY_CACHE[key] = (time.time() + THUMB_CACHE_TTL_SECONDS, value)


def _clamp_int(value: int, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, value))


def _download_image(thumbnail_url: str) -> bytes | None:
    try:
        response = requests.get(
            thumbnail_url,
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0"},
        )
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    content_type = (response.headers.get("content-type") or "").lower()
    if content_type and "image" not in content_type:
        return None
    return response.content


def _prepare_rgb_array(image_bytes: bytes):
    if Image is None or np is None:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb = image.convert("RGB")
            if rgb.width > ANALYSIS_MAX_WIDTH:
                ratio = ANALYSIS_MAX_WIDTH / float(rgb.width)
                new_height = max(1, int(round(rgb.height * ratio)))
                resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
                rgb = rgb.resize((ANALYSIS_MAX_WIDTH, new_height), resample=resample)
            return np.array(rgb)
    except Exception:
        return None


def _load_face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE
    if cv2 is None:
        return None
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None
    _FACE_CASCADE = cascade
    return _FACE_CASCADE


def _face_detect(gray_image):
    if cv2 is None:
        return False, 0.0
    cascade = _load_face_cascade()
    if cascade is None:
        return False, 0.0

    faces = cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(24, 24),
    )
    if faces is None or len(faces) == 0:
        return False, 0.0

    image_area = max(int(gray_image.shape[0]) * int(gray_image.shape[1]), 1)
    face_area = 0
    for (_, _, width, height) in faces:
        face_area += int(width) * int(height)
    ratio = min(1.0, float(face_area) / float(image_area))
    return True, ratio


def _is_ocr_available() -> bool:
    global _OCR_CHECKED, _OCR_READY
    if _OCR_CHECKED:
        return _OCR_READY
    _OCR_CHECKED = True

    if pytesseract is None or Image is None:
        _OCR_READY = False
        return _OCR_READY

    try:
        _ = pytesseract.get_tesseract_version()
        _OCR_READY = True
    except Exception:
        _OCR_READY = False
    return _OCR_READY


def _ocr_text(image_rgb):
    if not _is_ocr_available() or pytesseract is None or Image is None:
        return False, 0.0, 0

    try:
        pil_image = Image.fromarray(image_rgb)
        ocr_data = pytesseract.image_to_data(
            pil_image,
            output_type=pytesseract.Output.DICT,
            config="--oem 3 --psm 6",
        )
    except Exception:
        return False, 0.0, 0

    texts = ocr_data.get("text", [])
    confidences = ocr_data.get("conf", [])
    widths = ocr_data.get("width", [])
    heights = ocr_data.get("height", [])
    image_area = max(int(image_rgb.shape[0]) * int(image_rgb.shape[1]), 1)

    ocr_words = 0
    text_area = 0
    for idx, raw_text in enumerate(texts):
        text = (raw_text or "").strip()
        conf_raw = confidences[idx] if idx < len(confidences) else -1
        try:
            confidence = float(conf_raw)
        except Exception:
            confidence = -1.0

        if confidence < 60.0 or len(text) < 2:
            continue

        ocr_words += 1
        width = int(widths[idx]) if idx < len(widths) else 0
        height = int(heights[idx]) if idx < len(heights) else 0
        if width > 0 and height > 0:
            text_area += width * height

    text_area_ratio = min(1.0, float(text_area) / float(image_area))
    return ocr_words > 0, text_area_ratio, ocr_words


def _clutter_score(gray_image) -> int:
    if cv2 is None or np is None:
        return 0
    edges = cv2.Canny(gray_image, 100, 200)
    image_area = max(int(gray_image.shape[0]) * int(gray_image.shape[1]), 1)
    edge_ratio = float(np.count_nonzero(edges)) / float(image_area)
    return _clamp_int(int(edge_ratio * 400))


def _contrast_score(gray_image) -> int:
    if np is None:
        return 0
    p5, p95 = np.percentile(gray_image, [5, 95])
    contrast = (float(p95) - float(p5)) / 255.0
    return _clamp_int(int(contrast * 120))


def compute_quality_score(
    has_face: bool,
    face_area_ratio: float,
    text_present: bool,
    text_area_ratio: float,
    ocr_words: int,
    clutter_score: int,
    contrast_score: int,
) -> int:
    score = 0

    if has_face and face_area_ratio >= FACE_RATIO_MIN:
        score += 30
    if has_face and face_area_ratio >= FACE_RATIO_BONUS:
        score += 15
    if text_present and text_area_ratio >= TEXT_RATIO_MIN:
        score += 25
    if text_present and ocr_words <= 5:
        score += 10
    if clutter_score <= CLUTTER_GOOD_MAX:
        score += 15
    if contrast_score >= CONTRAST_GOOD_MIN:
        score += 10

    return _clamp_int(score)


def analyze_thumbnail(thumbnail_url: str | None) -> dict[str, Any]:
    if not thumbnail_url:
        return _empty_insights()

    key = _cache_key(thumbnail_url)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    insights = _empty_insights()
    image_bytes = _download_image(thumbnail_url)
    if not image_bytes:
        _cache_set(key, insights)
        return insights

    image_rgb = _prepare_rgb_array(image_bytes)
    if image_rgb is None:
        _cache_set(key, insights)
        return insights

    if np is None:
        _cache_set(key, insights)
        return insights

    try:
        if cv2 is not None:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = np.dot(image_rgb[..., :3], [0.299, 0.587, 0.114]).astype("uint8")

        height = int(image_rgb.shape[0])
        width = int(image_rgb.shape[1])
        aspect_ratio = float(width) / float(max(height, 1))
        if aspect_ratio <= 0.8:
            orientation = "portrait"
        elif aspect_ratio >= 1.2:
            orientation = "landscape"
        else:
            orientation = "square"

        has_face, face_area_ratio = _face_detect(gray)
        text_present, text_area_ratio, ocr_words = _ocr_text(image_rgb)
        clutter_score = _clutter_score(gray)
        contrast_score = _contrast_score(gray)
        quality_score = compute_quality_score(
            has_face=has_face,
            face_area_ratio=face_area_ratio,
            text_present=text_present,
            text_area_ratio=text_area_ratio,
            ocr_words=ocr_words,
            clutter_score=clutter_score,
            contrast_score=contrast_score,
        )

        insights = {
            "has_face": bool(has_face),
            "face_area_ratio": round(float(face_area_ratio), 4),
            "text_present": bool(text_present),
            "text_area_ratio": round(float(text_area_ratio), 4),
            "ocr_words": int(ocr_words),
            "aspect_ratio": round(aspect_ratio, 4),
            "aspect_orientation": orientation,
            "clutter_score": int(clutter_score),
            "contrast_score": int(contrast_score),
            "quality_score": int(quality_score),
        }
    except Exception:
        insights = _empty_insights()

    _cache_set(key, insights)
    return insights
