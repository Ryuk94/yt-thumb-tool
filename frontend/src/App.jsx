import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const RAW_API_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  (typeof window !== "undefined" && window.location.hostname === "localhost" ? "http://localhost:8000" : "/api");
const API_BASE = RAW_API_BASE.replace(/\/$/, "");
const ACCENT_RED = "#ff3b3f";
const REGION_OPTIONS = [
  { value: "GLOBAL", label: "Global" },
  { value: "AU", label: "AU" },
  { value: "CA", label: "CA" },
  { value: "GB", label: "GB" },
  { value: "US", label: "US" },
];
const WINNERS_CATEGORIES = [
  { value: "all", label: "All" },
  { value: "gaming", label: "Gaming" },
  { value: "music", label: "Music" },
  { value: "entertainment", label: "Entertainment" },
  { value: "sports", label: "Sports" },
];

function apiUrl(pathWithLeadingSlash) {
  return `${API_BASE}${pathWithLeadingSlash}`;
}

function isQuotaErrorText(value) {
  const lowered = String(value || "").toLowerCase();
  return lowered.includes("quota") || lowered.includes("youtube_quota_exhausted");
}

async function readApiErrorDetail(response, fallback) {
  let detail = fallback;
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string" && payload.detail.trim()) detail = payload.detail;
    if (typeof payload?.error_code === "string" && payload.error_code.trim()) {
      detail = `${detail} (${payload.error_code})`;
    }
  } catch {
    // ignore payload parse errors
  }
  if (isQuotaErrorText(detail)) {
    return "YouTube quota is exhausted right now. Keeping your last successful results visible.";
  }
  return detail;
}

function SubTabButton({ active, children, onClick, darkMode }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: active ? ACCENT_RED : "transparent",
        color: active ? "#fff" : darkMode ? "#fff" : "#111",
        border: `1px solid ${active ? ACCENT_RED : darkMode ? "#fff" : "#111"}`,
        borderRadius: 999,
        padding: "8px 16px",
        fontSize: 14,
        lineHeight: 1,
        cursor: "pointer",
        fontWeight: active ? 700 : 400,
        minWidth: 96,
        minHeight: 36,
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        boxSizing: "border-box",
      }}
    >
      {children}
    </button>
  );
}

function RoundedDropdown({ value, options, onChange, darkMode, minWidth = 96 }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);

  useEffect(() => {
    const onMouseDown = (event) => {
      if (!rootRef.current) return;
      if (!rootRef.current.contains(event.target)) setOpen(false);
    };
    document.addEventListener("mousedown", onMouseDown);
    return () => document.removeEventListener("mousedown", onMouseDown);
  }, []);

  const selected = options.find((o) => o.value === value) || options[0];

  return (
    <div ref={rootRef} style={{ position: "relative", minWidth, width: minWidth, flex: "0 0 auto" }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        style={{
          width: "100%",
          minWidth,
          borderRadius: 999,
          border: `1px solid ${darkMode ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.25)"}`,
          background: darkMode ? "#1c1c1f" : "#fff",
          color: darkMode ? "#fff" : "#111",
          padding: "0 34px 0 14px",
          textAlign: "center",
          cursor: "pointer",
          fontSize: 14,
          lineHeight: 1,
          minHeight: 36,
          height: 36,
          boxSizing: "border-box",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {selected?.label ?? String(selected?.value ?? "")}
      </button>

      <span
        style={{
          position: "absolute",
          right: 12,
          top: "50%",
          transform: "translateY(-50%)",
          pointerEvents: "none",
          color: darkMode ? "#fff" : "#111",
          fontSize: 12,
          fontWeight: 700,
        }}
      >
        {"\u25BE"}
      </span>

      {open && (
        <div
          style={{
            position: "absolute",
            top: "calc(100% + 6px)",
            left: 0,
            right: 0,
            zIndex: 40,
            borderRadius: 18,
            overflow: "hidden",
            border: `1px solid ${darkMode ? "rgba(255,255,255,0.25)" : "rgba(0,0,0,0.2)"}`,
            background: darkMode ? "#26272d" : "#fff",
            boxShadow: "0 12px 24px rgba(0,0,0,0.22)",
          }}
        >
          {options.map((option) => {
            const active = option.value === value;
            return (
              <button
                key={String(option.value)}
                type="button"
                onClick={() => {
                  onChange(option.value);
                  setOpen(false);
                }}
                style={{
                  width: "100%",
                  border: "none",
                  background: active ? ACCENT_RED : "transparent",
                  color: active ? "#fff" : darkMode ? "#fff" : "#111",
                  padding: "8px 10px",
                  cursor: "pointer",
                  fontSize: 14,
                  textAlign: "center",
                }}
              >
                {option.label}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ThumbnailGrid({ items, portraitMode = false }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
        gap: 12,
        marginTop: 16,
      }}
    >
      {items.map((v) => {
        const id = v.video_id || v.id;
        const href = v.url || `https://www.youtube.com/watch?v=${id}`;
        const thumb = v.thumbnail || v.thumbnail_url || "";
        const views = Number(v.views ?? v.view_count ?? 0);
        const channel = v.channelTitle || v.channel_title || "Channel";
        const outlier = v.outlier_score;
        const vpd = Number(v.views_per_day || 0);
        const qualityScore = v.thumb_insights?.quality_score;
        const aspectRatio = v.thumb_insights?.aspect_ratio ?? v.thumbnail_aspect_ratio;

        return (
          <button
            className="thumb-card"
            key={id || href}
            type="button"
            style={{
              textDecoration: "none",
              color: "#111",
              border: "1px solid #ddd",
              borderRadius: 12,
              overflow: "hidden",
              background: "#fff",
              width: "100%",
              padding: 0,
              cursor: "pointer",
              textAlign: "left",
            }}
            onClick={() => {
              window.dispatchEvent(
                new CustomEvent("open-thumbnail-preview", {
                  detail: {
                    ...v,
                    id,
                    href,
                    __portraitMode: portraitMode,
                    thumbnail: thumb,
                    channelTitle: channel,
                    views,
                    outlier_score: outlier,
                    views_per_day: vpd,
                    quality_score: qualityScore,
                    aspect_ratio: aspectRatio,
                  },
                })
              );
            }}
          >
            {portraitMode ? (
              <div style={{ width: "100%", aspectRatio: "9 / 16", background: "#111" }}>
                <img
                  src={thumb}
                  alt={v.title}
                  style={{ width: "100%", height: "100%", display: "block", objectFit: "cover" }}
                />
              </div>
            ) : (
              <img src={thumb} alt={v.title} style={{ width: "100%", display: "block", height: "auto" }} />
            )}
            <div style={{ padding: 10 }}>
              <div style={{ fontWeight: 600, fontSize: 14, lineHeight: 1.2, color: "#111" }}>{v.title}</div>
              <div style={{ opacity: 0.8, fontSize: 12, marginTop: 6, color: "#555" }}>
                {channel} - {views.toLocaleString()} views
              </div>
              {outlier != null && (
                <div style={{ marginTop: 6, fontSize: 11, display: "flex", gap: 6 }}>
                  <span
                    style={{
                      padding: "3px 6px",
                      background: "#232323",
                      color: "#fff",
                      borderRadius: 6,
                      fontWeight: 600,
                    }}
                  >
                    Outlier x{outlier.toFixed(1)}
                  </span>
                  <span style={{ color: "#666" }}>{vpd.toFixed(1)} views/day</span>
                </div>
              )}
              {qualityScore != null && (
                <div style={{ marginTop: 6, fontSize: 11, color: "#444" }}>
                  Quality {qualityScore}
                </div>
              )}
              {aspectRatio != null && aspectRatio > 0 && (
                <div style={{ marginTop: 4, fontSize: 11, color: "#666" }}>
                  AR {Number(aspectRatio).toFixed(2)}
                </div>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}

function LoadMoreButton({ onClick, disabled, style }) {
  return (
    <div style={{ display: "flex", justifyContent: "center", marginTop: 16 }}>
      <button onClick={onClick} disabled={disabled} aria-label="Load more" style={{ ...style, fontSize: 18, lineHeight: 1 }}>
        {"\u25BC"}
      </button>
    </div>
  );
}

function FloatingProgress({ active, message, progress, darkMode }) {
  if (!active) return null;
  return (
    <>
      <div className="floating-progress-backdrop" />
      <div className={`floating-progress ${darkMode ? "floating-progress--dark" : "floating-progress--light"}`}>
        <div className="floating-progress__head">
          <div className="floating-progress__spinner" />
          <div className="floating-progress__titles">
            <div className="floating-progress__label">Processing Request</div>
            <div className="floating-progress__text">{message}</div>
          </div>
          <div className="floating-progress__percent">{Math.max(0, Math.min(100, Math.round(progress)))}%</div>
        </div>
        <div className="floating-progress__track">
          <div className="floating-progress__bar" style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} />
        </div>
      </div>
    </>
  );
}

function formatMetricValue(value, fallback = "-") {
  if (value == null) return fallback;
  if (typeof value === "number") return Number.isFinite(value) ? value.toLocaleString() : fallback;
  if (typeof value === "string" && value.trim()) return value;
  return fallback;
}

function formatDuration(duration) {
  if (duration == null || !Number.isFinite(Number(duration))) return "-";
  const total = Math.max(0, Math.floor(Number(duration)));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function ThumbnailPreviewModal({
  item,
  onClose,
  onDownload,
  onSearchProfile,
  onVisitProfile,
  darkMode,
}) {
  if (!item) return null;
  const numericAr = Number(item.aspect_ratio || item.thumbnail_aspect_ratio || 0);
  const isShort = Boolean(item.__portraitMode) || (numericAr > 0 && numericAr < 1);
  const aspectText = item.aspect_ratio || item.thumbnail_aspect_ratio ? Number(item.aspect_ratio || item.thumbnail_aspect_ratio).toFixed(2) : "-";

  return (
    <div className="thumb-preview-overlay" onClick={onClose}>
      <div className="thumb-preview-shell" onClick={(e) => e.stopPropagation()}>
        <div className={`thumb-preview-frame ${isShort ? "thumb-preview-frame--shorts" : ""}`}>
          <img className="thumb-preview-image" src={item.thumbnail} alt={item.title || "Thumbnail preview"} />
        </div>

        <div className="thumb-preview-actions">
          <button type="button" className="thumb-preview-action-btn" title="Download thumbnail" onClick={onDownload}>
            {"\u2193"}
          </button>
          <button type="button" className="thumb-preview-action-btn" title="Search profile in app" onClick={onSearchProfile}>
            {"\u2315"}
          </button>
          <button type="button" className="thumb-preview-action-btn" title="Open profile on YouTube" onClick={onVisitProfile}>
            {"\u2197"}
          </button>
        </div>

        <aside className={`thumb-preview-panel ${darkMode ? "thumb-preview-panel--dark" : "thumb-preview-panel--light"}`}>
          <h3 className="thumb-preview-title">{item.title || "Untitled video"}</h3>
          <div className="thumb-preview-channel">{item.channelTitle || item.channel_title || "Unknown channel"}</div>

          <div className="thumb-preview-metrics">
            <div><strong>Views:</strong> {formatMetricValue(item.views ?? item.view_count)}</div>
            <div><strong>Duration:</strong> {formatDuration(item.duration ?? item.duration_seconds)}</div>
            <div><strong>Aspect Ratio:</strong> {aspectText}</div>
            <div><strong>Quality:</strong> {formatMetricValue(item.quality_score ?? item.thumb_insights?.quality_score)}</div>
            <div><strong>Outlier:</strong> {item.outlier_score != null ? `x${Number(item.outlier_score).toFixed(2)}` : "-"}</div>
            <div><strong>Views/Day:</strong> {item.views_per_day != null ? Number(item.views_per_day).toFixed(1) : "-"}</div>
            <div><strong>Published:</strong> {formatMetricValue(item.publishedAt || item.published_at)}</div>
            <div><strong>Source:</strong> {formatMetricValue(item.source_group)}</div>
          </div>
        </aside>
      </div>
    </div>
  );
}

function readCookie(name) {
  if (typeof document === "undefined") return null;
  const match = document.cookie.match(new RegExp(`\\b${name}=([^;]+)`));
  return match ? decodeURIComponent(match[1]) : null;
}

function writeCookie(name, value, days = 365) {
  if (typeof document === "undefined") return;
  const expires = new Date(Date.now() + days * 864e5).toUTCString();
  document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/`;
}

export default function App() {
  const [tab, setTab] = useState("trending");

  const [items, setItems] = useState([]);
  const [trendingError, setTrendingError] = useState("");
  const [trendingLoading, setTrendingLoading] = useState(false);
  const [region, setRegion] = useState(() => {
    try {
      const saved = localStorage.getItem("selectedRegion");
      return saved || "GLOBAL";
    } catch {
      return "GLOBAL";
    }
  });
  const [type, setType] = useState("all");
  const [nextTokenTop, setNextTokenTop] = useState(null);
  const [nextTokenDiscover, setNextTokenDiscover] = useState(null);

  const [profileUrl, setProfileUrl] = useState("");
  const [profileType, setProfileType] = useState("all");
  const [profileSort, setProfileSort] = useState("recent");
  const [profileItems, setProfileItems] = useState([]);
  const [profileNextToken, setProfileNextToken] = useState(null);
  const [profileError, setProfileError] = useState("");
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileHasSearched, setProfileHasSearched] = useState(false);
  const [profileSearches, setProfileSearches] = useState(() => {
    try {
      const saved = localStorage.getItem("profileSearches");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [showProfileSuggestions, setShowProfileSuggestions] = useState(false);

  const [winnersFormat, setWinnersFormat] = useState("videos");
  const [winnersCategory, setWinnersCategory] = useState("all");
  const [winnersLimit, setWinnersLimit] = useState(24);
  const [winnersWindowDays, setWinnersWindowDays] = useState(180);
  const [winnersQuality, setWinnersQuality] = useState(false);
  const [winnersMinQuality, setWinnersMinQuality] = useState(60);
  const [winnersItems, setWinnersItems] = useState([]);
  const [winnersLoading, setWinnersLoading] = useState(false);
  const [winnersError, setWinnersError] = useState("");
  const [winnersMeta, setWinnersMeta] = useState(null);

  const [darkMode, setDarkMode] = useState(() => readCookie("darkMode") === "1");
  const [globalLoading, setGlobalLoading] = useState({ active: false, message: "", progress: 0 });
  const loadingTokenRef = useRef(0);
  const [previewItem, setPreviewItem] = useState(null);

  const isShorts = type === "shorts";
  const baseUrl = useMemo(() => (isShorts ? apiUrl("/discover") : apiUrl("/top")), [isShorts]);

  const startGlobalLoading = useCallback((message, progress = 10) => {
    const token = loadingTokenRef.current + 1;
    loadingTokenRef.current = token;
    setGlobalLoading({ active: true, message, progress });
    return token;
  }, []);

  const updateGlobalLoading = useCallback((token, message, progress) => {
    if (loadingTokenRef.current !== token) return;
    setGlobalLoading((prev) => ({
      active: true,
      message: message ?? prev.message,
      progress: progress ?? prev.progress,
    }));
  }, []);

  const endGlobalLoading = useCallback((token) => {
    if (loadingTokenRef.current !== token) return;
    setGlobalLoading((prev) => ({ ...prev, progress: 100 }));
    window.setTimeout(() => {
      if (loadingTokenRef.current !== token) return;
      setGlobalLoading({ active: false, message: "", progress: 0 });
    }, 220);
  }, []);

  const handleTrendingTypeChange = (nextType) => {
    if (nextType === type) return;
    setTrendingLoading(true);
    setType(nextType);
  };

  const handleWinnersFormatChange = (nextFormat) => {
    if (nextFormat === winnersFormat) return;
    setWinnersLoading(true);
    setWinnersFormat(nextFormat);
  };

  const handleProfileTypeChange = (nextType) => {
    if (nextType === profileType) return;
    setProfileLoading(true);
    setProfileItems([]);
    setProfileNextToken(null);
    setProfileType(nextType);
  };

  const handleProfileSortChange = (nextSort) => {
    if (nextSort === profileSort) return;
    setProfileLoading(true);
    setProfileItems([]);
    setProfileNextToken(null);
    setProfileSort(nextSort);
  };

  useEffect(() => {
    document.body.style.backgroundColor = darkMode ? "#1e1f24" : "#f4f5f7";
    document.body.style.color = darkMode ? "#f5f5f5" : "#111";
    writeCookie("darkMode", darkMode ? "1" : "0");
  }, [darkMode]);

  useEffect(() => {
    try {
      localStorage.setItem("selectedRegion", region);
    } catch {
      // ignore
    }
  }, [region]);

  useEffect(() => {
    const onOpenPreview = (event) => {
      setPreviewItem(event.detail || null);
    };
    window.addEventListener("open-thumbnail-preview", onOpenPreview);
    return () => window.removeEventListener("open-thumbnail-preview", onOpenPreview);
  }, []);

  useEffect(() => {
    if (!previewItem) return;
    const onKeyDown = (event) => {
      if (event.key === "Escape") setPreviewItem(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [previewItem]);

  const buildParams = (pageToken = null) => {
    if (isShorts) {
      const p =
        `region=${region}&max_results=24` +
        `&days=7` +
        `&enforce_lang=true` +
        `&strict_shorts=true` +
        `&aspect_filter=true`;
      return pageToken ? `${p}&page_token=${pageToken}` : p;
    }

    const p = `region=${region}&type=${type}&max_results=24`;
    return pageToken ? `${p}&page_token=${pageToken}` : p;
  };

  useEffect(() => {
    const loadingToken = startGlobalLoading(`Loading ${type} feed...`, 12);
    setTrendingLoading(true);
    setTrendingError("");
    setNextTokenTop(null);
    setNextTokenDiscover(null);

    fetch(`${baseUrl}?${buildParams(null)}`)
      .then(async (r) => {
        updateGlobalLoading(loadingToken, "Processing response...", 55);
        if (!r.ok) throw new Error(await readApiErrorDetail(r, "Failed to load trending feed."));
        return r.json();
      })
      .then((d) => {
        updateGlobalLoading(loadingToken, "Rendering thumbnails...", 86);
        setItems(d.items || []);
        if (isShorts) setNextTokenDiscover(d.nextPageToken || null);
        else setNextTokenTop(d.nextPageToken || null);
      })
      .catch((err) => {
        setTrendingError(err.message || "Failed to load trending feed.");
        setNextTokenTop(null);
        setNextTokenDiscover(null);
      })
      .finally(() => {
        setTrendingLoading(false);
        endGlobalLoading(loadingToken);
      });
  }, [baseUrl, region, type, startGlobalLoading, updateGlobalLoading, endGlobalLoading]);

  const canLoadMoreTrending = isShorts ? !!nextTokenDiscover : !!nextTokenTop;

  const loadMoreTrending = () => {
    const token = isShorts ? nextTokenDiscover : nextTokenTop;
    if (!token) return;
    const loadingToken = startGlobalLoading("Loading more thumbnails...", 18);

    fetch(`${baseUrl}?${buildParams(token)}`)
      .then(async (r) => {
        updateGlobalLoading(loadingToken, "Merging new page...", 62);
        if (!r.ok) throw new Error(await readApiErrorDetail(r, "Failed to load more thumbnails."));
        return r.json();
      })
      .then((d) => {
        updateGlobalLoading(loadingToken, "Updating feed...", 88);
        setItems((prev) => [...prev, ...(d.items || [])]);
        if (isShorts) setNextTokenDiscover(d.nextPageToken || null);
        else setNextTokenTop(d.nextPageToken || null);
      })
      .catch((err) => {
        setTrendingError(err.message || "Failed to load more thumbnails.");
      })
      .finally(() => endGlobalLoading(loadingToken));
  };

  const persistProfileSearch = (term) => {
    if (!term) return;
    setProfileSearches((prev) => {
      let updated = [term, ...prev.filter((item) => item !== term)];
      if (updated.length > 5) updated = updated.slice(0, 5);
      try {
        localStorage.setItem("profileSearches", JSON.stringify(updated));
      } catch {
        // ignore
      }
      return updated;
    });
  };

  const fetchProfile = (reset = false, explicitProfileUrl = null) => {
    const cleaned = (explicitProfileUrl ?? profileUrl).trim();
    if (!cleaned) {
      setProfileError("Enter a YouTube profile URL, handle, or channel ID.");
      return;
    }

    const token = reset ? null : profileNextToken;
    if (!reset && !token) return;

    const params = new URLSearchParams({
      profile_url: cleaned,
      content_type: profileType,
      sort: profileSort,
      max_results: "24",
      strict_shorts: "true",
    });
    if (token) params.set("page_token", token);

    if (reset) {
      setProfileItems([]);
      setProfileNextToken(null);
      setProfileHasSearched(true);
    }

    setProfileError("");
    setProfileLoading(true);
    const loadingToken = startGlobalLoading(reset ? "Resolving channel..." : "Loading more profile items...", 12);

    fetch(`${apiUrl("/profile")}?${params.toString()}`)
      .then(async (r) => {
        updateGlobalLoading(loadingToken, "Fetching profile thumbnails...", 54);
        if (!r.ok) throw new Error(await readApiErrorDetail(r, "Failed to load profile feed."));
        return r.json();
      })
      .then((d) => {
        updateGlobalLoading(loadingToken, "Composing profile grid...", 86);
        const channelId = d?.meta?.channel_id || null;
        const next = (d.items || []).map((item) => ({
          ...item,
          channel_id: item.channel_id || channelId,
        }));
        setProfileItems((prev) => (reset ? next : [...prev, ...next]));
        setProfileNextToken(d.nextPageToken || null);
        if (reset) persistProfileSearch(cleaned);
      })
      .catch((err) => {
        setProfileError(err.message || "Failed to load profile feed.");
      })
      .finally(() => {
        setProfileLoading(false);
        endGlobalLoading(loadingToken);
      });
  };

  const handleDownloadPreview = async () => {
    if (!previewItem?.thumbnail) return;
    try {
      const response = await fetch(previewItem.thumbnail);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${previewItem.id || "thumbnail"}.jpg`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch {
      // no-op
    }
  };

  const handleSearchProfileFromPreview = () => {
    if (!previewItem) return;
    const profileQuery = String(previewItem.channel_id || previewItem.channelTitle || previewItem.channel_title || "").trim();
    if (!profileQuery) return;
    setTab("profile");
    setProfileUrl(profileQuery);
    setProfileHasSearched(true);
    setPreviewItem(null);
    fetchProfile(true, profileQuery);
  };

  const handleVisitProfileFromPreview = () => {
    if (!previewItem) return;
    const channelId = String(previewItem.channel_id || "").trim();
    const channelName = String(previewItem.channelTitle || previewItem.channel_title || "").trim();
    const url = channelId
      ? `https://www.youtube.com/channel/${encodeURIComponent(channelId)}`
      : channelName
      ? `https://www.youtube.com/results?search_query=${encodeURIComponent(channelName)}`
      : previewItem.href;
    if (url) window.open(url, "_blank", "noopener,noreferrer");
  };

  useEffect(() => {
    if (!profileHasSearched) return;
    if (!profileUrl.trim()) return;
    if (tab !== "profile") return;
    fetchProfile(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [profileType, profileSort]);

  const requestWinners = useCallback(() => {
    setWinnersLoading(true);
    setWinnersError("");
    const loadingToken = startGlobalLoading("Collecting winners pool...", 12);

    const params = new URLSearchParams({
      format: winnersFormat,
      category: winnersCategory,
      region,
      limit: String(winnersLimit),
      window_days: String(winnersWindowDays),
      sort: "outlier",
      quality: winnersQuality ? "1" : "0",
      min_quality: String(winnersMinQuality),
    });

    fetch(`${apiUrl("/youtube/winners")}?${params.toString()}`)
      .then(async (r) => {
        updateGlobalLoading(loadingToken, "Scoring thumbnail quality...", 62);
        if (!r.ok) throw new Error(await readApiErrorDetail(r, "Failed to load winners."));
        return r.json();
      })
      .then((d) => {
        updateGlobalLoading(loadingToken, "Ranking final winners...", 88);
        setWinnersItems(d.items || []);
        setWinnersMeta(d.meta || null);
      })
      .catch((err) => {
        setWinnersError(err.message || "Failed to load winners.");
      })
      .finally(() => {
        setWinnersLoading(false);
        endGlobalLoading(loadingToken);
      });
  }, [
    winnersFormat,
    winnersCategory,
    region,
    winnersLimit,
    winnersWindowDays,
    winnersQuality,
    winnersMinQuality,
    startGlobalLoading,
    updateGlobalLoading,
    endGlobalLoading,
  ]);

  useEffect(() => {
    if (tab === "winners") requestWinners();
  }, [tab, winnersFormat, winnersCategory, region, winnersLimit, winnersWindowDays, winnersQuality, winnersMinQuality, requestWinners]);

  const onProfileKeyDown = (e) => {
    if (e.key === "Enter") fetchProfile(true);
  };

  const baseButtonStyle = {
    borderRadius: 999,
    border: "1px solid #ddd",
    padding: "8px 16px",
    fontSize: 14,
    cursor: "pointer",
    minWidth: 96,
  };

  const roundedFieldStyle = {
    borderRadius: 999,
    border: `1px solid ${darkMode ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.2)"}`,
    padding: "0 14px",
    background: darkMode ? "#1c1c1f" : "#fff",
    color: darkMode ? "#fff" : "#111",
    minHeight: 36,
    height: 36,
    boxSizing: "border-box",
  };

  const tabButtonStyle = (active) => ({
    ...baseButtonStyle,
    background: active ? ACCENT_RED : "transparent",
    color: active ? "#fff" : darkMode ? "#fff" : "#111",
    borderColor: active ? ACCENT_RED : darkMode ? "#fff" : "#111",
    fontWeight: active ? 700 : 400,
  });

  const actionButtonStyle = (disabled = false) => ({
    ...baseButtonStyle,
    background: disabled ? "#9e9e9e" : ACCENT_RED,
    color: "#fff",
    borderColor: disabled ? "#9e9e9e" : "transparent",
    cursor: disabled ? "not-allowed" : "pointer",
  });

  const infoBoxStyle = {
    background: darkMode ? "#1c1c1f" : "#f5f5f5",
    color: darkMode ? "#ddd" : "#555",
  };

  const searchButtonStyle = {
    borderRadius: 999,
    border: "none",
    padding: "0 16px",
    minHeight: 36,
    height: 36,
    background: ACCENT_RED,
    color: "#fff",
    cursor: "pointer",
  };

  return (
    <div
      style={{
        width: "100%",
        boxSizing: "border-box",
        padding: "16px 20%",
        fontFamily: "system-ui",
      }}
    >
      <FloatingProgress
        active={globalLoading.active}
        message={globalLoading.message}
        progress={globalLoading.progress}
        darkMode={darkMode}
      />
      <ThumbnailPreviewModal
        item={previewItem}
        onClose={() => setPreviewItem(null)}
        onDownload={handleDownloadPreview}
        onSearchProfile={handleSearchProfileFromPreview}
        onVisitProfile={handleVisitProfileFromPreview}
        darkMode={darkMode}
      />
      <div style={{ width: "100%" }}>
        <h1 style={{ marginBottom: 12 }}>YouTube Top Thumbnails</h1>

        <div style={{ display: "flex", gap: 10, marginBottom: 12, justifyContent: "center", flexWrap: "wrap" }}>
          <button onClick={() => setTab("trending")} style={tabButtonStyle(tab === "trending")}>Trending</button>
          <button onClick={() => setTab("profile")} style={tabButtonStyle(tab === "profile")}>Profile</button>
          <button onClick={() => setTab("winners")} style={tabButtonStyle(tab === "winners")}>Winners</button>
        </div>

        {tab === "trending" && (
          <>
            <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
              <RoundedDropdown
                value={region}
                onChange={setRegion}
                darkMode={darkMode}
                minWidth={96}
                options={REGION_OPTIONS}
              />
              <div style={{ display: "flex", gap: 8 }}>
                <SubTabButton active={type === "all"} onClick={() => handleTrendingTypeChange("all")} darkMode={darkMode}>All</SubTabButton>
                <SubTabButton active={type === "shorts"} onClick={() => handleTrendingTypeChange("shorts")} darkMode={darkMode}>Shorts</SubTabButton>
                <SubTabButton active={type === "videos"} onClick={() => handleTrendingTypeChange("videos")} darkMode={darkMode}>Videos</SubTabButton>
              </div>
            </div>

            {trendingError && <div style={{ marginTop: 10, color: "#b00020", fontSize: 13, textAlign: "center" }}>{trendingError}</div>}

            {trendingLoading ? (
              <div style={{ minHeight: 240 }} />
            ) : (
              <>
                <ThumbnailGrid items={items} portraitMode={isShorts} />
                <LoadMoreButton onClick={loadMoreTrending} disabled={!canLoadMoreTrending} style={actionButtonStyle(!canLoadMoreTrending)} />
              </>
            )}
          </>
        )}

        {tab === "profile" && (
          <>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", justifyContent: "center", marginTop: 4 }}>
              <div style={{ flex: "1 1 520px", maxWidth: 720, minWidth: 280, display: "flex", gap: 0 }}>
                <div style={{ position: "relative", width: "100%" }}>
                  <input
                    type="text"
                    placeholder="Channel URL, @handle, or channel ID"
                    value={profileUrl}
                    onChange={(e) => setProfileUrl(e.target.value)}
                    onKeyDown={onProfileKeyDown}
                    onFocus={() => setShowProfileSuggestions(true)}
                    onBlur={() => setTimeout(() => setShowProfileSuggestions(false), 150)}
                    style={{ ...roundedFieldStyle, width: "100%", borderTopRightRadius: 0, borderBottomRightRadius: 0 }}
                  />
                  {showProfileSuggestions && profileSearches.length > 0 && (
                    <div
                      style={{
                        position: "absolute",
                        top: "100%",
                        left: 0,
                        right: 0,
                        marginTop: 4,
                        background: darkMode ? "#1c1c1f" : "#fff",
                        border: `1px solid ${darkMode ? "rgba(255,255,255,0.2)" : "rgba(0,0,0,0.2)"}`,
                        borderRadius: 12,
                        zIndex: 10,
                        boxShadow: "0 12px 24px rgba(0,0,0,0.15)",
                        overflow: "hidden",
                      }}
                    >
                      {profileSearches.map((search) => (
                        <div
                          key={search}
                          onMouseDown={(e) => {
                            e.preventDefault();
                            setProfileUrl(search);
                            setShowProfileSuggestions(false);
                          }}
                          style={{
                            padding: "8px 12px",
                            cursor: "pointer",
                            color: darkMode ? "#fff" : "#111",
                            borderBottom: "1px solid rgba(0,0,0,0.06)",
                          }}
                        >
                          {search}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => fetchProfile(true)}
                  disabled={profileLoading}
                    style={{ ...searchButtonStyle, borderTopLeftRadius: 0, borderBottomLeftRadius: 0, opacity: profileLoading ? 0.6 : 1 }}
                  >
                    Search
                  </button>
                </div>
              </div>

            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center", justifyContent: "center", marginTop: 10 }}>
              <div style={{ display: "flex", gap: 8 }}>
                <SubTabButton active={profileType === "all"} onClick={() => handleProfileTypeChange("all")} darkMode={darkMode}>All</SubTabButton>
                <SubTabButton active={profileType === "shorts"} onClick={() => handleProfileTypeChange("shorts")} darkMode={darkMode}>Shorts</SubTabButton>
                <SubTabButton active={profileType === "videos"} onClick={() => handleProfileTypeChange("videos")} darkMode={darkMode}>Videos</SubTabButton>
              </div>
            </div>

            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center", justifyContent: "center", marginTop: 10 }}>
              <div style={{ display: "flex", gap: 8 }}>
                <SubTabButton active={profileSort === "recent"} onClick={() => handleProfileSortChange("recent")} darkMode={darkMode}>Recent</SubTabButton>
                <SubTabButton active={profileSort === "popular"} onClick={() => handleProfileSortChange("popular")} darkMode={darkMode}>Popular</SubTabButton>
              </div>
            </div>

            {profileError && <div style={{ marginTop: 10, color: "#b00020", fontSize: 13 }}>{profileError}</div>}

            {profileLoading ? (
              <div style={{ minHeight: 240 }} />
            ) : (
              <ThumbnailGrid items={profileItems} portraitMode={profileType === "shorts"} />
            )}
          </>
        )}

        {tab === "winners" && (
          <>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center" }}>
              <SubTabButton active={winnersFormat === "videos"} onClick={() => handleWinnersFormatChange("videos")} darkMode={darkMode}>Videos</SubTabButton>
              <SubTabButton active={winnersFormat === "shorts"} onClick={() => handleWinnersFormatChange("shorts")} darkMode={darkMode}>Shorts</SubTabButton>
            </div>

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10, justifyContent: "center" }}>
              {WINNERS_CATEGORIES.map((category) => (
                <SubTabButton
                  key={category.value}
                  active={winnersCategory === category.value}
                  onClick={() => setWinnersCategory(category.value)}
                  darkMode={darkMode}
                >
                  {category.label}
                </SubTabButton>
              ))}
            </div>

            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginTop: 12, justifyContent: "center" }}>
              <RoundedDropdown
                value={region}
                onChange={setRegion}
                darkMode={darkMode}
                minWidth={96}
                options={REGION_OPTIONS}
              />
              <RoundedDropdown
                value={winnersLimit}
                onChange={setWinnersLimit}
                darkMode={darkMode}
                minWidth={96}
                options={[
                  { value: 12, label: "12" },
                  { value: 24, label: "24" },
                  { value: 48, label: "48" },
                ]}
              />
              <RoundedDropdown
                value={winnersWindowDays}
                onChange={setWinnersWindowDays}
                darkMode={darkMode}
                minWidth={124}
                options={[
                  { value: 30, label: "30 days" },
                  { value: 90, label: "90 days" },
                  { value: 180, label: "180 days" },
                ]}
              />
              <div style={{ display: "flex", gap: 8 }}>
                <SubTabButton active={!winnersQuality} onClick={() => setWinnersQuality(false)} darkMode={darkMode}>Quality Off</SubTabButton>
                <SubTabButton active={winnersQuality} onClick={() => setWinnersQuality(true)} darkMode={darkMode}>Quality On</SubTabButton>
              </div>
              {winnersQuality && (
                <RoundedDropdown
                  value={winnersMinQuality}
                  onChange={setWinnersMinQuality}
                  darkMode={darkMode}
                  minWidth={108}
                  options={[
                    { value: 40, label: "Q >= 40" },
                    { value: 50, label: "Q >= 50" },
                    { value: 60, label: "Q >= 60" },
                    { value: 70, label: "Q >= 70" },
                    { value: 80, label: "Q >= 80" },
                  ]}
                />
              )}
            </div>

            {winnersError && <div style={{ marginTop: 10, color: "#b00020", fontSize: 13 }}>{winnersError}</div>}

            {(winnersMeta?.message || winnersMeta?.candidate_count) && (
              <div
                style={{
                  marginTop: 10,
                  fontSize: 12,
                  padding: "8px 10px",
                  borderRadius: 8,
                  width: "100%",
                  display: "flex",
                  gap: 12,
                  flexWrap: "wrap",
                  justifyContent: "center",
                  textAlign: "center",
                  ...infoBoxStyle,
                }}
              >
                {winnersMeta?.candidate_count != null && <span>Candidates: {winnersMeta.candidate_count}</span>}
                {winnersMeta?.analyzed_count != null && <span>Analyzed: {winnersMeta.analyzed_count}</span>}
                {winnersMeta?.source_mix && (
                  <span>
                    Top: {winnersMeta.source_mix.top_performers || 0} | Established: {winnersMeta.source_mix.established_creators || 0}
                  </span>
                )}
                {winnersMeta?.message && <span>{winnersMeta.message}</span>}
              </div>
            )}

            {winnersLoading ? (
              <div style={{ minHeight: 240 }} />
            ) : (
              <ThumbnailGrid items={winnersItems} portraitMode={winnersFormat === "shorts"} />
            )}
          </>
        )}
      </div>

      <button
        onClick={() => setDarkMode((prev) => !prev)}
        style={{
          position: "fixed",
          bottom: 16,
          left: 16,
          zIndex: 20,
          width: 48,
          height: 48,
          borderRadius: "50%",
          padding: 0,
          background: darkMode ? "#fff" : "#050505",
          color: darkMode ? "#050505" : "#fff",
          border: "none",
          boxShadow: "0 8px 20px rgba(0,0,0,0.25)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 18,
        }}
      >
        {darkMode ? "\u2600\uFE0F" : "\uD83C\uDF19"}
      </button>
    </div>
  );
}


