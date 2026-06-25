"""
TrackFind — Your Personal AI Music Curator
============================================
A premium, dark-themed Streamlit music recommendation app powered by the
Google Gen AI SDK (Gemini).

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this repo (app.py + requirements.txt) to GitHub.
    2. On https://share.streamlit.io, create a new app pointing at app.py.
    3. In the app's "Secrets" settings, add:
           GEMINI_API_KEY = "your-real-key-here"
"""

import os
import re
import csv
import io
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Google Gen AI SDK
# ---------------------------------------------------------------------------
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ===========================================================================
# 1. CONFIGURATION & CONSTANTS
# ===========================================================================

APP_TITLE = "🎵 TrackFind — Your Personal AI Music Curator"

# Use gemini-1.5-flash as requested. If your API key only has access to
# newer model families, swap this string for "gemini-2.5-flash" or similar.
GEMINI_MODEL = "gemini-1.5-flash"

# ---------------------------------------------------------------------------
# 🔑 GEMINI API KEY — CONFIGURATION PLACEHOLDER
# ---------------------------------------------------------------------------
# TrackFind looks for your key in this priority order:
#   1. Streamlit secrets   -> .streamlit/secrets.toml  ->  GEMINI_API_KEY = "..."
#   2. Environment variable -> export GEMINI_API_KEY="..."
#   3. The placeholder string below (NOT recommended for production)
#
# >>> REPLACE THE LINE BELOW WITH YOUR OWN KEY, OR BETTER YET, USE SECRETS <<<
GEMINI_API_KEY_PLACEHOLDER = "YOUR_GEMINI_API_KEY_HERE"


def get_api_key() -> Optional[str]:
    """Resolve the Gemini API key from secrets, env vars, or placeholder."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key

    if GEMINI_API_KEY_PLACEHOLDER and GEMINI_API_KEY_PLACEHOLDER != "YOUR_GEMINI_API_KEY_HERE":
        return GEMINI_API_KEY_PLACEHOLDER

    return None


# ===========================================================================
# 2. PAGE CONFIG
# ===========================================================================

st.set_page_config(
    page_title="TrackFind | AI Music Curator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ===========================================================================
# 3. CUSTOM CSS — DARK / NEON-EMERALD MUSIC THEME
# ===========================================================================

CUSTOM_CSS = """
<style>
    /* ---------- Global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    .stApp {
        background: radial-gradient(circle at 10% 0%, #132022 0%, #0b0f10 45%, #08090a 100%);
        color: #E6F1EE;
    }

    /* ---------- Hero Header ---------- */
    .tf-hero {
        padding: 2.1rem 2.4rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(16,185,129,0.16) 0%, rgba(15,23,23,0.65) 60%);
        border: 1px solid rgba(16,185,129,0.25);
        box-shadow: 0 8px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.03);
        margin-bottom: 1.6rem;
        position: relative;
        overflow: hidden;
    }
    .tf-hero::after {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(16,185,129,0.35), transparent 70%);
        filter: blur(10px);
    }
    .tf-hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #34d399 0%, #a7f3d0 50%, #6ee7b7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .tf-hero p {
        margin: 0.45rem 0 0 0;
        color: #9CB8B0;
        font-size: 0.98rem;
        font-weight: 400;
    }

    /* ---------- Section / Card Containers ---------- */
    .tf-card {
        background: linear-gradient(155deg, rgba(255,255,255,0.035) 0%, rgba(255,255,255,0.012) 100%);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.25);
    }
    .tf-card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.02rem;
        font-weight: 600;
        color: #6EE7B7;
        margin-bottom: 0.65rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 0.82rem;
    }

    /* ---------- Track Row Card ---------- */
    .tf-track {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-left: 3px solid #10B981;
        border-radius: 14px;
        padding: 0.95rem 1.15rem;
        margin-bottom: 0.65rem;
        transition: all 0.15s ease;
    }
    .tf-track:hover {
        background: rgba(16,185,129,0.07);
        border-left-color: #34D399;
        transform: translateX(2px);
    }
    .tf-track-song {
        font-weight: 700;
        font-size: 1.02rem;
        color: #F0FDF9;
        margin: 0;
    }
    .tf-track-artist {
        color: #6EE7B7;
        font-weight: 500;
        font-size: 0.86rem;
        margin: 0.1rem 0 0.4rem 0;
    }
    .tf-track-reason {
        color: #9CB8B0;
        font-size: 0.84rem;
        font-style: italic;
        line-height: 1.35;
        margin: 0;
    }
    .tf-badge {
        display: inline-block;
        background: rgba(16,185,129,0.15);
        color: #6EE7B7;
        border: 1px solid rgba(16,185,129,0.35);
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
    }

    /* ---------- Vault track row ---------- */
    .tf-vault-row {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .tf-vault-row span.tf-vault-title { font-weight: 600; color: #F0FDF9; }
    .tf-vault-row span.tf-vault-artist { color: #6EE7B7; font-size: 0.85rem; }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(16,185,129,0.4) !important;
        background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(16,185,129,0.06)) !important;
        color: #6EE7B7 !important;
        font-weight: 600 !important;
        padding: 0.45rem 1rem !important;
        transition: all 0.15s ease !important;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #10B981, #059669) !important;
        color: #06120D !important;
        border-color: #34D399 !important;
        box-shadow: 0 0 18px rgba(16,185,129,0.35) !important;
    }
    .stButton > button:active { transform: scale(0.98); }

    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #10B981, #047857) !important;
        color: #06120D !important;
        border: none !important;
        font-weight: 700 !important;
        width: 100%;
    }

    /* Primary CTA (Generate Recommendations) */
    .tf-primary-btn .stButton > button {
        background: linear-gradient(135deg, #10B981, #059669) !important;
        color: #06120D !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.7rem 1.2rem !important;
        width: 100%;
        border: none !important;
        box-shadow: 0 0 24px rgba(16,185,129,0.3) !important;
    }

    /* ---------- Inputs ---------- */
    .stTextInput input, .stTextArea textarea {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
        color: #F0FDF9 !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #10B981 !important;
        box-shadow: 0 0 0 1px rgba(16,185,129,0.4) !important;
    }

    /* ---------- Slider ---------- */
    div[data-testid="stSlider"] [role="slider"] {
        background-color: #10B981 !important;
        box-shadow: 0 0 10px rgba(16,185,129,0.6) !important;
    }
    div[data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(90deg, #047857, #10B981) !important;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(255,255,255,0.02);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 10px;
        color: #9CB8B0;
        font-weight: 600;
        font-size: 0.92rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(16,185,129,0.22), rgba(16,185,129,0.08)) !important;
        color: #6EE7B7 !important;
        border: 1px solid rgba(16,185,129,0.3);
    }

    /* ---------- Radio / Segmented ---------- */
    div[role="radiogroup"] label {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.4rem 0.9rem;
        margin-right: 0.4rem;
    }

    /* ---------- Misc text ---------- */
    .tf-subtle { color: #7E978F; font-size: 0.82rem; }
    .tf-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 1.1rem 0;
    }
    .tf-empty-state {
        text-align: center;
        padding: 2.4rem 1rem;
        color: #6B8580;
    }
    .tf-empty-state .tf-emoji { font-size: 2.4rem; display: block; margin-bottom: 0.6rem; }

    /* Metric-like stat chip */
    .tf-stat-chip {
        display: inline-block;
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        color: #6EE7B7;
        font-weight: 700;
        font-size: 1.1rem;
        margin-right: 0.6rem;
    }
    .tf-stat-chip span { display: block; font-size: 0.68rem; color: #9CB8B0; font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; }

    /* Download button styling override */
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(16,185,129,0.06)) !important;
        color: #6EE7B7 !important;
        border: 1px solid rgba(16,185,129,0.4) !important;
        font-weight: 600 !important;
        width: 100%;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background: linear-gradient(135deg, #10B981, #059669) !important;
        color: #06120D !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ===========================================================================
# 4. DATA MODELS
# ===========================================================================

class RecommendedTrack(BaseModel):
    """Strict schema Gemini must follow for every recommendation."""
    Song: str = Field(description="The title of the recommended track.")
    Artist: str = Field(description="The artist name.")
    Reason: str = Field(
        description="A short, clear explanation of WHY this song was chosen."
    )


class RecommendationList(BaseModel):
    recommendations: List[RecommendedTrack]


@dataclass
class SeedTrack:
    artist: str = ""
    title: str = ""
    raw_source: str = ""  # e.g. original YouTube URL, for display/debug


# ===========================================================================
# 5. SESSION STATE INITIALIZATION
# ===========================================================================

def init_session_state():
    defaults = {
        "recommendations": [],          # list of dicts: Song, Artist, Reason
        "playlist_vault": [],           # list of dicts: Song, Artist, Reason
        "now_playing": None,            # dict: {"Song":..., "Artist":...}
        "last_seed": None,              # SeedTrack as dict
        "has_generated": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ===========================================================================
# 6. YOUTUBE LINK PARSING / CLEANING LOGIC
# ===========================================================================

YOUTUBE_ID_PATTERNS = [
    r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/shorts\/)([A-Za-z0-9_-]{11})",
]

# Fluff tokens commonly found in YouTube music video titles
FLUFF_PATTERNS = [
    r"\(\s*official\s*video\s*\)",
    r"\[\s*official\s*video\s*\]",
    r"\(\s*official\s*audio\s*\)",
    r"\[\s*official\s*audio\s*\]",
    r"\(\s*official\s*music\s*video\s*\)",
    r"\[\s*official\s*music\s*video\s*\]",
    r"\(\s*official\s*lyric\s*video\s*\)",
    r"\[\s*official\s*lyric\s*video\s*\]",
    r"\(\s*lyrics?\s*\)",
    r"\[\s*lyrics?\s*\]",
    r"\(\s*lyric\s*video\s*\)",
    r"\[\s*lyric\s*video\s*\]",
    r"\(\s*audio\s*\)",
    r"\[\s*audio\s*\]",
    r"\(\s*visualizer\s*\)",
    r"\[\s*visualizer\s*\]",
    r"\(\s*hd\s*\)",
    r"\[\s*hd\s*\]",
    r"\(\s*4k\s*\)",
    r"\[\s*4k\s*\]",
    r"\(\s*hq\s*\)",
    r"\[\s*hq\s*\]",
    r"\bofficial\s*video\b",
    r"\bofficial\s*audio\b",
    r"\bofficial\s*music\s*video\b",
    r"\bofficial\s*lyric\s*video\b",
    r"\bmusic\s*video\b",
    r"\blyric\s*video\b",
    r"\blyrics\b",
    r"\bvisualizer\b",
    r"\b4k\b",
    r"\bhd\b",
    r"\bhq\b",
    r"\bremastered\b",
    r"\bclean\s*version\b",
    r"\bexplicit\s*version\b",
    r"\(\s*explicit\s*\)",
    r"\[\s*explicit\s*\]",
]

FLUFF_REGEX = re.compile("|".join(FLUFF_PATTERNS), flags=re.IGNORECASE)


def extract_youtube_id(url: str) -> Optional[str]:
    for pattern in YOUTUBE_ID_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_youtube_title(video_id: str) -> Optional[str]:
    """
    Fetch the page <title> via YouTube's lightweight oEmbed endpoint.
    No API key required. Falls back to None on any failure (offline,
    network restrictions, private video, etc.) so the UI can degrade
    gracefully to manual input.
    """
    try:
        import urllib.request
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        with urllib.request.urlopen(oembed_url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("title")
    except Exception:
        return None


def clean_youtube_title(raw_title: str) -> str:
    """
    Strip common YouTube fluff like 'Official Video', 'HD', '4K', 'Lyrics',
    '[Official Audio]' etc., leaving a clean candidate track name.
    """
    if not raw_title:
        return ""

    cleaned = raw_title

    # Remove bracketed/parenthesized fluff and bare fluff keywords
    cleaned = FLUFF_REGEX.sub("", cleaned)

    # Remove any now-empty bracket/paren pairs left behind
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\[\s*\]", "", cleaned)

    # Collapse leftover separator junk (extra dashes, pipes, double spaces)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s*[-|]\s*$", "", cleaned)
    cleaned = re.sub(r"^\s*[-|]\s*", "", cleaned)
    cleaned = cleaned.strip(" -|·•").strip()

    return cleaned


def split_artist_title(cleaned_title: str) -> SeedTrack:
    """
    Attempt to split a cleaned 'Artist - Title' string into components.
    Falls back to treating the whole string as the title if no separator
    is found.
    """
    for sep in [" - ", " – ", " — ", " | "]:
        if sep in cleaned_title:
            parts = cleaned_title.split(sep, 1)
            return SeedTrack(artist=parts[0].strip(), title=parts[1].strip())
    return SeedTrack(artist="", title=cleaned_title.strip())


def parse_youtube_link(url: str) -> SeedTrack:
    video_id = extract_youtube_id(url)
    if not video_id:
        return SeedTrack(raw_source=url)

    raw_title = fetch_youtube_title(video_id)
    if not raw_title:
        return SeedTrack(raw_source=url)

    cleaned = clean_youtube_title(raw_title)
    seed = split_artist_title(cleaned)
    seed.raw_source = url
    return seed


def build_youtube_search_url(query: str) -> str:
    import urllib.parse
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"


def build_youtube_embed_search(artist: str, title: str) -> str:
    """Build a YouTube search URL used to anchor the playback preview."""
    query = f"{artist} {title}".strip() if (artist or title) else ""
    return build_youtube_search_url(query) if query else ""


# ===========================================================================
# 7. GEMINI RECOMMENDATION ENGINE
# ===========================================================================

def get_genai_client() -> Optional[genai.Client]:
    api_key = get_api_key()
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def build_recommendation_prompt(artist: str, title: str, num_recs: int) -> str:
    return f"""
You are TrackFind, an expert AI music curator with encyclopedic knowledge of
songs, artists, genres, eras, moods, and musical structure.

A user has provided this seed track:
    Artist: "{artist or 'Unknown'}"
    Title:  "{title or 'Unknown'}"

Recommend exactly {num_recs} songs that a fan of this track would genuinely
enjoy. Use a healthy mix of reasoning angles across the list — sonic/mood
similarity, shared genre or subgenre, same era, shared collaborators or
influences, similar tempo/instrumentation, or thematic/lyrical similarity.

Rules:
- Do NOT include the seed track itself in the results.
- Do NOT repeat the same song twice.
- Each "Reason" must be ONE short, specific sentence (under 15 words),
  e.g. "Similar dark-pop mood", "Same dynamic tempo and bassline",
  "Iconic late-90s era match", "Collaborated with the same artist".
- Favor real, well-known, verifiable songs and artists.
- Return ONLY the structured data — no preamble, no extra commentary.
""".strip()


def get_recommendations(artist: str, title: str, num_recs: int) -> List[dict]:
    """
    Calls Gemini with a forced JSON schema (Pydantic) so the response is
    guaranteed to match: [{"Song":..., "Artist":..., "Reason":...}, ...]
    """
    client = get_genai_client()
    if client is None:
        raise RuntimeError(
            "Gemini API key not configured. Add GEMINI_API_KEY to your "
            "Streamlit secrets or environment variables."
        )

    prompt = build_recommendation_prompt(artist, title, num_recs)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[RecommendedTrack],
            temperature=0.85,
            max_output_tokens=4096,
        ),
    )

    # The SDK exposes .parsed when response_schema is provided; fall back
    # to manual JSON parsing of .text for resilience across SDK versions.
    parsed = getattr(response, "parsed", None)
    if parsed:
        return [item.model_dump() if hasattr(item, "model_dump") else dict(item) for item in parsed]

    raw_text = response.text or "[]"
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(json)?", "", raw_text).rstrip("`").strip()
    data = json.loads(raw_text)
    return data


# ===========================================================================
# 8. HELPER UI FUNCTIONS
# ===========================================================================

def render_hero():
    st.markdown(
        f"""
        <div class="tf-hero">
            <h1>{APP_TITLE}</h1>
            <p>Discover your next favorite track — powered by Gemini AI, styled for the way you actually listen.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def add_to_playlist(track: dict):
    existing = {(t["Song"].lower(), t["Artist"].lower()) for t in st.session_state.playlist_vault}
    key = (track["Song"].lower(), track["Artist"].lower())
    if key not in existing:
        st.session_state.playlist_vault.append(track)
        st.toast(f"Added '{track['Song']}' to your Playlist Vault ✅", icon="🎶")
    else:
        st.toast(f"'{track['Song']}' is already in your vault.", icon="ℹ️")


def remove_from_playlist(index: int):
    if 0 <= index < len(st.session_state.playlist_vault):
        removed = st.session_state.playlist_vault.pop(index)
        st.toast(f"Removed '{removed['Song']}' from your vault.", icon="🗑️")


def playlist_to_csv_bytes() -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Song", "Artist", "Reason"])
    for track in st.session_state.playlist_vault:
        writer.writerow([track.get("Song", ""), track.get("Artist", ""), track.get("Reason", "")])
    return output.getvalue().encode("utf-8")


def playlist_to_markdown() -> str:
    if not st.session_state.playlist_vault:
        return "_Your playlist vault is empty._"
    lines = ["# 🎵 My TrackFind Playlist", ""]
    for i, track in enumerate(st.session_state.playlist_vault, start=1):
        lines.append(f"{i}. **{track.get('Song','')}** — {track.get('Artist','')}")
    return "\n".join(lines)


# ===========================================================================
# 9. RENDER: HERO HEADER
# ===========================================================================

render_hero()


# ===========================================================================
# 10. TABS
# ===========================================================================

tab_discover, tab_vault = st.tabs(["🔎  Discover & Sync", "🎧  My Playlist Vault"])


# ---------------------------------------------------------------------------
# TAB 1: DISCOVER & SYNC
# ---------------------------------------------------------------------------
with tab_discover:

    col_input, col_player = st.columns([1.35, 1], gap="large")

    # -------------------- LEFT COLUMN: INPUT & CONTROLS --------------------
    with col_input:
        st.markdown('<div class="tf-card">', unsafe_allow_html=True)
        st.markdown('<div class="tf-card-title">🎯 Seed Track Input</div>', unsafe_allow_html=True)

        input_mode = st.radio(
            "Choose input method",
            options=["🎤 Artist + Track", "🔗 YouTube Link"],
            horizontal=True,
            label_visibility="collapsed",
        )

        seed_artist, seed_title = "", ""

        if input_mode == "🎤 Artist + Track":
            c1, c2 = st.columns(2)
            with c1:
                seed_artist = st.text_input("Artist Name", placeholder="e.g. The Weeknd")
            with c2:
                seed_title = st.text_input("Track Title", placeholder="e.g. Blinding Lights")

        else:
            yt_url = st.text_input(
                "YouTube Link",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            if yt_url:
                with st.spinner("Parsing YouTube link..."):
                    seed = parse_youtube_link(yt_url)

                if seed.title:
                    seed_artist, seed_title = seed.artist, seed.title
                    st.success(f"✅ Detected: **{seed_title}**" + (f" — *{seed_artist}*" if seed_artist else ""))
                    if not seed_artist:
                        st.caption("Couldn't separate the artist automatically — feel free to refine below.")
                    seed_artist = st.text_input("Confirm / edit Artist", value=seed_artist, key="yt_artist_confirm")
                    seed_title = st.text_input("Confirm / edit Track Title", value=seed_title, key="yt_title_confirm")
                else:
                    st.warning("⚠️ Couldn't auto-extract a clean title from that link. Please enter details manually.")
                    seed_artist = st.text_input("Artist Name (manual)", key="yt_artist_manual")
                    seed_title = st.text_input("Track Title (manual)", key="yt_title_manual")

        st.markdown("<hr class='tf-divider'>", unsafe_allow_html=True)

        st.markdown('<div class="tf-card-title">🎛️ Recommendation Controls</div>', unsafe_allow_html=True)
        num_recs = st.slider(
            "How many recommendations do you want?",
            min_value=5,
            max_value=50,
            value=10,
            step=1,
            help="Scale from a quick 5-track sample to a full 50-track deep dive.",
        )

        st.markdown(
            f"""<span class="tf-stat-chip">{num_recs}<span>tracks requested</span></span>""",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="tf-primary-btn">', unsafe_allow_html=True)
        generate_clicked = st.button("✨ Generate Recommendations", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close tf-card

        # -------------------- GENERATE LOGIC --------------------
        if generate_clicked:
            if not seed_title and not seed_artist:
                st.error("Please provide at least a track title or artist name before generating recommendations.")
            else:
                try:
                    with st.spinner(f"🎧 Curating {num_recs} tracks with Gemini AI..."):
                        results = get_recommendations(seed_artist, seed_title, num_recs)
                    st.session_state.recommendations = results
                    st.session_state.has_generated = True
                    st.session_state.last_seed = {"artist": seed_artist, "title": seed_title}
                    st.session_state.now_playing = {"Song": seed_title, "Artist": seed_artist}
                    st.success(f"🎉 Generated {len(results)} recommendations based on '{seed_title or seed_artist}'!")
                except RuntimeError as e:
                    st.error(f"⚠️ {e}")
                except Exception as e:
                    st.error(f"⚠️ Something went wrong while contacting Gemini: {e}")

    # -------------------- RIGHT COLUMN: PLAYBACK PREVIEW --------------------
    with col_player:
        st.markdown('<div class="tf-card">', unsafe_allow_html=True)
        st.markdown('<div class="tf-card-title">▶️ Now Sampling</div>', unsafe_allow_html=True)

        now_playing = st.session_state.now_playing

        if now_playing and (now_playing.get("Song") or now_playing.get("Artist")):
            song = now_playing.get("Song", "")
            artist = now_playing.get("Artist", "")
            st.markdown(f"**{song}**")
            st.markdown(f"<span class='tf-subtle'>{artist}</span>", unsafe_allow_html=True)

            search_url = build_youtube_embed_search(artist, song)
            if search_url:
                st.video(search_url)
            st.caption("🔊 Preview anchored via YouTube search — click through to play the exact track.")
        else:
            st.markdown(
                """
                <div class="tf-empty-state">
                    <span class="tf-emoji">🎶</span>
                    Generate recommendations or pick a track below to preview it here.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------- RECOMMENDATIONS LIST --------------------
    st.markdown("<hr class='tf-divider'>", unsafe_allow_html=True)

    if st.session_state.recommendations:
        st.markdown('<div class="tf-card-title" style="font-size:1rem;">🪄 Curated For You</div>', unsafe_allow_html=True)

        for idx, track in enumerate(st.session_state.recommendations):
            song = track.get("Song", "Unknown Track")
            artist = track.get("Artist", "Unknown Artist")
            reason = track.get("Reason", "")

            row = st.container()
            with row:
                c_info, c_play, c_add = st.columns([5, 1, 1.3])
                with c_info:
                    st.markdown(
                        f"""
                        <div class="tf-track">
                            <span class="tf-badge">#{idx + 1}</span>
                            <p class="tf-track-song">{song}</p>
                            <p class="tf-track-artist">{artist}</p>
                            <p class="tf-track-reason">💡 {reason}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c_play:
                    if st.button("▶️ Play", key=f"play_{idx}"):
                        st.session_state.now_playing = {"Song": song, "Artist": artist}
                        st.rerun()
                with c_add:
                    if st.button("➕ Add to Playlist", key=f"add_{idx}"):
                        add_to_playlist({"Song": song, "Artist": artist, "Reason": reason})

    elif st.session_state.has_generated:
        st.info("No recommendations were returned. Try a different seed track.")
    else:
        st.markdown(
            """
            <div class="tf-empty-state">
                <span class="tf-emoji">🧭</span>
                Your recommendations will appear here once you generate them above.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# TAB 2: MY PLAYLIST VAULT
# ---------------------------------------------------------------------------
with tab_vault:
    vault = st.session_state.playlist_vault

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            f"""<span class="tf-stat-chip">{len(vault)}<span>saved tracks</span></span>""",
            unsafe_allow_html=True,
        )
    with col_b:
        unique_artists = len({t["Artist"] for t in vault}) if vault else 0
        st.markdown(
            f"""<span class="tf-stat-chip">{unique_artists}<span>unique artists</span></span>""",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='tf-divider'>", unsafe_allow_html=True)

    col_list, col_export = st.columns([1.4, 1], gap="large")

    # -------------------- SAVED TRACKS LIST --------------------
    with col_list:
        st.markdown('<div class="tf-card">', unsafe_allow_html=True)
        st.markdown('<div class="tf-card-title">💾 Saved Tracks</div>', unsafe_allow_html=True)

        if not vault:
            st.markdown(
                """
                <div class="tf-empty-state">
                    <span class="tf-emoji">📭</span>
                    Your vault is empty. Head to "Discover &amp; Sync" and add some tracks!
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for i, track in enumerate(vault):
                r1, r2, r3 = st.columns([4.2, 0.9, 0.9])
                with r1:
                    st.markdown(
                        f"""
                        <div class="tf-vault-row">
                            <div>
                                <span class="tf-vault-title">{track.get('Song','')}</span><br>
                                <span class="tf-vault-artist">{track.get('Artist','')}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with r2:
                    if st.button("▶️", key=f"vault_play_{i}", help="Preview this track"):
                        st.session_state.now_playing = {"Song": track.get("Song", ""), "Artist": track.get("Artist", "")}
                        st.toast("Switched preview — check the Discover tab. 🎧")
                with r3:
                    if st.button("🗑️", key=f"vault_remove_{i}", help="Remove from vault"):
                        remove_from_playlist(i)
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------- EXPORT TOOLS --------------------
    with col_export:
        st.markdown('<div class="tf-card">', unsafe_allow_html=True)
        st.markdown('<div class="tf-card-title">📤 Export &amp; Manage</div>', unsafe_allow_html=True)

        if vault:
            csv_bytes = playlist_to_csv_bytes()
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_bytes,
                file_name="trackfind_playlist.csv",
                mime="text/csv",
                use_container_width=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Copy-paste tracklist:")
            st.text_area(
                "Markdown tracklist",
                value=playlist_to_markdown(),
                height=220,
                label_visibility="collapsed",
            )

            st.markdown("<br>", unsafe_allow_html=True)
            confirm_clear = st.checkbox("Confirm: I want to clear my playlist")
            if st.button("🧹 Clear Playlist", use_container_width=True, disabled=not confirm_clear):
                st.session_state.playlist_vault = []
                st.toast("Playlist cleared. Fresh start! 🌱")
                st.rerun()
        else:
            st.caption("Add tracks to your vault to unlock export options.")

        st.markdown('</div>', unsafe_allow_html=True)


# ===========================================================================
# 11. FOOTER
# ===========================================================================
st.markdown(
    """
    <div style="text-align:center; padding: 1.5rem 0 0.5rem 0; color:#5C7A73; font-size:0.78rem;">
        Built with Streamlit &amp; Google Gemini · TrackFind © 2026
    </div>
    """,
    unsafe_allow_html=True,
)
