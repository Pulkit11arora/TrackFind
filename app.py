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

           # Optional — enables (a) automatic inline playback for every AI
           # recommendation, and (b) the "🔍 Search Track" verification step
           # under Artist + Track mode. Free tier: ~100 searches/day.
           # Get one at https://console.cloud.google.com/
           # (enable "YouTube Data API v3", then create an API key).
           YOUTUBE_API_KEY = "your-youtube-key-here"

QA / PM CHANGELOG (this revision):
    [FEATURE] Artist + Track mode now has an explicit "🔍 Search Track"
              confirmation step — matches are shown as selectable cards
              before any recommendation is generated.
    [BUG FIX] Pasting a YouTube link now updates the "Now Sampling" player
              immediately on parse, instead of waiting for the user to click
              "Generate Recommendations".
    [BUG FIX] YouTube title parsing is now tiered: a fast regex pass handles
              simple "Artist - Title" uploads; complex multi-segment titles
              (multiple pipes/dashes, e.g. corporate Bollywood/Punjabi
              uploads) are flagged low-confidence and refined with a tiny
              Gemini call so Artist/Title no longer get scrambled.
    [BUG FIX] Fixed the actual root cause of artist/title inversion on
              complex regional uploads: the Tier B regex heuristic and the
              Gemini refinement prompt both previously assumed the song
              title is always in a fixed position (e.g. "always last").
              Real uploads use BOTH orderings interchangeably for the same
              song (title-first AND title-last), so a position-only rule
              silently picked the movie name instead of the title whenever
              the order flipped. Both tiers now use a position-agnostic
              signal instead: the comma-containing segment (e.g. "Jassi
              Gill, Rubina Bajwa") is reliably the artist list wherever it
              falls in the string, which correctly resolves both orderings
              of "Fer Ohi Hoyea" / "Jassi Gill" / "Sargi".
    [UX FIX]  Low-confidence parses (ambiguous 3+ segment titles where even
              the improved heuristic is just a best-effort guess) now show
              an explicit "⚠️ Uncertain — please verify" badge instead of
              the same confident green badge as a clean match — a prior
              revision silently displayed both with equal confidence, which
              masked exactly the cases most likely to be wrong. A diagnostic
              caption also explains WHY (no GEMINI_API_KEY configured vs.
              Gemini was attempted but didn't resolve it) so users can tell
              the difference between "not configured" and "genuinely hard".
    [UX FIX]  "🔍 Search Track" now appends music-context keywords to the
              query behind the scenes (e.g. "official audio music song")
              when the user provides a sparse query (single keyword or
              artist-only), so YouTube search no longer surfaces unrelated
              talk-show clips, shorts, or OTT promo content instead of
              actual songs.
    [FEATURE] "My Playlist Vault" is now an active media queue: a Playlist
              Playback Bar above the tracklist shows the current
              Previous/Play-Pause/Next position, and the same controls are
              mirrored inline on the "Now Sampling" player. Next/Previous
              correctly gray out at the absolute ends of the list. Note:
              st.video() has no programmatic play/pause API for an embedded
              YouTube iframe, so "Pause" honestly unloads the player with a
              clear "Paused." state rather than faking a control that can't
              actually reach into the iframe.
    [FEATURE] Vault rows now have 🔼/🔽 reorder buttons (boundary-disabled
              at the first/last row). Reordering tracks the currently
              playing TRACK across the swap, not the numeric slot, so the
              active queue position never desyncs from what's actually
              loaded in the player.
"""

import os
import re
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

# Google retires Gemini model IDs on a rolling basis — gemini-1.5-flash and
# gemini-2.0-flash have both already been shut down (404 NOT_FOUND on any
# request). gemini-2.5-flash is the current widely-available default as of
# June 2026. If you hit a 404 again in the future, just change this one
# string — everything else in the app is model-agnostic.
GEMINI_MODEL = "gemini-2.5-flash"

# If GEMINI_MODEL above 404s for your key/account, TrackFind will automatically
# retry against each of these, in order, before giving up.
GEMINI_MODEL_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]

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


# ---------------------------------------------------------------------------
# 🔑 YOUTUBE DATA API KEY — OPTIONAL CONFIGURATION PLACEHOLDER
# ---------------------------------------------------------------------------
# This key is OPTIONAL but unlocks two features:
#   (a) The "🔍 Search Track" verification step in Artist + Track mode.
#   (b) Automatic inline playback for AI-recommended tracks (not just pasted
#       links), since Gemini only ever returns song names, never URLs.
#
# Without it, both features degrade gracefully: search becomes a manual
# confirmation step instead of real candidates, and playback falls back to
# a "Find & play on YouTube" link button.
#
# Get a free key in ~2 minutes:
#   1. https://console.cloud.google.com/ -> create/select a project
#   2. APIs & Services -> Library -> enable "YouTube Data API v3"
#   3. APIs & Services -> Credentials -> Create Credentials -> API key
#
# Free quota: 10,000 units/day; each search costs 100 units (~100 searches/day).
#
# Same priority order as the Gemini key:
#   1. Streamlit secrets   -> YOUTUBE_API_KEY = "..."
#   2. Environment variable -> export YOUTUBE_API_KEY="..."
#   3. The placeholder string below (NOT recommended for production)
#
# >>> REPLACE THE LINE BELOW WITH YOUR OWN KEY, OR BETTER YET, USE SECRETS <<<
YOUTUBE_API_KEY_PLACEHOLDER = "YOUR_YOUTUBE_API_KEY_HERE"


def get_youtube_api_key() -> Optional[str]:
    """Resolve the optional YouTube Data API key from secrets, env, or placeholder."""
    try:
        if "YOUTUBE_API_KEY" in st.secrets:
            return st.secrets["YOUTUBE_API_KEY"]
    except Exception:
        pass

    env_key = os.environ.get("YOUTUBE_API_KEY")
    if env_key:
        return env_key

    if YOUTUBE_API_KEY_PLACEHOLDER and YOUTUBE_API_KEY_PLACEHOLDER != "YOUR_YOUTUBE_API_KEY_HERE":
        return YOUTUBE_API_KEY_PLACEHOLDER

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
        transition: all 0.15s ease;
    }
    .tf-vault-row span.tf-vault-title { font-weight: 600; color: #F0FDF9; }
    .tf-vault-row span.tf-vault-artist { color: #6EE7B7; font-size: 0.85rem; }
    .tf-vault-row-active {
        background: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.4);
        box-shadow: 0 0 16px rgba(16,185,129,0.15);
    }

    /* ---------- Playlist Playback Bar ---------- */
    .tf-playback-bar {
        background: linear-gradient(135deg, rgba(16,185,129,0.14) 0%, rgba(15,23,23,0.6) 70%);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 16px;
        padding: 1rem 1.3rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .tf-playback-bar-info { display: flex; flex-direction: column; min-width: 0; }
    .tf-playback-bar-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #6EE7B7;
        margin-bottom: 0.15rem;
    }
    .tf-playback-bar-track {
        font-weight: 700;
        color: #F0FDF9;
        font-size: 1rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 360px;
    }
    .tf-playback-bar-artist { color: #9CB8B0; font-size: 0.84rem; }

    /* ---------- Search-result candidate card (Feature 1) ---------- */
    .tf-candidate {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.7rem;
        transition: all 0.15s ease;
    }
    .tf-candidate:hover {
        border-color: rgba(16,185,129,0.4);
        background: rgba(16,185,129,0.06);
    }
    .tf-candidate img {
        border-radius: 8px;
        width: 64px;
        height: 48px;
        object-fit: cover;
        flex-shrink: 0;
    }
    .tf-candidate-title {
        font-weight: 600;
        font-size: 0.88rem;
        color: #F0FDF9;
        line-height: 1.25;
        margin: 0;
    }
    .tf-candidate-channel {
        font-size: 0.74rem;
        color: #7E978F;
        margin: 0.1rem 0 0 0;
    }
    .tf-confidence-pill {
        display: inline-block;
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        padding: 0.1rem 0.5rem;
        border-radius: 999px;
        text-transform: uppercase;
        margin-left: 0.4rem;
    }
    .tf-confidence-high {
        background: rgba(16,185,129,0.15);
        color: #6EE7B7;
        border: 1px solid rgba(16,185,129,0.35);
    }
    .tf-confidence-refined {
        background: rgba(167,139,250,0.15);
        color: #C4B5FD;
        border: 1px solid rgba(167,139,250,0.35);
    }
    .tf-confidence-low {
        background: rgba(245,158,11,0.15);
        color: #FCD34D;
        border: 1px solid rgba(245,158,11,0.35);
    }

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
    .stButton > button:disabled {
        background: rgba(255,255,255,0.04) !important;
        color: #6B8580 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        box-shadow: none !important;
        cursor: default !important;
    }

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

    /* Secondary CTA (Search Track) */
    .tf-search-btn .stButton > button {
        background: linear-gradient(135deg, rgba(110,231,183,0.16), rgba(16,185,129,0.05)) !important;
        color: #6EE7B7 !important;
        font-weight: 700 !important;
        width: 100%;
        border: 1px dashed rgba(110,231,183,0.45) !important;
    }
    .tf-search-btn .stButton > button:hover {
        border-style: solid !important;
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


class ParsedTitle(BaseModel):
    """Strict schema for the Gemini micro-parse fallback (Bug Fix #3 / #5)."""
    artist: str = Field(description="The primary recording singer/artist's name only — a person or group, never a movie/film/album name, no featured artists, no promotional text.")
    title: str = Field(description="The actual standalone song title only — never a movie/film/album name, never a language or genre tag, no promotional text.")


@dataclass
class SeedTrack:
    artist: str = ""
    title: str = ""
    raw_source: str = ""   # original pasted YouTube URL, for display/debug
    video_id: str = ""     # real YouTube video ID, set only when known
    parse_confidence: str = "high"  # "high" (simple regex), "low" (ambiguous regex guess), or "refined" (Gemini-assisted)
    refine_note: str = ""   # diagnostic: why Gemini refinement didn't happen/help, if applicable


# ===========================================================================
# 5. SESSION STATE INITIALIZATION
# ===========================================================================

def init_session_state():
    defaults = {
        "recommendations": [],          # list of dicts: Song, Artist, Reason
        "playlist_vault": [],           # list of dicts: Song, Artist, Reason
        "now_playing": None,            # dict: {"Song":..., "Artist":..., "video_id":...}
        "last_seed": None,              # SeedTrack as dict
        "has_generated": False,
        "working_model": None,          # whichever Gemini model actually succeeded
        "search_candidates": [],        # list of dicts from YouTube search, for Feature 1
        "confirmed_seed": None,         # dict: {"artist":..., "title":..., "video_id":...} once user confirms a candidate
        "last_search_query": "",        # actual augmented query sent to YouTube search
        "last_search_display_query": "",  # clean user-facing version (no injected keywords)
        "search_performed": False,      # persists across reruns (fixes a transient-button-state bug)
        "auto_generate_enabled": True,  # [UX FEATURE] auto-generate recommendations on seed confirm
        "_last_auto_generated_fingerprint": None,  # guards against re-triggering on every rerun
        "_last_restored_upload_id": None,  # guards against re-importing the same vault file every rerun
        "current_queue_index": None,    # [FEATURE] index into playlist_vault for Next/Previous — None when playback isn't sourced from the vault queue
        "confirm_clear_pending": False,  # [UX] tracks the single-click-then-confirm flow for Clear Playlist
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
    r"\(\s*full\s*video\s*\)",
    r"\[\s*full\s*video\s*\]",
    r"\(\s*full\s*song\s*\)",
    r"\[\s*full\s*song\s*\]",
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
    r"\bfull\s*video\b",
    r"\bremastered\b",
    r"\bclean\s*version\b",
    r"\bexplicit\s*version\b",
    r"\(\s*explicit\s*\)",
    r"\[\s*explicit\s*\]",
]

FLUFF_REGEX = re.compile("|".join(FLUFF_PATTERNS), flags=re.IGNORECASE)

# Whole-segment noise — matches an ENTIRE pipe/dash-delimited segment that is
# pure metadata cruft (not just a substring within a meaningful segment).
# Used only in the multi-segment (Tier B) parsing path.
NOISE_SEGMENT_PATTERNS = [
    r"^(latest|new|hit|top|best)?\s*(punjabi|hindi|bollywood|english|haryanvi)?\s*song(s)?\s*\d*$",
    r"^official\s*(video|audio)?$",
]
NOISE_SEGMENT_REGEX = re.compile("|".join(NOISE_SEGMENT_PATTERNS), flags=re.IGNORECASE)


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


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_youtube_description(video_id: str, api_key: str) -> Optional[str]:
    """
    [BUG FIX] oEmbed (used by fetch_youtube_title above) does NOT expose the
    video description at all — only title/author/thumbnail. Many regional
    uploads put the authoritative "Song: X / Movie: Y / Singer: Z" labels in
    the DESCRIPTION, not the title, so without this the parser never even
    sees that structured information.

    This uses the official YouTube Data API v3 videos.list endpoint
    (part=snippet), which DOES include the description, at a cost of just 1
    quota unit per call (out of the free 10,000/day). Requires the optional
    YOUTUBE_API_KEY — returns None gracefully if it's not configured or the
    call fails for any reason, so callers always have the title-only path
    as a fallback.
    """
    if not api_key or not video_id:
        return None
    try:
        import urllib.request
        import urllib.parse

        params = {"part": "snippet", "id": video_id, "key": api_key}
        url = "https://www.googleapis.com/youtube/v3/videos?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = data.get("items", [])
        if items:
            return items[0].get("snippet", {}).get("description", "") or None
        return None
    except Exception:
        return None


# Matches "Song:", "Track:", "Singer:", "Artist:" style labels commonly
# found in regional music video DESCRIPTIONS (not titles). Deliberately
# conservative: only activates on cleanly-spaced text and caps value length,
# since a wrong confident guess is worse than admitting uncertainty.
_DESC_TARGET_LABELS = ["song", "track", "singer", "singers", "artist"]
_DESC_LABEL_REGEX = re.compile(
    r"\b(" + "|".join(_DESC_TARGET_LABELS) + r")\s*[:\-]\s*([^:\-\n]{1,50}?)(?=\s+\b[A-Za-z]+\s*[:\-]|\n|$)",
    flags=re.IGNORECASE,
)
_DESC_GENRE_WORD = r"(punjabi|hindi|bollywood|english|haryanvi|movie|film)"

# A lowercase letter immediately followed by an uppercase letter with no
# space (e.g. "TalliMovie") signals the description has no real spacing
# between credit-line fields. Our simple regex can't reliably parse that
# format — when detected, skip regex entirely and defer to Gemini, which
# has no trouble reading run-together text correctly.
_RUN_TOGETHER_SIGNAL = re.compile(r"[a-z][A-Z]")


def extract_labels_from_description(description: str) -> dict:
    """
    [BUG FIX] Fast, free, zero-API-call extraction of explicit "Song:" /
    "Singer:" / "Artist:" / "Track:" labels from a video DESCRIPTION (not
    title). When present and cleanly spaced (very common on official
    Bollywood/Punjabi label uploads), this is authoritative and far more
    reliable than any title-string heuristic — it's the uploader directly
    telling us which field is which.

    Returns a dict like {"song": "Ho Gaya Talli", "singer": "Diljit Dosanjh"}
    with whichever labels were found. Returns an empty dict if no labels are
    found OR if the text looks run-together with no real spacing (that case
    is better handled by Gemini, which doesn't share this limitation).
    """
    if not description:
        return {}

    if _RUN_TOGETHER_SIGNAL.search(description):
        return {}

    found = {}
    for match in _DESC_LABEL_REGEX.finditer(description):
        label = match.group(1).strip().lower()
        if label in ("singers", "artist"):
            label = "singer"
        value = match.group(2).strip().strip(".,;").strip()
        value = re.sub(r"^" + _DESC_GENRE_WORD + r"\s+", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s+" + _DESC_GENRE_WORD + r"$", "", value, flags=re.IGNORECASE).strip()
        if value and len(value.split()) <= 6 and label not in found:
            found[label] = value
    return found


def clean_youtube_title(raw_title: str) -> str:
    """
    Strip common YouTube fluff like 'Official Video', 'HD', '4K', 'Lyrics',
    '[Official Audio]' etc., leaving a clean candidate track string. This is
    intentionally conservative — it removes known noise tokens but does NOT
    attempt to split Artist/Title (see split_artist_title below for that).
    """
    if not raw_title:
        return ""

    cleaned = FLUFF_REGEX.sub("", raw_title)

    # Remove any now-empty bracket/paren pairs left behind
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\[\s*\]", "", cleaned)

    # Collapse leftover separator junk (extra dashes, pipes, double spaces)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s*[-|]\s*$", "", cleaned)
    cleaned = re.sub(r"^\s*[-|]\s*", "", cleaned)
    cleaned = cleaned.strip(" -|·•").strip()

    return cleaned


def _is_structurally_complex(cleaned_title: str) -> bool:
    """
    Decides whether a cleaned title is simple enough for pure regex, or
    complex enough to need the Gemini micro-parse fallback (Bug Fix #3).

    "Complex" = 2 or more pipe characters, OR a mix of pipes AND dashes,
    OR 3+ total separators of any kind. These are the corporate-upload
    patterns (e.g. "Artist | Album | Latest Punjabi Song — Title") where a
    fixed-position regex split reliably grabs the wrong segment.
    """
    pipe_count = cleaned_title.count("|")
    dash_count = len(re.findall(r"\s[-–—]\s", cleaned_title))
    total_separators = pipe_count + dash_count
    return pipe_count >= 2 or (pipe_count >= 1 and dash_count >= 1) or total_separators >= 3


def split_artist_title_regex(cleaned_title: str) -> SeedTrack:
    """
    TIER A — fast, free regex split for simple, unambiguous titles.

    Handles the common "Artist - Title" / "Artist | Title" case directly.
    For anything structurally complex (see _is_structurally_complex), this
    still produces a best-effort guess, but flags parse_confidence="low" so
    the caller knows to attempt a Gemini refinement instead of trusting it.
    """
    if not cleaned_title:
        return SeedTrack(artist="", title="", parse_confidence="high")

    if not _is_structurally_complex(cleaned_title):
        # Simple case: split on the FIRST separator found.
        for sep_pattern in [r"\s+[-–—]\s+", r"\s*\|\s*"]:
            match = re.search(sep_pattern, cleaned_title)
            if match:
                idx = match.start()
                artist = cleaned_title[:idx].strip()
                title = cleaned_title[match.end():].strip()
                if artist and title:
                    return SeedTrack(artist=artist, title=title, parse_confidence="high")
        return SeedTrack(artist="", title=cleaned_title.strip(), parse_confidence="high")

    # TIER B fallback heuristic (used only if Gemini refinement is
    # unavailable, or as a sanity check against it) — explicitly
    # low-confidence, since 3+ segment titles are genuinely ambiguous.
    #
    # [BUG FIX] Earlier versions assumed a FIXED POSITION: "first segment is
    # always the artist, last segment is always the title". That assumption
    # is wrong about as often as it's right — regional uploads commonly use
    # BOTH "Title - Artist | Movie | tag" AND "Artist | Movie | tag - Title"
    # orderings interchangeably, so a position-only rule silently picks the
    # movie name as the title whenever the order flips (e.g. "Sargi" picked
    # instead of "Fer Ohi Hoyea").
    #
    # Instead, this uses a CONTENT-based signal that doesn't care about
    # position: the segment containing a comma (e.g. "Jassi Gill, Rubina
    # Bajwa" — primary singer + featured actor/actress) is reliably the
    # artist list in this title format, regardless of where it falls in the
    # string. The primary artist is the first name before the comma. Some
    # uploads join co-stars with "&" instead of a comma (e.g. "Diljit
    # Dosanjh & Sonam Bajwa") — treated the same way.
    segments = re.split(r"\s*\|\s*|\s+[-–—]\s+", cleaned_title)
    segments = [s.strip() for s in segments if s.strip()]

    if len(segments) <= 1:
        return SeedTrack(artist="", title=cleaned_title.strip(), parse_confidence="low")

    if len(segments) == 2:
        return SeedTrack(artist=segments[0].strip(), title=segments[1].strip(), parse_confidence="low")

    meaningful = [s for s in segments if not NOISE_SEGMENT_REGEX.match(s)]
    if not meaningful:
        meaningful = segments

    multi_name_segments = [s for s in meaningful if "," in s or re.search(r"\s&\s", s)]

    if multi_name_segments:
        # The multi-name segment is the artist/cast list, wherever it falls.
        artist_segment = multi_name_segments[0]
        first_name = re.split(r",|\s&\s", artist_segment)[0].strip()
        artist_guess = first_name
        remaining = [s for s in meaningful if s != artist_segment]
    else:
        # No comma or "&" anywhere in the string — fall back to the
        # original first-segment-is-artist assumption, since we have no
        # better signal.
        artist_guess = meaningful[0].strip()
        remaining = meaningful[1:]

    if not remaining:
        return SeedTrack(artist=artist_guess, title="", parse_confidence="low")

    if len(remaining) == 1:
        return SeedTrack(artist=artist_guess, title=remaining[0].strip(), parse_confidence="low")

    # 2+ candidates remain after removing the artist segment and pure noise
    # (typically: the movie/album name AND the song title). Movie/album
    # names in this title format are overwhelmingly a single short word
    # (e.g. "Sargi", "Sufna"), while song titles are usually multi-word.
    # Prefer a multi-word remaining segment as the title; only fall back to
    # plain position (last remaining segment) if every candidate is a
    # single word, since we then have no remaining signal to use.
    multi_word = [s for s in remaining if len(s.split()) > 1]
    title_guess = multi_word[0].strip() if multi_word else remaining[-1].strip()

    return SeedTrack(artist=artist_guess, title=title_guess, parse_confidence="low")


def split_artist_title(cleaned_title: str) -> SeedTrack:
    """Backwards-compatible wrapper name used elsewhere in the app."""
    return split_artist_title_regex(cleaned_title)


def refine_title_with_gemini(messy_title: str, description: Optional[str] = None) -> Optional[ParsedTitle]:
    """
    TIER C — micro Gemini call used ONLY when the regex split was flagged
    low-confidence (structurally complex titles like multi-pipe corporate
    uploads). This is a tiny, cheap, single-purpose call distinct from the
    main recommendation engine — it does not consume the recommendation
    model-fallback chain and fails silently (returns None) so callers always
    have the regex guess as a safety net.

    [BUG FIX] An earlier revision assumed the song title is always the LAST
    segment of the string. That is wrong about as often as it's right —
    real-world regional uploads use BOTH orderings interchangeably:
        "Jassi Gill, Rubina Bajwa | Sargi | Latest Punjabi Song — Fer Ohi Hoyea"
        "Fer Ohi Hoyea - Jassi Gill, Rubina Bajwa | Sargi | Latest Punjabi Song"
    are the SAME song, with the title in opposite positions. A fixed-position
    rule silently mis-picks the movie name ("Sargi") as the title whenever
    the order flips. This prompt now uses POSITION-AGNOSTIC reasoning
    instead: the comma-containing segment (multiple names) is reliably the
    artist list wherever it falls, and a short single-word segment is
    reliably the movie/album name wherever IT falls — neither rule depends
    on first/last position.

    [BUG FIX] Also accepts the video DESCRIPTION when available. oEmbed
    (used for the title) never exposes the description, so earlier
    revisions never saw the explicit "Song:"/"Singer:" labels that many
    regional uploads put there instead of in the title — this was the
    actual root cause behind cases like "Ho Gaya Talli | Super Singh |
    Diljit Dosanjh & Sonam Bajwa | Jatinder Shah" (no comma — co-stars
    joined with "&" instead — so the title-only heuristic had no reliable
    signal at all, while the description plainly states
    "Song - Ho Gaya Talli Movie - Super Singh Singer - Diljit Dosanjh").
    """
    client = get_genai_client()
    if client is None:
        return None

    description_block = (
        f'\n\nVideo description (may contain explicit "Song:"/"Singer:" labels '
        f'— if so, these are MORE reliable than guessing from the title alone):\n"{description.strip()[:800]}"'
        if description else ""
    )

    prompt = f"""
You are a metadata extraction specialist for music video titles, especially
Bollywood, Punjabi, and other South Asian regional film/music uploads where
the singer, movie/film name, genre tag, and song title are mixed together
with "|" or "-"/"—" as separators, in NO CONSISTENT ORDER. The song title
may appear FIRST, LAST, or in the middle of the string — do not assume a
fixed position. Co-stars or featured names may be joined with a comma OR
with "&" (e.g. "Diljit Dosanjh & Sonam Bajwa") — both indicate a multi-name
list where the FIRST name is the one to prefer as primary artist if no
clearer "Singer:" label is available.

Raw title: "{messy_title}"{description_block}

STEP 1 — Check for explicit key-value labels first, in BOTH the title and
the description if provided.
If either text contains explicit indicators such as "Song:", "Track:",
"Singer:", "Artist:", "Movie:", or "Film:", these are AUTHORITATIVE — use the
value following "Song:" or "Track:" as the title, and the value following
"Singer:" or "Artist:" as the artist, regardless of where they appear or
which of the two texts they're found in. Ignore the value following
"Movie:" or "Film:" entirely — that is never the artist or the title.

STEP 2 — If there are no explicit labels anywhere, identify segments by
CONTENT, not position (the title is typically several "|" or dash-separated
segments):
- Find the segment that lists MULTIPLE NAMES separated by a comma OR by "&"
  (e.g. "Jassi Gill, Rubina Bajwa" or "Diljit Dosanjh & Sonam Bajwa"). This
  is reliably the ARTIST/CAST segment, no matter where in the string it
  falls. The FIRST name in that list is usually the primary singer; the
  rest are featured/secondary names (co-stars, actors) to discard.
- Find any segment that is generic filler — "Latest Punjabi Song", "Hindi
  Song", "Bollywood", "Official", "Full Video", "Full Song", or a bare year.
  Discard these entirely; they are never the artist or the title.
- Among the segments that remain after removing the artist/cast segment and
  the filler segments, the MOVIE/ALBUM name is typically a SHORT segment of
  just one or two words that reads like a single proper noun (e.g. "Sargi",
  "Super Singh"). Discard this too — it is never the artist or the title,
  even though its brevity might make it look like a plausible title.
- A segment that is a person's name but reads more like a music director,
  composer, or lyricist (common trailing segment, e.g. "Jatinder Shah",
  "B Praak") rather than the lead singer should be treated as a CREDIT, not
  the primary artist, unless no better candidate exists.
- Whatever segment is left after removing the artist/cast, the filler, and
  the movie/album name IS the song title — regardless of whether it was the
  first, last, or a middle segment in the original string.

STEP 3 — Self-check before answering:
- Is the artist you chose an actual singer/performer, not a movie/film name
  and not a composer/music-director credit (unless that's truly the only
  name available)? If not, you picked the wrong segment.
- Is the title you chose the actual song name — not the movie/album name,
  and not a generic genre/language tag? Double-check it isn't simply the
  shortest remaining segment chosen by position rather than content.

Return ONLY the primary singer/artist and the actual standalone song title.
""".strip()

    try:
        response = client.models.generate_content(
            model=st.session_state.get("working_model") or GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ParsedTitle,
                temperature=0.1,
                max_output_tokens=256,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if parsed:
            return parsed
        raw = (response.text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(json)?", "", raw).rstrip("`").strip()
        data = json.loads(raw)
        return ParsedTitle(**data)
    except Exception:
        return None


def parse_youtube_link(url: str, allow_gemini_refine: bool = True) -> SeedTrack:
    """
    Full pipeline:
      1. Extract video ID.
      2. Fetch raw title (oEmbed, no key needed) -> clean fluff -> Tier A/B
         regex split.
      3. [BUG FIX] If a YOUTUBE_API_KEY is configured, ALSO fetch the video
         DESCRIPTION (oEmbed never exposes this — only the Data API does).
         Regional uploads frequently put authoritative "Song:"/"Singer:"
         labels in the description rather than the title, which earlier
         revisions never even looked at.
      4. If the regex split is low-confidence: try the free description
         label extractor first (instant, no API call); if that doesn't
         resolve it, fall back to a Gemini micro-call given BOTH the title
         and (if available) the description for maximum context.
    """
    video_id = extract_youtube_id(url)
    if not video_id:
        return SeedTrack(raw_source=url)

    raw_title = fetch_youtube_title(video_id)
    if not raw_title:
        return SeedTrack(raw_source=url, video_id=video_id)

    cleaned = clean_youtube_title(raw_title)
    seed = split_artist_title_regex(cleaned)
    seed.raw_source = url
    seed.video_id = video_id

    if seed.parse_confidence != "low":
        return seed

    # Low-confidence regex guess — try to do better.
    yt_key = get_youtube_api_key()
    description = fetch_youtube_description(video_id, yt_key) if yt_key else None

    # FAST PATH: free, instant label extraction from the description, if
    # it's cleanly formatted enough to trust.
    if description:
        labels = extract_labels_from_description(description)
        if labels.get("song") and labels.get("singer"):
            seed.artist = labels["singer"]
            seed.title = labels["song"]
            seed.parse_confidence = "refined"
            return seed

    # SLOW PATH: Gemini micro-call, given the title AND description (when
    # available) for maximum context — Gemini handles run-together/messy
    # description text far better than regex can.
    if not allow_gemini_refine:
        seed.refine_note = "Gemini refinement skipped for this call."
    elif not get_api_key():
        seed.refine_note = "Add a GEMINI_API_KEY to enable automatic refinement of complex titles like this one."
    else:
        refined = refine_title_with_gemini(cleaned, description=description)
        if refined and refined.artist and refined.title:
            seed.artist = refined.artist
            seed.title = refined.title
            seed.parse_confidence = "refined"
        else:
            seed.refine_note = "Gemini refinement was attempted but didn't return a confident result — please double-check the fields below."

    return seed


def build_music_search_query(typed_artist: str, typed_title: str) -> str:
    """
    [UX FIX] Constructs the YouTube search query used by the "🔍 Search
    Track" verification step. A bare single-keyword query (e.g. just
    "diljit" or just "aha") returns broad, irrelevant results from YouTube's
    general search — talk-show clips, Shorts, or even OTT platform promos
    (e.g. the "aha" streaming service) instead of actual songs.

    To keep results strictly music-focused without requiring the user to
    type anything extra, this appends explicit audio/music modifiers behind
    the scenes:
        - Artist only            -> "<artist> official audio music song"
        - Title only              -> "<title> official audio music song"
        - Both artist and title   -> "<artist> <title> song"
        - Neither                 -> "" (caller handles the empty case)
    """
    typed_artist = (typed_artist or "").strip()
    typed_title = (typed_title or "").strip()

    if typed_artist and typed_title:
        return f"{typed_artist} {typed_title} song"
    if typed_artist and not typed_title:
        return f"{typed_artist} official audio music song"
    if typed_title and not typed_artist:
        return f"{typed_title} official audio music song"
    return ""


def build_youtube_search_url(query: str) -> str:
    import urllib.parse
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"


def build_youtube_watch_url(video_id: str) -> str:
    """A real, embeddable YouTube watch URL — safe to pass to st.video()."""
    return f"https://www.youtube.com/watch?v={video_id}" if video_id else ""


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def search_youtube_video_id(query: str, api_key: str) -> Optional[str]:
    """
    Resolves a search query to a single best-match real YouTube video ID
    using the official YouTube Data API v3 search.list endpoint. Used for
    auto-resolving playback on AI recommendations (no key = graceful None).
    """
    results = search_youtube_tracks(query, api_key, max_results=1)
    return results[0]["video_id"] if results else None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def search_youtube_tracks(query: str, api_key: str, max_results: int = 5) -> List[dict]:
    """
    [FEATURE 1] Resolves a search query to multiple candidate videos via the
    official YouTube Data API v3 search.list endpoint, so the user can
    explicitly confirm which exact track they mean before generating
    recommendations.

    Returns a list of dicts: {video_id, title, channel, thumbnail}.
    Returns an empty list on any failure (no key, bad key, quota exceeded,
    network issue, no results) so the UI can degrade to a manual
    confirmation step instead of crashing.

    Results are cached for 24h per unique (query, max_results) pair to
    conserve the free 100-searches/day quota.
    """
    if not api_key or not query:
        return []
    try:
        import urllib.request
        import urllib.parse

        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max(1, min(max_results, 10)),
            "key": api_key,
        }
        url = "https://www.googleapis.com/youtube/v3/search?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []
        for item in data.get("items", []):
            vid = item.get("id", {}).get("videoId")
            if not vid:
                continue
            snippet = item.get("snippet", {})
            results.append({
                "video_id": vid,
                "title": snippet.get("title", "Untitled"),
                "channel": snippet.get("channelTitle", ""),
                "thumbnail": snippet.get("thumbnails", {}).get("default", {}).get("url", ""),
            })
        return results
    except Exception:
        return []


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


def _call_gemini_model(client: "genai.Client", model_name: str, prompt: str) -> List[dict]:
    """Single-shot call against one specific model name. Raises on failure."""
    response = client.models.generate_content(
        model=model_name,
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


def get_recommendations(artist: str, title: str, num_recs: int) -> List[dict]:
    """
    Calls Gemini with a forced JSON schema (Pydantic) so the response is
    guaranteed to match: [{"Song":..., "Artist":..., "Reason":...}, ...]

    Google periodically retires Gemini model IDs (gemini-1.5-flash and
    gemini-2.0-flash are both already shut down as of mid-2026). To keep
    TrackFind resilient to the next retirement wave, this tries GEMINI_MODEL
    first, then walks GEMINI_MODEL_FALLBACKS on a 404/NOT_FOUND, and remembers
    whichever model actually worked for the rest of the session.
    """
    client = get_genai_client()
    if client is None:
        raise RuntimeError(
            "Gemini API key not configured. Add GEMINI_API_KEY to your "
            "Streamlit secrets or environment variables."
        )

    prompt = build_recommendation_prompt(artist, title, num_recs)

    # Try the last-known-good model first (if any), then the configured
    # default, then the fallback chain — without duplicating attempts.
    candidates = []
    sticky_model = st.session_state.get("working_model")
    for m in [sticky_model, GEMINI_MODEL, *GEMINI_MODEL_FALLBACKS]:
        if m and m not in candidates:
            candidates.append(m)

    last_error: Optional[Exception] = None
    for model_name in candidates:
        try:
            result = _call_gemini_model(client, model_name, prompt)
            st.session_state["working_model"] = model_name
            return result
        except Exception as e:
            error_str = str(e)
            last_error = e
            # Only keep trying the next candidate on a "model not found"
            # style error. Any other error (bad key, quota, network) should
            # surface immediately instead of silently retrying 4x.
            if "404" not in error_str and "NOT_FOUND" not in error_str.upper():
                raise

    raise RuntimeError(
        f"None of the configured Gemini models are available for this API key "
        f"(tried: {', '.join(candidates)}). Last error: {last_error}"
    )


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
        # [FEATURE] Resolve a real YouTube link at save-time so exports can
        # include a playable URL, not just text. Uses the official Data API
        # (already cached) when a key is configured; falls back to a
        # YouTube search URL otherwise so the field is never empty.
        video_id = track.get("video_id", "")
        if not video_id:
            yt_key = get_youtube_api_key()
            query = f"{track.get('Artist','')} {track.get('Song','')}".strip()
            if yt_key and query:
                video_id = search_youtube_video_id(query, yt_key) or ""

        if video_id:
            track["video_id"] = video_id
            track["youtube_url"] = build_youtube_watch_url(video_id)
        else:
            query = f"{track.get('Artist','')} {track.get('Song','')}".strip()
            track["youtube_url"] = build_youtube_search_url(query) if query else ""

        st.session_state.playlist_vault.append(track)
        st.toast(f"Added '{track['Song']}' to your Playlist Vault ✅", icon="🎶")
    else:
        st.toast(f"'{track['Song']}' is already in your vault.", icon="ℹ️")


def remove_from_playlist(index: int):
    if 0 <= index < len(st.session_state.playlist_vault):
        removed = st.session_state.playlist_vault.pop(index)
        st.toast(f"Removed '{removed['Song']}' from your vault.", icon="🗑️")

        # [BUG FIX] Keep the active queue index consistent after a deletion
        # so Next/Previous don't silently point at the wrong track.
        current_idx = st.session_state.get("current_queue_index")
        if current_idx is not None:
            if index == current_idx:
                # The currently-playing track was removed — drop out of
                # queue mode rather than guess which track should play next.
                st.session_state.current_queue_index = None
            elif index < current_idx:
                # Everything after the removed slot shifts down by one.
                st.session_state.current_queue_index = current_idx - 1


def load_queue_track(index: int):
    """
    [FEATURE] Loads the vault track at `index` into the player and marks it
    as the active queue position, so subsequent Next/Previous clicks know
    where they are. Used by the vault's own "▶️ Play" buttons AND by the
    Previous/Next controls under the player.
    """
    vault = st.session_state.playlist_vault
    if not (0 <= index < len(vault)):
        return
    track = vault[index]
    st.session_state.current_queue_index = index
    st.session_state.now_playing = {
        "Song": track.get("Song", ""),
        "Artist": track.get("Artist", ""),
        "video_id": track.get("video_id", ""),
        "Reason": track.get("Reason", ""),
    }


def queue_play_next():
    """[FEATURE] Advances the queue by one track, if not already at the end."""
    idx = st.session_state.get("current_queue_index")
    vault = st.session_state.playlist_vault
    if idx is None or not vault:
        return
    if idx + 1 < len(vault):
        load_queue_track(idx + 1)


def queue_play_previous():
    """[FEATURE] Steps the queue back by one track, if not already at the start."""
    idx = st.session_state.get("current_queue_index")
    vault = st.session_state.playlist_vault
    if idx is None or not vault:
        return
    if idx - 1 >= 0:
        load_queue_track(idx - 1)


def move_track_in_vault(index: int, direction: int):
    """
    [FEATURE] Swaps the track at `index` with its neighbor at `index +
    direction` (direction is -1 for "move up", +1 for "move down"). No-ops
    safely if the swap would go out of bounds.

    [BUG FIX] Keeps `current_queue_index` pointed at the SAME TRACK (not the
    same numeric slot) across the swap — without this, reordering while a
    track is actively playing would silently make Next/Previous jump to the
    wrong song, since the integer index alone doesn't follow the track when
    its position changes.
    """
    vault = st.session_state.playlist_vault
    target = index + direction
    if not (0 <= index < len(vault)) or not (0 <= target < len(vault)):
        return

    current_idx = st.session_state.get("current_queue_index")
    playing_track_id = None
    if current_idx is not None and 0 <= current_idx < len(vault):
        t = vault[current_idx]
        playing_track_id = (t.get("Song", "").lower(), t.get("Artist", "").lower())

    vault[index], vault[target] = vault[target], vault[index]

    if playing_track_id is not None:
        for i, t in enumerate(vault):
            if (t.get("Song", "").lower(), t.get("Artist", "").lower()) == playing_track_id:
                st.session_state.current_queue_index = i
                break


def playlist_to_json_bytes() -> bytes:
    """
    Full-fidelity export of the playlist vault as JSON — every field saved
    per track (Song, Artist, Reason — the mood/match explanation Gemini
    gave when recommending it — video_id, and youtube_url) comes along
    automatically since this is a direct dump of the stored track dicts.
    This is what powers "resume later" without requiring an account: a
    returning user just re-uploads this file to restore their playlist.
    """
    payload = {
        "trackfind_export_version": 1,
        "playlist_vault": st.session_state.playlist_vault,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def restore_playlist_from_file(uploaded_bytes: bytes, filename: str = "") -> tuple[bool, str]:
    """
    Parses a previously-downloaded playlist file and restores it into the
    current session's vault — merging with, not replacing, anything
    already saved (duplicates by Song+Artist are skipped). Returns
    (success, message).
    """
    try:
        data = json.loads(uploaded_bytes.decode("utf-8"))
    except Exception:
        return False, "That file couldn't be read — try downloading a fresh copy of your playlist and uploading that."

    tracks = data.get("playlist_vault")
    if not isinstance(tracks, list):
        return False, "That file doesn't contain a recognizable playlist."

    existing = {(t.get("Song", "").lower(), t.get("Artist", "").lower()) for t in st.session_state.playlist_vault}
    added = 0
    for track in tracks:
        if not isinstance(track, dict) or not track.get("Song"):
            continue
        key = (track.get("Song", "").lower(), track.get("Artist", "").lower())
        if key not in existing:
            st.session_state.playlist_vault.append(track)
            existing.add(key)
            added += 1

    if added == 0:
        return True, "No new tracks to add — everything in that file is already in your vault."
    return True, f"Restored {added} track{'s' if added != 1 else ''} into your vault."


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
            key="input_mode_radio",
        )

        # Reset the confirmed seed if the user switches input modes, so a
        # confirmation from one mode doesn't leak into the other.
        if st.session_state.get("_last_input_mode") != input_mode:
            st.session_state.confirmed_seed = None
            st.session_state.search_candidates = []
            st.session_state.search_performed = False
            st.session_state["_last_input_mode"] = input_mode

        seed_artist, seed_title = "", ""
        seed_video_id = ""

        # ===================================================================
        # MODE A: 🎤 Artist + Track  ([FEATURE] Search & Verify)
        # ===================================================================
        if input_mode == "🎤 Artist + Track":
            c1, c2 = st.columns(2)
            with c1:
                typed_artist = st.text_input("Artist Name", placeholder="e.g. The Weeknd", key="typed_artist")
            with c2:
                typed_title = st.text_input("Track Title", placeholder="e.g. Blinding Lights", key="typed_title")

            st.markdown('<div class="tf-search-btn">', unsafe_allow_html=True)
            search_clicked = st.button("🔍 Search Track", use_container_width=True, key="search_track_btn")
            st.markdown('</div>', unsafe_allow_html=True)

            yt_key = get_youtube_api_key()

            # NOTE: `search_clicked` is only True on the exact script run where
            # the button was pressed — it resets to False on every later rerun
            # (e.g. when the user types into a fallback confirm field below).
            # We persist the "a search was performed" state separately so the
            # results/fallback UI doesn't vanish the instant the user
            # interacts with anything else on the page.
            if search_clicked:
                display_query = f"{typed_artist} {typed_title}".strip()
                query = build_music_search_query(typed_artist, typed_title)
                if not query:
                    st.error("Type an artist name and/or track title before searching.")
                    st.session_state.search_performed = False
                else:
                    st.session_state.confirmed_seed = None
                    st.session_state.last_search_query = query
                    st.session_state.last_search_display_query = display_query
                    st.session_state.search_performed = True
                    if yt_key:
                        with st.spinner("Searching for matching tracks..."):
                            candidates = search_youtube_tracks(query, yt_key, max_results=5)
                        st.session_state.search_candidates = candidates
                    else:
                        # No YouTube key — smart fallback: skip real search,
                        # treat the typed values as a single confirmable
                        # candidate so the verification step still exists.
                        st.session_state.search_candidates = []

            search_performed = st.session_state.get("search_performed", False)

            # ---- Render candidate results (real search) ----
            if st.session_state.search_candidates:
                display_query = st.session_state.get("last_search_display_query") or st.session_state.last_search_query
                st.caption(f"Top matches for **{display_query}** — confirm the exact track:")
                option_labels = []
                for cand in st.session_state.search_candidates:
                    option_labels.append(f"{cand['title']}  ·  {cand['channel']}")

                chosen_label = st.radio(
                    "Select the exact track",
                    options=option_labels,
                    label_visibility="collapsed",
                    key="candidate_radio",
                )
                chosen_idx = option_labels.index(chosen_label) if chosen_label in option_labels else 0
                chosen = st.session_state.search_candidates[chosen_idx]

                # Show a small thumbnail preview card for the highlighted pick
                st.markdown(
                    f"""
                    <div class="tf-candidate">
                        <img src="{chosen['thumbnail']}" alt="thumbnail">
                        <div>
                            <p class="tf-candidate-title">{chosen['title']}</p>
                            <p class="tf-candidate-channel">{chosen['channel']}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("✅ Confirm This Track", use_container_width=True, key="confirm_candidate_btn"):
                    cleaned = clean_youtube_title(chosen["title"])
                    guess = split_artist_title_regex(cleaned)
                    confirmed_artist = guess.artist or typed_artist
                    confirmed_title = guess.title or typed_title or cleaned
                    st.session_state.confirmed_seed = {
                        "artist": confirmed_artist,
                        "title": confirmed_title,
                        "video_id": chosen["video_id"],
                    }
                    # Instant playback the moment a track is confirmed. This
                    # isn't vault-queue playback, so step out of queue mode.
                    st.session_state.current_queue_index = None
                    st.session_state.now_playing = {
                        "Song": confirmed_title,
                        "Artist": confirmed_artist,
                        "video_id": chosen["video_id"],
                    }
                    st.toast(f"Confirmed: {confirmed_title} — {confirmed_artist} ✅", icon="🎯")
                    st.rerun()

            elif search_performed and not yt_key:
                # Smart fallback when no YOUTUBE_API_KEY is configured: a
                # clean manual confirmation step instead of real candidates.
                st.info(
                    "No `YOUTUBE_API_KEY` configured, so TrackFind can't pull real search "
                    "matches. Confirm the details below to proceed anyway:"
                )
                fb_artist = st.text_input("Confirm Artist", value=typed_artist, key="fallback_artist")
                fb_title = st.text_input("Confirm Track Title", value=typed_title, key="fallback_title")
                if st.button("✅ Confirm This Track", use_container_width=True, key="confirm_fallback_btn"):
                    st.session_state.confirmed_seed = {
                        "artist": fb_artist,
                        "title": fb_title,
                        "video_id": "",
                    }
                    st.session_state.current_queue_index = None
                    st.session_state.now_playing = {
                        "Song": fb_title,
                        "Artist": fb_artist,
                        "video_id": "",
                    }
                    st.session_state.search_performed = False
                    st.toast(f"Confirmed: {fb_title} — {fb_artist} ✅", icon="🎯")
                    st.rerun()

            elif search_performed and yt_key and not st.session_state.search_candidates:
                st.warning("No matches found on YouTube for that search. Try simplifying the query, or confirm manually below.")
                fb_artist = st.text_input("Confirm Artist", value=typed_artist, key="fallback_artist_noresults")
                fb_title = st.text_input("Confirm Track Title", value=typed_title, key="fallback_title_noresults")
                if st.button("✅ Confirm This Track", use_container_width=True, key="confirm_fallback_noresults_btn"):
                    st.session_state.confirmed_seed = {
                        "artist": fb_artist,
                        "title": fb_title,
                        "video_id": "",
                    }
                    st.session_state.current_queue_index = None
                    st.session_state.now_playing = {
                        "Song": fb_title,
                        "Artist": fb_artist,
                        "video_id": "",
                    }
                    st.session_state.search_performed = False
                    st.toast(f"Confirmed: {fb_title} — {fb_artist} ✅", icon="🎯")
                    st.rerun()

            # ---- Show current confirmation status ----
            if st.session_state.confirmed_seed:
                cs = st.session_state.confirmed_seed
                st.success(f"🎯 Locked in: **{cs['title']}** — *{cs['artist']}*")
                seed_artist, seed_title = cs["artist"], cs["title"]
                seed_video_id = cs.get("video_id", "")
            else:
                # Allow generating straight from typed fields too (search is
                # a recommended verification step, not a hard gate), but
                # nudge the user toward it.
                seed_artist, seed_title = typed_artist, typed_title
                if (typed_artist or typed_title) and not search_performed:
                    st.caption("💡 Tip: click **🔍 Search Track** to verify the exact match before generating recommendations.")

        # ===================================================================
        # MODE B: 🔗 YouTube Link  ([BUG FIX] Instant playback + tiered parsing)
        # ===================================================================
        else:
            yt_url = st.text_input(
                "YouTube Link",
                placeholder="https://www.youtube.com/watch?v=...",
                key="yt_url_input",
            )

            if yt_url:
                # Only re-parse when the URL actually changes, to avoid
                # re-fetching/re-parsing on every unrelated widget rerun.
                if st.session_state.get("_last_parsed_url") != yt_url:
                    with st.spinner("Parsing YouTube link..."):
                        seed = parse_youtube_link(yt_url)
                    st.session_state["_last_parsed_seed"] = asdict(seed)
                    st.session_state["_last_parsed_url"] = yt_url

                    # [BUG FIX 2] Update the player THE MOMENT we have a
                    # video_id — independent of clicking Generate.
                    if seed.video_id:
                        st.session_state.current_queue_index = None
                        st.session_state.now_playing = {
                            "Song": seed.title,
                            "Artist": seed.artist,
                            "video_id": seed.video_id,
                        }

                parsed = st.session_state.get("_last_parsed_seed", {})
                seed_video_id = parsed.get("video_id", "")

                if parsed.get("title"):
                    # Silently use the best-available parse. Any uncertainty
                    # about the parse is an internal concern, not something
                    # to surface to a listener who just wants to hear music —
                    # the "Now Sampling" card below is the single source of
                    # truth they see.
                    seed_artist = parsed.get("artist", "")
                    seed_title = parsed.get("title", "")

                    # Keep the player in sync with whatever we just parsed.
                    if st.session_state.now_playing and st.session_state.now_playing.get("video_id") == seed_video_id:
                        st.session_state.now_playing["Song"] = seed_title
                        st.session_state.now_playing["Artist"] = seed_artist

                elif seed_video_id:
                    seed_artist = st.text_input("Artist", key="yt_artist_manual", placeholder="e.g. The Weeknd")
                    seed_title = st.text_input("Track Title", key="yt_title_manual", placeholder="e.g. Blinding Lights")
                else:
                    st.caption("That doesn't look like a valid YouTube link. Try pasting the full URL, or switch to Artist + Track.")
                    seed_artist = st.text_input("Artist", key="yt_artist_manual", placeholder="e.g. The Weeknd")
                    seed_title = st.text_input("Track Title", key="yt_title_manual", placeholder="e.g. Blinding Lights")

        st.markdown("<hr class='tf-divider'>", unsafe_allow_html=True)

        st.markdown('<div class="tf-card-title">🎛️ Recommendation Controls</div>', unsafe_allow_html=True)
        num_recs = st.slider(
            "How many recommendations do you want?",
            min_value=5,
            max_value=50,
            value=5,
            step=1,
            help="Scale from a quick 5-track sample to a full 50-track deep dive.",
        )

        st.markdown(
            f"""<span class="tf-stat-chip">{num_recs}<span>tracks requested</span></span>""",
            unsafe_allow_html=True,
        )

        st.session_state.auto_generate_enabled = st.checkbox(
            "⚡ Auto-generate as soon as a track is detected or confirmed",
            value=st.session_state.get("auto_generate_enabled", True),
            help="When on, recommendations generate automatically — no need to click the button below. Turn off if you'd rather review or edit the artist/title first.",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="tf-primary-btn">', unsafe_allow_html=True)
        button_label = "🔁 Regenerate Recommendations" if st.session_state.auto_generate_enabled else "✨ Generate Recommendations"
        generate_clicked = st.button(button_label, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close tf-card

        # -------------------- GENERATE LOGIC --------------------
        def run_generation(artist: str, title: str, count: int, video_id: str = ""):
            """
            Shared generation routine used by both the manual "✨ Generate
            Recommendations" button AND the new auto-generate trigger, so
            the two paths can't drift out of sync.
            """
            if not title and not artist:
                st.error("Please provide at least a track title or artist name before generating recommendations.")
                return
            try:
                with st.spinner(f"🎧 Curating {count} tracks with Gemini AI..."):
                    results = get_recommendations(artist, title, count)
                st.session_state.recommendations = results
                st.session_state.has_generated = True
                st.session_state.last_seed = {"artist": artist, "title": title}
                st.session_state.current_queue_index = None
                st.session_state.now_playing = {
                    "Song": title,
                    "Artist": artist,
                    "video_id": video_id,
                }
                model_used = st.session_state.get("working_model", GEMINI_MODEL)
                st.success(f"🎉 Generated {len(results)} recommendations based on '{title or artist}'!")
                st.caption(f"Served by `{model_used}`")
            except RuntimeError as e:
                st.error(f"⚠️ {e}")
            except Exception as e:
                st.error(f"⚠️ Something went wrong while contacting Gemini: {e}")

        # [UX FEATURE] Auto-generate the moment a seed is confirmed — either
        # a YouTube link finishes parsing, or a Search & Verify candidate is
        # confirmed — instead of requiring an extra manual button click.
        # Guarded by a "fingerprint" of the current seed so it only fires
        # ONCE per new seed, not on every unrelated rerun (e.g. dragging the
        # slider, or editing the artist/title fields afterward).
        current_seed_fingerprint = f"{seed_artist}|{seed_title}|{seed_video_id}"
        should_auto_generate = (
            (seed_artist or seed_title)
            and current_seed_fingerprint != st.session_state.get("_last_auto_generated_fingerprint")
            and st.session_state.get("auto_generate_enabled", True)
        )

        if should_auto_generate:
            st.session_state["_last_auto_generated_fingerprint"] = current_seed_fingerprint
            run_generation(seed_artist, seed_title, num_recs, seed_video_id)
        elif generate_clicked:
            run_generation(seed_artist, seed_title, num_recs, seed_video_id)

    # -------------------- RIGHT COLUMN: PLAYBACK PREVIEW --------------------
    with col_player:
        st.markdown('<div class="tf-card">', unsafe_allow_html=True)
        st.markdown('<div class="tf-card-title">▶️ Now Sampling</div>', unsafe_allow_html=True)

        now_playing = st.session_state.now_playing
        queue_idx = st.session_state.get("current_queue_index")
        in_queue_mode = queue_idx is not None and 0 <= queue_idx < len(st.session_state.playlist_vault)

        if now_playing and (now_playing.get("Song") or now_playing.get("Artist")):
            song = now_playing.get("Song", "")
            artist = now_playing.get("Artist", "")
            video_id = now_playing.get("video_id", "")

            st.markdown(f"**{song}**")
            st.markdown(f"<span class='tf-subtle'>{artist}</span>", unsafe_allow_html=True)

            # If we don't already have a confirmed video (e.g. this came from
            # an AI recommendation or manual typing rather than a pasted
            # link or confirmed search result), try to auto-resolve a real
            # one via the YouTube Data API — but only if that optional key
            # is configured.
            if not video_id:
                yt_key = get_youtube_api_key()
                if yt_key:
                    query = f"{artist} {song}".strip()
                    if query:
                        with st.spinner("Finding video..."):
                            resolved_id = search_youtube_video_id(query, yt_key)
                        if resolved_id:
                            video_id = resolved_id
                            # Cache the resolution on now_playing so we don't
                            # re-search on every rerun of the script.
                            st.session_state.now_playing["video_id"] = resolved_id

            if video_id:
                # We have a real, confirmed YouTube video — this embeds and
                # plays directly inline.
                watch_url = build_youtube_watch_url(video_id)
                st.video(watch_url, autoplay=True)
                st.caption("🔊 Now playing.")
            else:
                # No confirmed video and no YouTube key configured (or the
                # search came up empty) — st.video() cannot play a
                # search-results page, so offer a clear, honest link instead
                # of a broken embed.
                query = f"{artist} {song}".strip()
                search_url = build_youtube_search_url(query) if query else ""
                st.markdown(
                    """
                    <div class="tf-empty-state" style="padding:1.6rem 1rem;">
                        <span class="tf-emoji">🔎</span>
                        No direct video found for this track yet.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if search_url:
                    st.link_button("🔗 Find & play on YouTube", search_url, use_container_width=True)
                if not get_youtube_api_key():
                    st.caption("Tip: add a free YOUTUBE_API_KEY to enable automatic inline playback for every recommendation.")
                else:
                    st.caption("No matching video was found automatically — try the link above.")

            # [FEATURE] Active Queue Controller — Previous / Next, inline
            # under the player. Only meaningfully active when the current
            # track came from the vault queue (clicking ▶️ on a saved track,
            # or navigating with these same controls); disabled entirely
            # otherwise since there's no queue context to step through.
            # Buttons also gray out at the absolute ends of the vault list
            # so Next/Previous can't run past the boundaries.
            st.markdown("<br>", unsafe_allow_html=True)
            vault_len = len(st.session_state.playlist_vault)
            ctrl_prev, ctrl_next = st.columns(2)
            with ctrl_prev:
                st.button(
                    "⏮️ Previous",
                    key="player_prev",
                    use_container_width=True,
                    disabled=not in_queue_mode or queue_idx == 0,
                    on_click=queue_play_previous,
                )
            with ctrl_next:
                st.button(
                    "⏭️ Next",
                    key="player_next",
                    use_container_width=True,
                    disabled=not in_queue_mode or queue_idx == vault_len - 1,
                    on_click=queue_play_next,
                )
            if not in_queue_mode:
                st.caption("Play a track from your Playlist Vault to enable queue controls.")

            # [FEATURE] Save whatever's currently playing straight to the
            # vault, without needing to find it again in the recommendations
            # list below — handy for a track pasted/searched directly, or
            # one you're sampling that you want to keep regardless of how
            # it got here.
            st.markdown("<br>", unsafe_allow_html=True)
            already_saved = any(
                t.get("Song", "").lower() == song.lower() and t.get("Artist", "").lower() == artist.lower()
                for t in st.session_state.playlist_vault
            )
            if already_saved:
                st.button("✅ Already in your Playlist", use_container_width=True, disabled=True, key="add_current_saved")
            else:
                if st.button("➕ Add to Playlist", use_container_width=True, key="add_current_to_playlist"):
                    add_to_playlist({
                        "Song": song,
                        "Artist": artist,
                        "Reason": now_playing.get("Reason", ""),
                        "video_id": video_id,
                    })
                    st.rerun()
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
                        st.session_state.current_queue_index = None
                        st.session_state.now_playing = {"Song": song, "Artist": artist, "video_id": ""}
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

    # -------------------- 🎧 PLAYLIST PLAYBACK BAR --------------------
    queue_idx = st.session_state.get("current_queue_index")
    if vault and queue_idx is not None and 0 <= queue_idx < len(vault):
        active_track = vault[queue_idx]

        st.markdown('<div class="tf-playback-bar">', unsafe_allow_html=True)
        bar_info_col, bar_controls_col = st.columns([2, 1])
        with bar_info_col:
            st.markdown(
                f"""
                <div class="tf-playback-bar-info">
                    <span class="tf-playback-bar-label">🎧 Now in Queue · Track {queue_idx + 1} of {len(vault)}</span>
                    <span class="tf-playback-bar-track">{active_track.get('Song','')}</span>
                    <span class="tf-playback-bar-artist">{active_track.get('Artist','')}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with bar_controls_col:
            bar_prev, bar_next = st.columns(2)
            with bar_prev:
                if st.button("⏮️", key="bar_prev", help="Previous track", use_container_width=True, disabled=(queue_idx == 0)):
                    queue_play_previous()
                    st.rerun()
            with bar_next:
                if st.button("⏭️", key="bar_next", help="Next track", use_container_width=True, disabled=(queue_idx == len(vault) - 1)):
                    queue_play_next()
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    elif vault:
        st.caption("💡 Tip: click ▶️ on any saved track below to start your queue, then use the Previous/Next controls on the player.")

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
                is_current = st.session_state.get("current_queue_index") == i
                r1, r2, r3, r4, r5 = st.columns([3.6, 0.7, 0.55, 0.55, 0.7])
                with r1:
                    link = track.get("youtube_url", "")
                    link_html = f'<a href="{link}" target="_blank" style="color:#6EE7B7; font-size:0.78rem; text-decoration:none;">🔗 YouTube</a>' if link else ""
                    now_playing_badge = (
                        ' <span class="tf-badge" style="margin-bottom:0;">▶ Playing</span>' if is_current else ""
                    )
                    row_class = "tf-vault-row tf-vault-row-active" if is_current else "tf-vault-row"
                    st.markdown(
                        f"""
                        <div class="{row_class}">
                            <div>
                                <span class="tf-vault-title">{track.get('Song','')}</span>{now_playing_badge}<br>
                                <span class="tf-vault-artist">{track.get('Artist','')}</span> {link_html}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with r2:
                    if st.button("▶️", key=f"vault_play_{i}", help="Play this track"):
                        load_queue_track(i)
                        st.toast("Switched preview — check the Discover tab. 🎧")
                        st.rerun()
                with r3:
                    if st.button("🔼", key=f"vault_up_{i}", help="Move up", disabled=(i == 0)):
                        move_track_in_vault(i, -1)
                        st.rerun()
                with r4:
                    if st.button("🔽", key=f"vault_down_{i}", help="Move down", disabled=(i == len(vault) - 1)):
                        move_track_in_vault(i, 1)
                        st.rerun()
                with r5:
                    if st.button("🗑️", key=f"vault_remove_{i}", help="Remove from vault"):
                        remove_from_playlist(i)
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------- PLAYLIST DOWNLOAD, RESUME & MANAGE --------------------
    with col_export:
        st.markdown('<div class="tf-card">', unsafe_allow_html=True)
        st.markdown('<div class="tf-card-title">💾 Save &amp; Resume Later</div>', unsafe_allow_html=True)
        st.caption("No account needed — download your playlist now, and upload it next time to pick up right where you left off.")

        if vault:
            json_bytes = playlist_to_json_bytes()
            st.download_button(
                label="⬇️ Download Playlist",
                data=json_bytes,
                file_name="trackfind_playlist.json",
                mime="application/json",
                use_container_width=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

        uploaded_vault = st.file_uploader(
            "Resume a saved playlist",
            label_visibility="collapsed",
            key="vault_resume_uploader",
        )
        if uploaded_vault is not None:
            already_processed = st.session_state.get("_last_restored_upload_id")
            upload_id = f"{uploaded_vault.name}_{uploaded_vault.size}"
            if already_processed != upload_id:
                success, message = restore_playlist_from_file(uploaded_vault.read(), uploaded_vault.name)
                st.session_state["_last_restored_upload_id"] = upload_id
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"⚠️ {message}")

        if vault:
            st.markdown("<br>", unsafe_allow_html=True)
            # [UX] Single button, ask-on-click instead of a checkbox +
            # permanently-disabled button. First click arms a confirmation
            # prompt; a second click on the confirm button actually clears.
            # Clicking anything else cancels the pending confirmation.
            if st.session_state.get("confirm_clear_pending"):
                st.warning("Clear your entire playlist? This can't be undone.")
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("✅ Yes, clear it", use_container_width=True, key="clear_playlist_confirm_yes"):
                        st.session_state.playlist_vault = []
                        st.session_state.confirm_clear_pending = False
                        st.toast("Playlist cleared. Fresh start! 🌱")
                        st.rerun()
                with cc2:
                    if st.button("Cancel", use_container_width=True, key="clear_playlist_confirm_no"):
                        st.session_state.confirm_clear_pending = False
                        st.rerun()
            else:
                if st.button("🧹 Clear Playlist", use_container_width=True, key="clear_playlist_btn"):
                    st.session_state.confirm_clear_pending = True
                    st.rerun()
        elif not vault:
            st.caption("Add tracks to your playlist to unlock download options.")

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
