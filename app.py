"""
TrackFind — Your Personal AI Music Curator
============================================
A premium, dark-themed Streamlit music recommendation app powered by the
Google Gen AI SDK (Gemini).

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os
import re
import csv
import io
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Google Gen AI SDK & Validation Models
# ---------------------------------------------------------------------------
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ===========================================================================
# 1. STRUCTURAL GLOBAL VARIABLES & CONSTANTS (DEFINED FIRST)
# ===========================================================================

APP_TITLE = "🎵 TrackFind — Your Personal AI Music Curator"
GEMINI_MODEL = "gemini-2.5-flash"

GEMINI_MODEL_FALLBACKS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]

GEMINI_API_KEY_PLACEHOLDER = "YOUR_GEMINI_API_KEY_HERE"
YOUTUBE_API_KEY_PLACEHOLDER = "YOUR_YOUTUBE_API_KEY_HERE"

YOUTUBE_ID_PATTERNS = [
    r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/shorts\/)([A-Za-z0-9_-]{11})",
]

FLUFF_PATTERNS = [
    r"\(\s*official\s*video\s*\)", r"\[\s*official\s*video\s*\]",
    r"\(\s*official\s*audio\s*\)", r"\[\s*official\s*audio\s*\]",
    r"\(\s*official\s*music\s*video\s*\)", r"\[\s*official\s*music\s*video\s*\]",
    r"\(\s*official\s*lyric\s*video\s*\)", r"\[\s*official\s*lyric\s*video\s*\]",
    r"\(\s*lyrics?\s*\)", r"\[\s*lyrics?\s*\]",
    r"\(\s*lyric\s*video\s*\)", r"\[\s*lyric\s*video\s*\]",
    r"\(\s*audio\s*\)", r"\[\s*audio\s*\]",
    r"\(\s*visualizer\s*\)", r"\[\s*visualizer\s*\]",
    r"\(\s*hd\s*\)", r"\[\s*hd\s*\]", r"\(\s*4k\s*\)", r"\[\s*4k\s*\]", r"\(\s*hq\s*\)", r"\[\s*hq\s*\]",
    r"\(\s*full\s*video\s*\)", r"\[\s*full\s*video\s*\]",
    r"\(\s*full\s*song\s*\)", r"\[\s*full\s*song\s*\]",
    r"\bofficial\s*video\b", r"\bofficial\s*audio\b",
    r"\bofficial\s*music\s*video\b", r"\bofficial\s*lyric\s*video\b",
    r"\bmusic\s*video\b", r"\blyric\s*video\b", r"\blyrics\b",
    r"\bvisualizer\b", r"\b4k\b", r"\bhd\b", r"\bhq\b", r"\bfull\s*video\b",
    r"\bremastered\b", r"\bclean\s*version\b", r"\bexplicit\s*version\b",
    r"\(\s*explicit\s*\)", r"\[\s*explicit\s*\]",
]

FLUFF_REGEX = re.compile("|".join(FLUFF_PATTERNS), flags=re.IGNORECASE)

NOISE_SEGMENT_PATTERNS = [
    r"^(latest|new|hit|top|best)?\s*(punjabi|hindi|bollywood|english|haryanvi)?\s*song(s)?\s*\d*$",
    r"^official\s*(video|audio)?$",
]
NOISE_SEGMENT_REGEX = re.compile("|".join(NOISE_SEGMENT_PATTERNS), flags=re.IGNORECASE)


# ===========================================================================
# 2. DATA MODELS
# ===========================================================================

class RecommendedTrack(BaseModel):
    Song: str = Field(description="The title of the recommended track.")
    Artist: str = Field(description="The artist name.")
    Reason: str = Field(description="A short, clear explanation of WHY this song was chosen.")


class RecommendationList(BaseModel):
    recommendations: List[RecommendedTrack]


class ParsedTitle(BaseModel):
    artist: str = Field(description="The primary recording singer/artist's name only — a person or group, never a movie/film/album name, no featured artists, no promotional text.")
    title: str = Field(description="The actual standalone song title only — never a movie/film/album name, never a language or genre tag, no promotional text.")


@dataclass
class SeedTrack:
    artist: str = ""
    title: str = ""
    raw_source: str = ""   
    video_id: str = ""     
    parse_confidence: str = "high" 


# ===========================================================================
# 3. INTERFACE FORMATTING & DESIGN HELPERS
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


# ===========================================================================
# 4. RESOLUTION PIPELINES (API & CORE DATA KEYS)
# ===========================================================================

def get_api_key() -> Optional[str]:
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


def get_youtube_api_key() -> Optional[str]:
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


def get_genai_client() -> Optional[genai.Client]:
    api_key = get_api_key()
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


# ===========================================================================
# 5. PARSING, METADATA ISOLATION, & CONTEXTUAL FILTERING
# ===========================================================================

def extract_youtube_id(url: str) -> Optional[str]:
    for pattern in YOUTUBE_ID_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_youtube_title(video_id: str) -> Optional[str]:
    try:
        import urllib.request
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        with urllib.request.urlopen(oembed_url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("title")
    except Exception:
        return None


def clean_youtube_title(raw_title: str) -> str:
    if not raw_title:
        return ""
    cleaned = FLUFF_REGEX.sub("", raw_title)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\[\s*\]", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s*[-|]\s*$", "", cleaned)
    cleaned = re.sub(r"^\s*[-|]\s*", "", cleaned)
    cleaned = cleaned.strip(" -|·•").strip()
    return cleaned


def _is_structurally_complex(cleaned_title: str) -> bool:
    pipe_count = cleaned_title.count("|")
    dash_count = len(re.findall(r"\s[-–—]\s", cleaned_title))
    total_separators = pipe_count + dash_count
    return pipe_count >= 2 or (pipe_count >= 1 and dash_count >= 1) or total_separators >= 3


def split_artist_title_regex(cleaned_title: str) -> SeedTrack:
    if not cleaned_title:
        return SeedTrack(artist="", title="", parse_confidence="high")

    if not _is_structurally_complex(cleaned_title):
        for sep_pattern in [r"\s+[-–—]\s+", r"\s*\|\s*"]:
            match = re.search(sep_pattern, cleaned_title)
            if match:
                idx = match.start()
                artist = cleaned_title[:idx].strip()
                title = cleaned_title[match.end():].strip()
                if artist and title:
                    return SeedTrack(artist=artist, title=title, parse_confidence="high")
        return SeedTrack(artist="", title=cleaned_title.strip(), parse_confidence="high")

    segments = re.split(r"\s*\|\s*|\s+[-–—]\s+", cleaned_title)
    segments = [s.strip() for s in segments if s.strip()]

    if len(segments) <= 1:
        return SeedTrack(artist="", title=cleaned_title.strip(), parse_confidence="low")

    meaningful = [s for s in segments if not NOISE_SEGMENT_REGEX.match(s)]
    if not meaningful:
        meaningful = segments

    title_guess = meaningful[-1].strip()
    artist_segment = meaningful[0].strip()
    artist_guess = artist_segment.split(",")[0].strip() if "," in artist_segment else artist_segment

    return SeedTrack(artist=artist_guess, title=title_guess, parse_confidence="low")


def refine_title_with_gemini(messy_title: str) -> Optional[ParsedTitle]:
    client = get_genai_client()
    if client is None:
        return None

    prompt = f"""
You are a metadata extraction specialist for music video titles, especially
Bollywood, Punjabi, and other South Asian regional film/music uploads where
the singer, movie/film name, genre tag, and song title are often mixed
together in inconsistent order with "|" or "-"/"—" as separators.

Raw title: "{messy_title}"

STEP 1 — Check for explicit key-value labels first.
If the text contains explicit indicators such as "Song:", "Track:",
"Singer:", "Artist:", "Movie:", or "Film:", these are AUTHORITATIVE — use the
value following "Song:" or "Track:" as the title, and the value following
"Singer:" or "Artist:" as the artist, regardless of where they appear in the
string. Ignore the value following "Movie:" or "Film:" entirely — that is
never the artist or the title.

STEP 2 — If there are no explicit labels, use structural reasoning instead:
- The SONG TITLE is almost always the LAST segment in the string, especially
  if it immediately follows a dash ("-", "–", or "—") near the end.
- The PRIMARY SINGER/ARTIST is almost always the FIRST segment. If that
  segment lists multiple names separated by a comma, the first name is
  usually the singer and the remaining names are usually featured artists or
  film actors — prefer the first name as the primary artist.
- A SHORT middle segment that is just one or two words and reads like a
  single proper noun (e.g. a short film/movie name like 'Sargi') is almost always the
  MOVIE or ALBUM name, NOT the song title — even though its brevity might
  make it look like a plausible title. Movie/album names must never be
  returned as either the artist or the title.
- A segment containing generic filler like "Latest", "New", "Punjabi Song",
  "Hindi Song", "Bollywood", "Official", "Full Video", "Full Song", or a
  bare year is pure noise — discard it; it is never the artist or the title.

STEP 3 — Self-check before answering:
- Is the artist you chose an actual person or group name? If it looks like a
  movie/film/album name instead, you picked the wrong segment — go back and
  find the real singer, usually the first segment.
- Is the title you chose the actual song name? If it looks like a short
  film/movie name or a generic genre/language tag instead, you picked the
  wrong segment — go back and find the real song title, usually the last
  segment, often right after a dash.

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

    if seed.parse_confidence == "low" and allow_gemini_refine and get_api_key():
        refined = refine_title_with_gemini(cleaned)
        if refined and refined.artist and refined.title:
            seed.artist = refined.artist
            seed.title = refined.title
            seed.parse_confidence = "refined"

    return seed


def build_music_search_query(typed_artist: str, typed_title: str) -> str:
    artist_clean = (typed_artist or "").strip()
    title_clean = (typed_title or "").strip()

    if artist_clean and title_clean:
        return f"{artist_clean} {title_clean} song"
    if artist_clean and not title_clean:
        return f"{artist_clean} official audio music song"
    if title_clean and not artist_clean:
        return f"{title_clean} official audio music song"
    return ""


def build_youtube_search_url(query: str) -> str:
    import urllib.parse
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"


def build_youtube_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}" if video_id else ""


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def search_youtube_tracks(query: str, api_key: str, max_results: int = 5) -> List[dict]:
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


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def search_youtube_video_id(query: str, api_key: str) -> Optional[str]:
    results = search_youtube_tracks(query, api_key, max_results=1)
    return results[0]["video_id"] if results else None


def build_recommendation_prompt(artist: str, title: str, num_recs: int) -> str:
    return f"""
You are TrackFind, an expert AI music curator with encyclopedic knowledge of songs, artists, genres, eras, moods, and musical structure.

A user has provided this seed track:
    Artist: "{artist or 'Unknown'}"
    Title:  "{title or 'Unknown'}"

Recommend exactly {num_recs} songs that a fan of this track would genuinely enjoy.

Rules:
- Do NOT include the seed track itself in the results.
- Do NOT repeat the same song twice.
- Each "Reason" must be ONE short, specific sentence (under 15 words).
- Favor real, well-known, verifiable songs and artists.
- Return ONLY the structured data — no preamble, no extra commentary.
""".strip()


def _call_gemini_model(client: "genai.Client", model_name: str, prompt: str) -> List[dict]:
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
    parsed = getattr(response, "parsed", None)
    if parsed:
        return [item.model_dump() if hasattr(item, "model_dump") else dict(item) for item in parsed]

    raw_text = response.text or "[]"
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(json)?", "", raw_text).rstrip("`").strip()
    return json.loads(raw_text)


def get_recommendations(artist: str, title: str, num_recs: int) -> List[dict]:
    client = get_genai_client()
    if client is None:
        raise RuntimeError("Gemini API key not configured. Add GEMINI_API_KEY to secrets.")

    prompt = build_recommendation_prompt(artist, title, num_recs)

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
            if "404" not in error_str and "NOT_FOUND" not in error_str.upper():
                raise

    raise RuntimeError(f"None of the configured Gemini models are available. Last error: {last_error}")


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
# 6. APP RENDERING LAYOUT ENGINE (EXECUTES LAST CHRONOLOGICALLY)
# ===========================================================================

# Inject customized visual theme container definitions
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Render main app header dashboard elements
render_hero()
tab_discover, tab_vault = st.tabs(["🔎  Discover & Sync", "🎧  My Playlist Vault"])

with tab_discover:
    col_input, col_player = st.columns([1.35, 1], gap="large")

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

        if st.session_state.get("_last_input_mode") != input_mode:
            st.session_state.confirmed_seed = None
            st.session_state.search_candidates = []
            st.session_state.search_performed = False
            st.session_state["_last_input_mode"] = input_mode

        seed_artist, seed_title, seed_video_id = "", "", ""

        if input_mode == "🎤 Artist + Track":
            c1, c2 = st.columns(2)
            with c1: typed_artist = st.text_input("Artist Name", placeholder="e.g. Diljit Dosanjh", key="typed_artist")
            with c2: typed_title = st.text_input("Track Title", placeholder="e.g. Lover", key="typed_title")

            st.markdown('<div class="tf-search-btn">', unsafe_allow_html=True)
            search_clicked = st.button("🔍 Search Track", use_container_width=True, key="search_track_btn")
            st.markdown('</div>', unsafe_allow_html=True)

            yt_key = get_youtube_api_key()

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
                        st.session_state.search_candidates = []

            search_performed = st.session_state.get("search_performed", False)

            if st.session_state.search_candidates:
                display_query = st.session_state.get("last_search_display_query") or st.session_state.last_search_query
                st.caption(f"Top matches for **{display_query}** — confirm the exact track:")
                option_labels = [f"{cand['title']}  ·  {cand['channel']}" for cand in st.session_state.search_candidates]

                chosen_label = st.radio(
                    "Select the exact track",
                    options=option_labels,
                    label_visibility="collapsed",
                    key="candidate_radio",
                )
                chosen_idx = option_labels.index(chosen_label) if chosen_label in option_labels else 0
                chosen = st.session_state.search_candidates[chosen_idx]

                st.markdown(f'<div class="tf-candidate"><img src="{chosen["thumbnail"]}"><div><p class="tf-candidate-title">{chosen["title"]}</p><p class="tf-candidate-channel">{chosen["channel"]}</p></div></div>', unsafe_allow_html=True)

                if st.button("✅ Confirm This Track", use_container_width=True, key="confirm_candidate_btn"):
                    cleaned = clean_youtube_title(chosen["title"])
                    guess = split_artist_title_regex(cleaned)
                    confirmed_artist = guess.artist or typed_artist
                    confirmed_title = guess.title or typed_title or cleaned
                    st.session_state.confirmed_seed = {"artist": confirmed_artist, "title": confirmed_title, "video_id": chosen["video_id"]}
                    st.session_state.now_playing = {"Song": confirmed_title, "Artist": confirmed_artist, "video_id": chosen["video_id"]}
                    st.toast(f"Confirmed: {confirmed_title} — {confirmed_artist} ✅", icon="🎯")
                    st.rerun()

            if st.session_state.confirmed_seed:
                cs = st.session_state.confirmed_seed
                st.success(f"🎯 Locked in: **{cs['title']}** — *{cs['artist']}*")
                seed_artist, seed_title, seed_video_id = cs["artist"], cs["title"], cs["video_id"]
            else:
                seed_artist, seed_title = typed_artist, typed_title

        else:
            yt_url = st.text_input("YouTube Link", placeholder="https://www.youtube.com/watch?v=...", key="yt_url_input")

            if yt_url:
                if st.session_state.get("_last_parsed_url") != yt_url:
                    with st.spinner("Parsing YouTube link..."):
                        seed = parse_youtube_link(yt_url)
                    st.session_state["_last_parsed_seed"] = asdict(seed)
                    st.session_state["_last_parsed_url"] = yt_url

                    if seed.video_id:
                        st.session_state.now_playing = {"Song": seed.title, "Artist": seed.artist, "video_id": seed.video_id}

                parsed = st.session_state.get("_last_parsed_seed", {})
                seed_video_id = parsed.get("video_id", "")
                parse_confidence = parsed.get("parse_confidence", "high")

                if parsed.get("title"):
                    seed_artist_default = parsed.get("artist", "")
                    seed_title_default = parsed.get("title", "")

                    confidence_badge = (
                        '<span class="tf-confidence-pill tf-confidence-refined">✨ Gemini-refined</span>'
                        if parse_confidence == "refined"
                        else '<span class="tf-confidence-pill tf-confidence-high">Auto-detected</span>'
                    )
                    st.markdown(f"✅ Detected: **{seed_title_default}** — *{seed_artist_default or 'Unknown artist'}* {confidence_badge}", unsafe_allow_html=True)

                    seed_artist = st.text_input("Confirm / edit Artist", value=seed_artist_default, key="yt_artist_confirm")
                    seed_title = st.text_input("Confirm / edit Track Title", value=seed_title_default, key="yt_title_confirm")

                    if st.session_state.now_playing and st.session_state.now_playing.get("video_id") == seed_video_id:
                        st.session_state.now_playing["Song"] = seed_title
                        st.session_state.now_playing["Artist"] = seed_artist

        st.markdown("<hr class='tf-divider'>", unsafe_allow_html=True)
        st.markdown('<div class="tf-card-title">🎛️ Recommendation Controls</div>', unsafe_allow_html=True)
        num_recs = st.slider("How many recommendations do you want?", 5, 50, 10, step=1, key="recs_slider")

        st.markdown(f"""<span class="tf-stat-chip">{num_recs}<span>tracks requested</span></span>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="tf-primary-btn">', unsafe_allow_html=True)
        generate_clicked = st.button("✨ Generate Recommendations", use_container_width=True, key="generate_btn")
        st.markdown('</div></div>', unsafe_allow_html=True)

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
                    st.session_state.now_playing = {"Song": seed_title, "Artist": seed_artist, "video_id": seed_video_id}
                    st.rerun()
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

    with col_player:
        st.markdown('<div class="tf-card">', unsafe_allow_html=True)
        st.markdown('<div class="tf-card-title">▶️ Now Sampling</div>', unsafe_allow_html=True)

        now_playing = st.session_state.now_playing

        if now_playing and (now_playing.get("Song") or now_playing.get("Artist")):
            song = now_playing.get("Song", "")
            artist = now_playing.get("Artist", "")
            video_id = now_playing.get("video_id", "")

            st.markdown(f"**{song}**")
            st.markdown(f"<span class='tf-subtle'>{artist}</span>", unsafe_allow_html=True)

            if not video_id:
                yt_key = get_youtube_api_key()
                if yt_key:
                    query = f"{artist} {song}".strip()
                    if query:
                        with st.spinner("Finding video..."):
                            resolved_id = search_youtube_video_id(query, yt_key)
                        if resolved_id:
                            video_id = resolved_id
                            st.session_state.now_playing["video_id"] = resolved_id

            if video_id:
                st.video(build_youtube_watch_url(video_id))
                st.caption("🔊 Now playing.")
            else:
                query = f"{artist} {song}".strip()
                search_url = build_youtube_search_url(query) if query else ""
                st.markdown('<div class="tf-empty-state" style="padding:1.6rem 1rem;"><span class="tf-emoji">🔎</span>No direct video found for this track yet.</div>', unsafe_allow_html=True)
                if search_url:
                    st.link_button("🔗 Find & play on YouTube", search_url, use_container_width=True)
        else:
            st.markdown('<div class="tf-empty-state"><span class="tf-emoji">🎶</span>Generate recommendations or pick a track below to preview it here.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

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
                    st.markdown(f'<div class="tf-track"><span class="tf-badge">#{idx + 1}</span><p class="tf-track-song">{song}</p><p class="tf-track-artist">{artist}</p><p class="tf-track-reason">💡 {reason}</p></div>', unsafe_allow_html=True)
                with c_play:
                    if st.button("▶️ Play", key=f"play_{idx}"):
                        st.session_state.now_playing = {"Song": song, "Artist": artist, "video_id": ""}
                        st.rerun()
                with c_add:
                    if st.button("➕ Add to Playlist", key=f"add_{idx}"):
                        add_to_playlist({"Song": song, "Artist": artist, "Reason": reason})
    elif st.session_state.has_generated:
        st.info("No recommendations were returned. Try a different seed track.")
    else:
        st.markdown('<div class="tf-empty-state"><span class="tf-emoji">🧭</span>Your recommendations will appear here once you generate them above.</div>', unsafe_allow_html=True)

with tab_vault:
    vault = st.session_state.playlist_vault

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""<span class="tf-stat-chip">{len(vault)}<span>saved tracks</span></span>""", unsafe_allow_html=True)
    with col_b:
        unique_artists = len({t["Artist"] for t in vault}) if vault else 0
        st.markdown(f"""<span class="tf-stat-chip">{unique_artists}<span>unique artists</span></span>""", unsafe_allow_html=True)

    st.markdown("<hr class='tf-divider'>", unsafe_allow_html=True)
    col_list, col_export = st.columns([1.4, 1], gap="large")

    with col_list:
        st.markdown('<div class="tf-card">', unsafe_allow_html=True)
        st.markdown('<div class="tf-card-title">💾 Saved Tracks</div>', unsafe_allow_html=True)

        if not vault:
            st.markdown('<div class="tf-empty-state"><span class="tf-emoji">📭</span>Your vault is empty. Head to "Discover &amp; Sync" and add some tracks!</div>', unsafe_allow_html=True)
        else:
            for i, track in enumerate(vault):
                r1, r2, r3 = st.columns([4.2, 0.9, 0.9])
                with r1:
                    st.markdown(f'<div class="tf-vault-row"><div><span class="tf-vault-title">{track.get("Song","")}</span><br><span class="tf-vault-artist">{track.get("Artist","")}</span></div></div>', unsafe_allow_html=True)
                with r2:
                    if st.button("▶️", key=f"vault_play_{i}", help="Preview this track"):
                        st.session_state.now_playing = {"Song": track.get("Song", ""), "Artist": track.get("Artist", ""), "video_id": ""}
                        st.toast("Switched preview — check the Discover tab. 🎧")
                with r3:
                    if st.button("🗑️", key=f"vault_remove_{i}", help="Remove from vault"):
                        st.session_state.playlist_vault.pop(i)
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_export:
        st.markdown('<div class="tf-card">', unsafe_allow_html=True)
        st.markdown('<div class="tf-card-title">📤 Export &amp; Manage</div>', unsafe_allow_html=True)

        if vault:
            st.download_button(label="⬇️ Download CSV", data=playlist_to_csv_bytes(), file_name="trackfind_playlist.csv", mime="text/csv", use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Copy-paste tracklist:")
            st.text_area("Markdown tracklist", value=playlist_to_markdown(), height=220, label_visibility="collapsed")
            st.markdown("<br>", unsafe_allow_html=True)
            confirm_clear = st.checkbox("Confirm: I want to clear my playlist")
            if st.button("🧹 Clear Playlist", use_container_width=True, disabled=not confirm_clear):
                st.session_state.playlist_vault = []
                st.toast("Playlist cleared. Fresh start! 🌱")
                st.rerun()
        else:
            st.caption("Add tracks to your vault to unlock export options.")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center; padding: 1.5rem 0 0.5rem 0; color:#5C7A73; font-size:0.78rem;">Built with Streamlit &amp; Google Gemini · TrackFind © 2026</div>', unsafe_allow_html=True)
