"""
YouTube Video Summarizer — Streamlit + Groq
============================================
Paste a YouTube link, pull its transcript, and get an AI summary via the
Groq API (fast, cheap, no local model download).

Run locally:
    pip install -r requirements.txt
    # add your key to .streamlit/secrets.toml  (see README)
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
import textwrap
from urllib.parse import urlparse, parse_qs

import streamlit as st


# --------------------------------------------------------------------------- #
#  Page config + styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="YouTube Summarizer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(1200px 600px at 10% -10%, rgba(255,0,80,.08), transparent 60%),
            radial-gradient(1000px 500px at 100% 0%, rgba(120,80,255,.10), transparent 55%);
    }
    #MainMenu, footer {visibility: hidden;}

    .hero { text-align: center; padding: 2.2rem 1rem 1.4rem 1rem; }
    .hero h1 {
        font-size: 2.7rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: .35rem;
        background: linear-gradient(90deg, #ff2d55, #ff6b6b 40%, #a06bff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .hero p { font-size: 1.05rem; color: #8a8f98; margin: 0 auto; max-width: 620px; }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
        padding: 1rem 1.1rem; border-radius: 16px; backdrop-filter: blur(6px);
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] p { color: #8a8f98; }

    .summary-card {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
        border-radius: 18px; padding: 1.6rem 1.8rem; line-height: 1.7; font-size: 1.03rem;
    }

    .stButton > button {
        border-radius: 12px; font-weight: 600; padding: .55rem 1.1rem;
        border: 1px solid rgba(255,255,255,0.12);
        transition: transform .06s ease, box-shadow .2s ease;
    }
    .stButton > button[kind="primary"] { background: linear-gradient(90deg, #ff2d55, #ff5e62); border: none; }
    .stButton > button:hover { transform: translateY(-1px); }

    .stTextInput > div > div > input { border-radius: 12px; padding: .7rem .9rem; }

    .badge {
        display: inline-block; padding: .2rem .7rem; border-radius: 999px;
        font-size: .78rem; font-weight: 600; background: rgba(160,107,255,.15);
        color: #b79bff; border: 1px solid rgba(160,107,255,.3);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
# Current Groq production models (the old llama-3.x chat models were deprecated
# on 2026-06-17). Verify the live list at https://console.groq.com/docs/models
GROQ_MODELS = {
    "GPT-OSS 20B — fast & cheap (recommended)": "openai/gpt-oss-20b",
    "GPT-OSS 120B — higher quality": "openai/gpt-oss-120b",
}

# Roughly how many words to send per LLM call before we map-reduce.
CHUNK_WORDS = 6000


def get_api_key() -> str | None:
    """Look for the key in Streamlit secrets, then the environment."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


# --------------------------------------------------------------------------- #
#  YouTube helpers
# --------------------------------------------------------------------------- #
def extract_video_id(url: str) -> str:
    """Extract a YouTube video ID from many URL shapes (watch, youtu.be,
    shorts, embed, live) or a bare 11-char ID. Raises ValueError otherwise."""
    url = (url or "").strip()
    if not url:
        raise ValueError("Please enter a URL.")

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    if "://" not in url:
        url = "https://" + url

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().replace("www.", "")

    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
        if candidate:
            return candidate

    if "youtube.com" in host:
        v = parse_qs(parsed.query).get("v")
        if v and v[0]:
            return v[0]
        m = re.match(r"^/(?:embed|shorts|live|v)/([A-Za-z0-9_-]{11})", parsed.path)
        if m:
            return m.group(1)

    raise ValueError(f"Couldn't find a video ID in: {url}")


@st.cache_data(show_spinner=False)
def fetch_transcript(video_id: str, preferred_langs: tuple[str, ...]):
    """Fetch a transcript, returning (full_text, language_code). Tries the
    preferred languages first, then falls back to any available transcript."""
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()

    try:
        fetched = api.fetch(video_id, languages=list(preferred_langs))
        text = " ".join(s.text for s in fetched).strip()
        lang = getattr(fetched, "language_code", preferred_langs[0])
        if text:
            return text, lang
    except Exception:
        pass

    transcripts = api.list(video_id)
    for t in transcripts:
        try:
            data = t.fetch()
            text = " ".join(s.text for s in data).strip()
            if text:
                return text, t.language_code
        except Exception:
            continue

    raise RuntimeError("No usable transcript could be retrieved for this video.")


# --------------------------------------------------------------------------- #
#  Summarization via Groq
# --------------------------------------------------------------------------- #
STYLE_INSTRUCTIONS = {
    "Short": "Write a short summary of 3-5 sentences capturing only the main point.",
    "Medium": "Write a clear summary of 2-3 short paragraphs covering the key points.",
    "Detailed": ("Write a thorough, well-structured summary that covers all the "
                 "important points, arguments, and conclusions. Use short paragraphs."),
}


def _chunk_words(text: str, size: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def _call_groq(client, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def summarize(api_key: str, model: str, text: str, style: str,
              bullets: bool, progress=None) -> str:
    """Summarize a transcript. Long transcripts are map-reduced: each chunk is
    summarized, then the chunk-summaries are combined into one final summary."""
    from groq import Groq

    client = Groq(api_key=api_key)
    system = ("You are an expert assistant that summarizes YouTube video "
              "transcripts into clear, faithful summaries. Do not invent facts.")
    style_line = STYLE_INSTRUCTIONS[style]
    if bullets:
        style_line += " Format the summary as concise bullet points."

    chunks = _chunk_words(text, CHUNK_WORDS)

    # Single-shot for normal-length videos.
    if len(chunks) == 1:
        out = _call_groq(
            client, model, system,
            f"{style_line}\n\nTranscript:\n\"\"\"\n{text}\n\"\"\"",
        )
        if progress:
            progress(1.0)
        return out

    # Map: summarize each chunk.
    partials = []
    for i, chunk in enumerate(chunks):
        part = _call_groq(
            client, model, system,
            "Summarize this part of a longer video transcript, keeping the "
            f"important details:\n\n\"\"\"\n{chunk}\n\"\"\"",
        )
        partials.append(part)
        if progress:
            progress((i + 1) / (len(chunks) + 1))

    # Reduce: combine the partial summaries.
    combined = "\n\n".join(partials)
    final = _call_groq(
        client, model, system,
        f"{style_line}\n\nBelow are summaries of consecutive parts of one "
        f"video. Combine them into a single coherent summary:\n\n{combined}",
    )
    if progress:
        progress(1.0)
    return final


# --------------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------------- #
def word_count(text: str) -> int:
    return len(text.split())


def reading_time(text: str) -> str:
    return f"{max(1, round(word_count(text) / 200))} min"


# --------------------------------------------------------------------------- #
#  Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    model_label = st.selectbox("Model", list(GROQ_MODELS.keys()), index=0)
    model = GROQ_MODELS[model_label]

    lang_choice = st.selectbox(
        "Preferred transcript language",
        ["English (en)", "Arabic (ar)", "Spanish (es)",
         "French (fr)", "German (de)", "Auto (any)"],
        index=0,
        help="Tried first. Falls back to any available transcript.",
    )
    lang_map = {
        "English (en)": ("en",),
        "Arabic (ar)": ("ar", "en"),
        "Spanish (es)": ("es", "en"),
        "French (fr)": ("fr", "en"),
        "German (de)": ("de", "en"),
        "Auto (any)": ("en", "ar", "es", "fr", "de"),
    }
    preferred_langs = lang_map[lang_choice]

    detail = st.select_slider("Detail level",
                              ["Short", "Medium", "Detailed"], value="Detailed")
    bullets = st.toggle("Bullet points", value=False)

    st.divider()

    key_in_config = get_api_key()
    if key_in_config:
        st.success("Groq API key loaded ✅")
        api_key = key_in_config
    else:
        st.warning("No Groq API key found in secrets/env.")
        api_key = st.text_input("Groq API key", type="password",
                                placeholder="gsk_…",
                                help="Get one free at console.groq.com/keys")

    st.caption("Powered by Groq. Free tier is enough for personal use — "
               "no local model download.")


# --------------------------------------------------------------------------- #
#  Header + input
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
        <h1>🎬 YouTube Video Summarizer</h1>
        <p>Paste a link, pull the transcript, and get a clean AI summary —
        no need to watch the whole thing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_in, col_btn = st.columns([5, 1], vertical_alignment="bottom")
with col_in:
    url = st.text_input("YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=…",
                        label_visibility="collapsed")
with col_btn:
    go = st.button("Summarize", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- #
#  Run
# --------------------------------------------------------------------------- #
if "result" not in st.session_state:
    st.session_state.result = None

if go:
    if not api_key:
        st.error("❌ Add your Groq API key in the sidebar first.")
        st.stop()

    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    pl, pr = st.columns([1, 2], vertical_alignment="center")
    with pl:
        st.image(f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
                 use_container_width=True)
    with pr:
        st.markdown(f'<span class="badge">Video ID · {video_id}</span>',
                    unsafe_allow_html=True)
        st.markdown(f"[Open on YouTube ↗](https://youtu.be/{video_id})")

    try:
        with st.status("Fetching transcript…", expanded=False) as status:
            transcript, used_lang = fetch_transcript(video_id, preferred_langs)
            status.update(
                label=f"Transcript fetched ({used_lang}, {word_count(transcript):,} words)",
                state="complete")
    except Exception as e:
        st.error("❌ Couldn't get a transcript. The video may have captions "
                 "disabled, be private/age-restricted, or lack subtitles in "
                 f"the chosen language.\n\n*Details: {e}*")
        st.stop()

    bar = st.progress(0.0, text="Summarizing with Groq…")
    try:
        summary = summarize(api_key, model, transcript, detail, bullets,
                            progress=lambda f: bar.progress(f, text="Summarizing with Groq…"))
    except Exception as e:
        st.error(f"❌ Summarization failed: {e}\n\n"
                 "Check that your API key is valid and the selected model is "
                 "still available at console.groq.com/docs/models.")
        st.stop()
    finally:
        bar.empty()

    st.session_state.result = {"video_id": video_id, "summary": summary,
                               "transcript": transcript, "lang": used_lang}


# --------------------------------------------------------------------------- #
#  Results
# --------------------------------------------------------------------------- #
res = st.session_state.result
if res:
    summary, transcript = res["summary"], res["transcript"]

    st.markdown("### 📊 At a glance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transcript words", f"{word_count(transcript):,}")
    m2.metric("Summary words", f"{word_count(summary):,}")
    ratio = word_count(summary) / max(1, word_count(transcript))
    m3.metric("Condensed to", f"{ratio*100:.0f}%")
    m4.metric("Summary read time", reading_time(summary))

    st.write("")
    tab_sum, tab_trans = st.tabs(["✨ Summary", "📜 Full transcript"])
    with tab_sum:
        st.markdown(f'<div class="summary-card">{summary}</div>',
                    unsafe_allow_html=True)
        st.write("")
        st.download_button("⬇️ Download summary (.txt)", data=summary,
                           file_name=f"summary_{res['video_id']}.txt",
                           mime="text/plain")
    with tab_trans:
        st.text_area("Transcript", transcript, height=420,
                     label_visibility="collapsed")
        st.download_button("⬇️ Download transcript (.txt)", data=transcript,
                           file_name=f"transcript_{res['video_id']}.txt",
                           mime="text/plain")
else:
    st.info("👆 Paste a YouTube link above and hit **Summarize** to get started.")
