# 🎬 YouTube Video Summarizer (Streamlit + Groq)

Paste a YouTube link, pull its transcript, and get an AI summary via the Groq API.
No local ML model — total install is under ~50 MB.

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
```

## Add your Groq API key

1. Get a free key at https://console.groq.com/keys
2. Open `.streamlit/secrets.toml` and paste it in:

   ```toml
   GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
   ```

   (Or set an env var `GROQ_API_KEY`, or just paste it into the sidebar when the app runs.)

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501

## Notes
- Models: `openai/gpt-oss-20b` (default, fast) or `openai/gpt-oss-120b`.
  Verify the current list at https://console.groq.com/docs/models
- Long transcripts are automatically split and combined (map-reduce).
- `youtube-transcript-api` can be rate-limited by YouTube; if a video with
  captions errors out, wait and retry.
