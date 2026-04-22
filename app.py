"""
PopcornPicks — Popcorn-themed web UI for the agentic movie recommender.

Run locally:
    OLLAMA_API_KEY=your_key uvicorn app:app --reload --port 8080

Deployed on Leapcell via leapcell.yaml.
"""

import os
from typing import List

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from llm import TOP_MOVIES, get_recommendation

app = FastAPI(title="PopcornPicks")

# ---------------------------------------------------------------------------
# Helper: enrich LLM result with movie metadata from the CSV
# ---------------------------------------------------------------------------

def _enrich(result: dict) -> dict:
    row = TOP_MOVIES[TOP_MOVIES["tmdb_id"] == result["tmdb_id"]]
    if row.empty:
        return result
    r = row.iloc[0]

    def safe_int(val):
        try:
            v = int(val)
            return v if v > 0 else None
        except (ValueError, TypeError):
            return None

    result["title"] = str(r.get("title", ""))
    result["year"] = safe_int(r.get("year"))
    result["genres"] = str(r.get("genres", ""))
    result["director"] = str(r.get("director", ""))
    result["runtime"] = safe_int(r.get("runtime_min"))
    result["rating"] = str(r.get("us_rating", ""))
    result["vote_average"] = round(float(r.get("vote_average", 0)), 1)
    result["cast"] = str(r.get("top_cast", ""))[:120]
    result["tagline"] = str(r.get("tagline", ""))
    result["poster_url"] = str(r.get("poster_path", ""))
    result["tmdb_url"] = str(r.get("tmdb_url", ""))
    return result


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    preferences: str
    history: List[str] = []
    history_ids: List[int] = []


# ---------------------------------------------------------------------------
# HTML — single-file popcorn theme
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>🍿 PopcornPicks</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');

  :root {
    --bg:        #080810;
    --surface:   #10101f;
    --card:      #16162e;
    --border:    #27274a;
    --yellow:    #FFD700;
    --red:       #E63946;
    --cream:     #FFF8DC;
    --muted:     #7777aa;
    --radius:    14px;
    --shadow:    0 8px 40px rgba(0,0,0,0.6);
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--cream);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    background-image:
      radial-gradient(ellipse 60% 40% at 15% 60%, rgba(230,57,70,0.06) 0%, transparent 70%),
      radial-gradient(ellipse 60% 40% at 85% 40%, rgba(255,215,0,0.06) 0%, transparent 70%);
  }

  /* ── Header ── */
  header {
    text-align: center;
    padding: 3.5rem 1rem 2.5rem;
    border-bottom: 1px solid var(--border);
    position: relative;
    overflow: hidden;
  }

  .kernels {
    position: absolute;
    inset: 0;
    pointer-events: none;
    overflow: hidden;
  }
  .kernel {
    position: absolute;
    font-size: 1.5rem;
    opacity: 0.12;
    animation: float var(--d, 8s) var(--delay, 0s) ease-in-out infinite alternate;
  }
  @keyframes float {
    from { transform: translateY(0) rotate(0deg); }
    to   { transform: translateY(-30px) rotate(20deg); }
  }

  .logo-icon {
    font-size: 4rem;
    display: block;
    animation: pop 3s ease-in-out infinite;
    margin-bottom: 0.5rem;
  }
  @keyframes pop {
    0%,100% { transform: scale(1) rotate(-3deg); }
    50%      { transform: scale(1.12) rotate(3deg); }
  }

  .logo {
    font-family: 'Cinzel', serif;
    font-size: clamp(2rem, 6vw, 3.5rem);
    font-weight: 900;
    letter-spacing: 0.08em;
    background: linear-gradient(135deg, var(--yellow) 0%, #ff9f43 50%, var(--red) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .tagline {
    color: var(--muted);
    font-size: 0.78rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    margin-top: 0.6rem;
  }

  /* ── Layout ── */
  .container {
    max-width: 760px;
    margin: 0 auto;
    padding: 2rem 1.25rem 4rem;
  }

  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.5rem;
  }

  /* ── Form ── */
  .field { margin-bottom: 1.4rem; }

  label {
    display: block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.55rem;
  }

  textarea, input[type="text"] {
    width: 100%;
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--cream);
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    padding: 0.85rem 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
    resize: vertical;
  }
  textarea { height: 110px; }
  textarea:focus, input:focus {
    outline: none;
    border-color: var(--yellow);
    box-shadow: 0 0 0 3px rgba(255,215,0,0.12);
  }
  ::placeholder { color: var(--muted); opacity: 0.7; }

  .btn {
    width: 100%;
    padding: 1rem;
    background: linear-gradient(135deg, var(--yellow) 0%, #ff9f43 50%, var(--red) 100%);
    border: none;
    border-radius: 10px;
    color: #000;
    font-family: 'Cinzel', serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s;
    box-shadow: 0 4px 24px rgba(255,215,0,0.2);
  }
  .btn:hover:not(:disabled) {
    opacity: 0.92;
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(255,215,0,0.35);
  }
  .btn:active:not(:disabled) { transform: translateY(0); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }

  /* ── Loading ── */
  #loading {
    display: none;
    text-align: center;
    padding: 2.5rem;
  }
  #loading.show { display: block; }
  .spin {
    font-size: 3rem;
    display: inline-block;
    animation: spin 1.2s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-text {
    margin-top: 1rem;
    color: var(--muted);
    font-size: 0.85rem;
    letter-spacing: 0.12em;
  }

  /* ── Error ── */
  #error {
    display: none;
    background: rgba(230,57,70,0.1);
    border: 1px solid rgba(230,57,70,0.4);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: var(--red);
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
  }
  #error.show { display: block; }

  /* ── Result card ── */
  #result {
    display: none;
    animation: rise 0.45s cubic-bezier(.22,1,.36,1) both;
  }
  #result.show { display: block; }
  @keyframes rise {
    from { opacity: 0; transform: translateY(28px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .result-inner {
    display: flex;
    gap: 1.75rem;
    align-items: flex-start;
  }

  .poster-wrap {
    flex-shrink: 0;
    width: 140px;
  }
  .movie-poster {
    width: 140px;
    height: 210px;
    object-fit: cover;
    border-radius: 10px;
    box-shadow: 0 6px 28px rgba(255,215,0,0.22), 0 2px 8px rgba(0,0,0,0.5);
    display: block;
  }
  .poster-fallback {
    width: 140px;
    height: 210px;
    background: var(--border);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3.5rem;
  }

  .movie-info { flex: 1; min-width: 0; }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: linear-gradient(135deg, var(--yellow), var(--red));
    color: #000;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    margin-bottom: 0.85rem;
  }

  .movie-title {
    font-family: 'Cinzel', serif;
    font-size: clamp(1.1rem, 3vw, 1.5rem);
    font-weight: 700;
    color: var(--yellow);
    margin-bottom: 0.3rem;
    line-height: 1.25;
  }

  .movie-tagline {
    color: var(--muted);
    font-style: italic;
    font-size: 0.88rem;
    margin-bottom: 0.5rem;
  }

  .meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }
  .meta-chip {
    background: rgba(255,255,255,0.06);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.2rem 0.65rem;
    font-size: 0.74rem;
    color: var(--muted);
  }
  .meta-chip.rating {
    border-color: rgba(255,215,0,0.3);
    color: var(--yellow);
  }
  .meta-chip.stars {
    border-color: rgba(255,215,0,0.3);
    color: var(--yellow);
  }

  .description {
    font-size: 0.97rem;
    line-height: 1.72;
    color: var(--cream);
    border-left: 3px solid var(--red);
    padding-left: 1rem;
    margin-bottom: 1rem;
  }

  .tmdb-link {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    color: var(--muted);
    font-size: 0.78rem;
    text-decoration: none;
    transition: color 0.2s;
  }
  .tmdb-link:hover { color: var(--yellow); }

  /* ── Decorative divider ── */
  .section-divider {
    text-align: center;
    color: var(--border);
    font-size: 1.4rem;
    letter-spacing: 0.6rem;
    margin: 0.5rem 0 1.5rem;
    user-select: none;
  }

  /* ── Footer ── */
  footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.72rem;
    padding: 2rem;
    border-top: 1px solid var(--border);
    letter-spacing: 0.08em;
  }

  @media (max-width: 520px) {
    .result-inner { flex-direction: column; }
    .poster-wrap { width: 100%; }
    .movie-poster, .poster-fallback { width: 100%; height: 260px; }
    header { padding: 2rem 1rem 1.75rem; }
    .card { padding: 1.25rem; }
  }
</style>
</head>
<body>

<header>
  <div class="kernels" aria-hidden="true">
    <span class="kernel" style="left:5%;top:20%;--d:7s;--delay:0s">🌽</span>
    <span class="kernel" style="left:12%;top:60%;--d:9s;--delay:1s">🍿</span>
    <span class="kernel" style="left:80%;top:15%;--d:8s;--delay:2s">🌽</span>
    <span class="kernel" style="left:88%;top:65%;--d:6s;--delay:0.5s">🍿</span>
    <span class="kernel" style="left:50%;top:80%;--d:10s;--delay:1.5s">🌽</span>
    <span class="kernel" style="left:35%;top:10%;--d:11s;--delay:3s">🍿</span>
    <span class="kernel" style="left:65%;top:75%;--d:7.5s;--delay:2.5s">🌽</span>
  </div>
  <span class="logo-icon" aria-hidden="true">🍿</span>
  <h1 class="logo">PopcornPicks</h1>
  <p class="tagline">Your AI Movie Guru &nbsp;·&nbsp; Never Scroll Alone Again</p>
</header>

<div class="container">

  <div class="card">
    <div class="field">
      <label for="preferences">🎭 What's your vibe right now?</label>
      <textarea id="preferences"
        placeholder="Tell me how you're feeling, what you're in the mood for, a specific actor or director you love, where you want the story to be set — even just 'I'm heartbroken' works. I read between the lines."
      ></textarea>
    </div>
    <div class="field">
      <label for="history">👀 Already seen? (comma-separated)</label>
      <input type="text" id="history"
        placeholder="The Dark Knight, Inception, Interstellar…"/>
    </div>
    <button class="btn" id="submit-btn" onclick="getRecommendation()">
      🎬 &nbsp;Find My Movie
    </button>
  </div>

  <div id="loading">
    <div class="spin">🍿</div>
    <p class="loading-text">Consulting the popcorn oracle…</p>
  </div>

  <div id="error"></div>

  <div class="card" id="result">
    <div class="section-divider">· · ·</div>
    <div class="result-inner">
      <div class="poster-wrap" id="poster-wrap"></div>
      <div class="movie-info">
        <div class="badge">🍿 Your Pick</div>
        <div class="movie-title" id="movie-title"></div>
        <div class="movie-tagline" id="movie-tagline"></div>
        <div class="meta-row" id="meta-row"></div>
        <p class="description" id="description"></p>
        <a class="tmdb-link" id="tmdb-link" href="#" target="_blank" rel="noopener">
          ↗ View on TMDB
        </a>
      </div>
    </div>
  </div>

</div>

<footer>PopcornPicks &nbsp;·&nbsp; Powered by Gemma4 31B via Ollama &nbsp;·&nbsp; Built on Leapcell</footer>

<script>
  function chip(text, cls) {
    const s = document.createElement('span');
    s.className = 'meta-chip' + (cls ? ' ' + cls : '');
    s.textContent = text;
    return s;
  }

  async function getRecommendation() {
    const pref = document.getElementById('preferences').value.trim();
    if (!pref) { alert('Tell me what you\'re in the mood for! 🍿'); return; }

    const histRaw = document.getElementById('history').value.trim();
    const history = histRaw ? histRaw.split(',').map(s => s.trim()).filter(Boolean) : [];

    // UI state: loading
    document.getElementById('loading').classList.add('show');
    document.getElementById('result').classList.remove('show');
    document.getElementById('error').classList.remove('show');
    document.getElementById('submit-btn').disabled = true;

    try {
      const res = await fetch('/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preferences: pref, history, history_ids: [] })
      });
      if (!res.ok) throw new Error(await res.text());
      const d = await res.json();

      // Poster
      const pw = document.getElementById('poster-wrap');
      if (d.poster_url) {
        pw.innerHTML = '<img class="movie-poster" src="' + d.poster_url + '" alt="' + (d.title||'') + '" onerror="this.parentNode.innerHTML=\'<div class=poster-fallback>🎬</div>\'">';
      } else {
        pw.innerHTML = '<div class="poster-fallback">🎬</div>';
      }

      // Title & tagline
      const yr = d.year ? ' (' + d.year + ')' : '';
      document.getElementById('movie-title').textContent = (d.title || 'Unknown') + yr;
      document.getElementById('movie-tagline').textContent = d.tagline || '';

      // Meta chips
      const mr = document.getElementById('meta-row');
      mr.innerHTML = '';
      if (d.genres)        mr.appendChild(chip(d.genres));
      if (d.runtime)       mr.appendChild(chip(d.runtime + ' min'));
      if (d.director)      mr.appendChild(chip('Dir. ' + d.director));
      if (d.rating)        mr.appendChild(chip(d.rating, 'rating'));
      if (d.vote_average)  mr.appendChild(chip('★ ' + d.vote_average, 'stars'));

      // Description
      document.getElementById('description').textContent = d.description || '';

      // TMDB link
      const link = document.getElementById('tmdb-link');
      if (d.tmdb_url) { link.href = d.tmdb_url; link.style.display = 'inline-flex'; }
      else             { link.style.display = 'none'; }

      document.getElementById('result').classList.add('show');
      document.getElementById('result').scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
      console.error(err);
      const errEl = document.getElementById('error');
      errEl.textContent = 'Oops — the popcorn burnt! Try again? 🍿 (' + err.message + ')';
      errEl.classList.add('show');
    } finally {
      document.getElementById('loading').classList.remove('show');
      document.getElementById('submit-btn').disabled = false;
    }
  }

  // Ctrl+Enter submits
  document.getElementById('preferences').addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) getRecommendation();
  });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML


@app.post("/recommend")
async def recommend(body: RecommendRequest):
    history_ids = list(body.history_ids)

    # Auto-resolve history titles → tmdb_ids if caller didn't supply ids
    if body.history and not history_ids:
        for title in body.history:
            match = TOP_MOVIES[TOP_MOVIES["title"].str.lower() == title.lower()]
            if not match.empty:
                history_ids.append(int(match.iloc[0]["tmdb_id"]))

    result = get_recommendation(body.preferences, body.history, history_ids)
    return JSONResponse(_enrich(result))


@app.get("/health")
async def health():
    return {"status": "ok", "movies": len(TOP_MOVIES)}
