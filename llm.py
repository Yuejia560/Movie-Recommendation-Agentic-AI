"""
Agentic Movie Recommender - llm.py

get_recommendation() is called once per request with the user's input.
Returns a dict with keys "tmdb_id" and "description".

IMPORTANT: Do NOT hard-code your API key. The grader supplies OLLAMA_API_KEY
at runtime via the environment.

Additional env vars used (all optional):
  OLLAMA_API_KEY  - required; injected by grader
"""

import json
import os
import re
import time
import argparse

import ollama
import pandas as pd

# ---------------------------------------------------------------------------
# Model & data
# ---------------------------------------------------------------------------

MODEL = "gemma4:31b-cloud"
DATA_PATH = os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv")

# All 1,000 movies are valid candidates — test.py derives VALID_IDS from this.
TOP_MOVIES = pd.read_csv(DATA_PATH).fillna("")

# ---------------------------------------------------------------------------
# Sentiment → genre preference map
# ---------------------------------------------------------------------------

_SENTIMENT_GENRES: dict[str, list[str]] = {
    "heartbroken":   ["Romance", "Drama"],
    "heart broken":  ["Romance", "Drama"],
    "heartbreak":    ["Romance", "Drama"],
    "broken heart":  ["Romance", "Drama"],
    "sad":           ["Comedy", "Animation", "Family"],
    "depressed":     ["Comedy", "Family", "Adventure"],
    "lonely":        ["Romance", "Drama", "Comedy"],
    "anxious":       ["Comedy", "Animation", "Family"],
    "stressed":      ["Comedy", "Adventure", "Animation"],
    "nervous":       ["Comedy", "Animation"],
    "angry":         ["Comedy", "Action", "Animation"],
    "bored":         ["Thriller", "Mystery", "Crime", "Action"],
    "excited":       ["Action", "Adventure", "Science Fiction"],
    "thrilled":      ["Action", "Adventure", "Thriller"],
    "happy":         ["Comedy", "Adventure", "Animation"],
    "joyful":        ["Comedy", "Family", "Animation"],
    "scared":        ["Horror", "Thriller"],
    "love":          ["Romance", "Drama"],
    "romantic":      ["Romance", "Drama"],
    "hate":          ["Action", "Thriller", "Comedy"],
    "nostalgic":     ["Animation", "Family", "Drama"],
    "tired":         ["Comedy", "Animation", "Family"],
    "confused":      ["Mystery", "Thriller", "Drama"],
    "inspired":      ["Drama", "Biography", "Adventure"],
    "motivated":     ["Drama", "Sport", "Adventure"],
    "cozy":          ["Comedy", "Family", "Romance"],
    "adventurous":   ["Adventure", "Action", "Science Fiction"],
}

_STOP_WORDS = {
    "i", "me", "my", "we", "us", "our", "you", "your",
    "the", "a", "an", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "is", "am", "are",
    "was", "were", "want", "like", "watch", "something", "movie",
    "film", "good", "great", "really", "very", "just", "some",
    "can", "would", "could", "should", "please", "thank", "need",
    "looking", "feeling", "feel", "been", "have", "has", "that",
    "this", "it", "so", "do", "did", "not", "no", "yes", "get",
    "see", "make", "know", "think", "go", "one", "more", "up",
    "about", "into", "out", "also", "tell", "give", "show",
}


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def _detect_sentiment_genres(preferences: str) -> list[str]:
    prefs_lower = preferences.lower()
    matched: list[str] = []
    for sentiment, genres in _SENTIMENT_GENRES.items():
        if sentiment in prefs_lower:
            matched.extend(genres)
    return list(set(matched))


def _score_movie(row: dict, prefs_lower: str, pref_words: set, sentiment_genres: list[str]) -> float:
    """Score a movie's relevance to the user's query. Higher = better match."""
    searchable = " ".join([
        str(row.get("title", "")),
        str(row.get("overview", "")),
        str(row.get("genres", "")),
        str(row.get("keywords", "")),
        str(row.get("director", "")),
        str(row.get("top_cast", "")),
        str(row.get("tagline", "")),
        str(row.get("production_countries", "")),
        str(row.get("us_rating", "")),
        str(row.get("original_title", "")),
    ]).lower()

    score = 0.0

    # Exact phrase match in any field (very strong signal)
    if len(prefs_lower) > 4 and prefs_lower in searchable:
        score += 15.0

    # Individual word matches
    for word in pref_words:
        if len(word) > 2 and word in searchable:
            score += 2.5

    # Bigram matches (e.g., "tom hanks", "new york")
    words_seq = [w for w in re.findall(r'\b\w+\b', prefs_lower) if w not in _STOP_WORDS and len(w) > 2]
    for i in range(len(words_seq) - 1):
        bigram = words_seq[i] + " " + words_seq[i + 1]
        if bigram in searchable:
            score += 5.0

    # Sentiment genre bonus
    genres_lower = str(row.get("genres", "")).lower()
    for genre in sentiment_genres:
        if genre.lower() in genres_lower:
            score += 4.0

    # Quality signal
    try:
        score += float(row.get("vote_average", 0)) / 10.0 * 2.0
    except (ValueError, TypeError):
        pass
    try:
        score += min(float(row.get("vote_count", 0)) / 5000.0, 1.5)
    except (ValueError, TypeError):
        pass

    return score


def _select_candidates(preferences: str, history_ids: list[int], n: int = 20) -> pd.DataFrame:
    """Return the top-N candidate movies relevant to preferences, excluding watched."""
    candidates = TOP_MOVIES[~TOP_MOVIES["tmdb_id"].isin(history_ids)].copy()

    prefs_lower = preferences.lower()
    pref_words = set(re.findall(r'\b\w+\b', prefs_lower)) - _STOP_WORDS
    sentiment_genres = _detect_sentiment_genres(preferences)

    records = candidates.to_dict("records")
    scores = [_score_movie(r, prefs_lower, pref_words, sentiment_genres) for r in records]
    candidates = candidates.copy()
    candidates["_score"] = scores

    return candidates.nlargest(n, "_score")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_movie_entry(row: dict) -> str:
    """Format all useful movie fields for the LLM prompt."""
    parts = [
        f'tmdb_id={row.get("tmdb_id")}',
        f'"{row.get("title")}" ({row.get("year")})',
        f'genres: {row.get("genres")}',
    ]
    director = str(row.get("director", ""))
    if director:
        parts.append(f'director: {director}')

    cast = str(row.get("top_cast", ""))[:150]
    if cast:
        parts.append(f'cast: {cast}')

    countries = str(row.get("production_countries", ""))
    if countries:
        parts.append(f'filmed in: {countries}')

    tagline = str(row.get("tagline", ""))
    if tagline:
        parts.append(f'tagline: "{tagline}"')

    overview = str(row.get("overview", ""))[:280]
    if overview:
        parts.append(f'overview: {overview}')

    keywords = str(row.get("keywords", ""))[:120]
    if keywords:
        parts.append(f'keywords: {keywords}')

    rating = str(row.get("us_rating", ""))
    if rating:
        parts.append(f'rated: {rating}')

    return "- " + " | ".join(parts)


def build_prompt(preferences: str, history: list[str], history_ids: list[int]) -> tuple[str, pd.DataFrame]:
    """Build the LLM prompt; return it alongside the candidate DataFrame."""
    candidates = _select_candidates(preferences, history_ids)

    movie_list = "\n".join(
        _format_movie_entry(row) for row in candidates.to_dict("records")
    )

    history_text = (
        ", ".join(f'"{name}" (id={tid})' for name, tid in zip(history, history_ids))
        if history else "none"
    )

    prompt = f"""You are an empathetic, witty movie recommendation guru who reads between the lines.

USER'S EXACT MESSAGE: "{preferences}"

Already watched (NEVER recommend these): {history_text}

CANDIDATE MOVIES — you MUST pick exactly one tmdb_id from this list:
{movie_list}

INSTRUCTIONS:
1. Analyse the user's emotional state, specific desires, and what they're going through RIGHT NOW.
   Consider: mood, sentiment (heartbroken, bored, excited, etc.), any specific actors/directors/
   locations/themes they mention.
2. Pick the single best movie for this exact moment in their life.
3. Write a description that is EITHER:
   - Emotionally resonant: make them feel SEEN, reference their exact situation or mood, explain
     why this movie is what they need right now (comfort, catharsis, distraction, inspiration…)
   - Delightfully funny/cheeky: if humor suits the context, be playful and witty about why this
     is their movie — lean into the irony or the perfect absurdity of the match.
4. Be SPECIFIC: mention actual plot details, actors, or themes that connect to the user's request.
5. Keep description ≤ 500 characters. Every word must earn its place.

Respond with ONLY valid JSON, no markdown, no extra keys:
{{
  "tmdb_id": <integer from the candidate list>,
  "description": "<your personalized blurb, ≤500 chars>"
}}"""

    return prompt, candidates


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> dict:
    client = ollama.Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )
    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    return json.loads(response.message.content)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_recommendation(preferences: str, history: list[str], history_ids: list[int] = []) -> dict:
    """Return a dict with keys 'tmdb_id' (int) and 'description' (str ≤500 chars)."""
    prompt, candidates = build_prompt(preferences, history, history_ids)
    result = call_llm(prompt)

    # Guard: tmdb_id must be in our candidate set
    valid_ids = {int(x) for x in candidates["tmdb_id"].tolist()}
    if int(result.get("tmdb_id", -1)) not in valid_ids:
        result["tmdb_id"] = int(candidates.iloc[0]["tmdb_id"])

    result["tmdb_id"] = int(result["tmdb_id"])

    # Enforce 500-char description cap
    desc = str(result.get("description", ""))
    if len(desc) > 500:
        result["description"] = desc[:497] + "..."

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local movie recommendation test.")
    parser.add_argument("--preferences", type=str, help="User preferences text.")
    parser.add_argument("--history", type=str, help="Comma-separated watch history titles.")
    args = parser.parse_args()

    print("🍿 PopcornPicks — AI Movie Recommender")
    print("Enter your mood/preferences and we'll find the perfect movie.\n")

    preferences = (
        args.preferences.strip()
        if args.preferences and args.preferences.strip()
        else input("What are you in the mood for? ").strip()
    )
    history_raw = (
        args.history.strip()
        if args.history and args.history.strip()
        else input("Movies you've already seen (optional, comma-separated): ").strip()
    )
    history = [t.strip() for t in history_raw.split(",") if t.strip()] if history_raw else []

    print("\n🎬 Thinking...\n")
    start = time.perf_counter()
    result = get_recommendation(preferences, history)
    elapsed = time.perf_counter() - start

    # Look up movie title for friendly output
    movie_row = TOP_MOVIES[TOP_MOVIES["tmdb_id"] == result["tmdb_id"]]
    if not movie_row.empty:
        r = movie_row.iloc[0]
        print(f"Recommended: {r['title']} ({r['year']})")
    print(json.dumps(result, indent=2))
    print(f"\n⏱  Served in {elapsed:.2f}s")
