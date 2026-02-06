import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# ── Tarot API ───────────────────────────────────────────────────────────────
TAROT_API_URL = "https://tarotapi.dev/api/v1"

# Base URL for public-domain Rider-Waite card images (sacred-texts.com)
TAROT_IMAGE_BASE = "https://www.sacred-texts.com/tarot/pkt/img"

CONTEXTS = [
    "Career", "Love", "Health", "Spirituality",
    "Finances", "Family", "Creativity", "Personal Growth",
]


def draw_tarot_cards(n: int = 3) -> list[dict]:
    """Draw n random tarot cards from the Tarot API."""
    resp = requests.get(f"{TAROT_API_URL}/cards/random", params={"n": n})
    resp.raise_for_status()
    return resp.json()["cards"]

# ── Helper functions ────────────────────────────────────────────────────────

def build_prompt(cards: list[str], context: str) -> str:
    return f"""
You are a reflective assistant combining tarot symbolism and music psychology.
You do not predict the future. You provide grounded, introspective interpretations.

Tarot cards drawn:
{', '.join(cards)}

Context:
{context}

Tasks:
1. Interpret the tarot reading
2. Identify emotional themes
3. Recommend 3 music genres or styles
4. Explain why each recommendation fits

Respond clearly and thoughtfully.
"""


def get_tarot_reading(cards: list[str], context: str) -> str:
    """Send the tarot prompt to Groq and return the LLM response text."""
    prompt = build_prompt(cards, context)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7,
    }
    resp = requests.post(GROQ_API_URL, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_genres(reading_text: str) -> list[str]:
    """Ask the LLM to extract genre names from the reading as a JSON list."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "user",
                "content": (
                    "From the following text, extract ONLY the music genre/style names "
                    "as a JSON list of strings. Return ONLY the JSON array, nothing else.\n\n"
                    + reading_text
                ),
            }
        ],
        "max_tokens": 100,
        "temperature": 0.0,
    }
    resp = requests.post(GROQ_API_URL, headers=headers, json=payload)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    return json.loads(raw)


def get_spotify_token() -> str:
    """Obtain a Spotify access token via Client Credentials flow."""
    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "client_credentials",
            "client_id": SPOTIFY_CLIENT_ID,
            "client_secret": SPOTIFY_CLIENT_SECRET,
        },
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def search_tracks(genres: list[str], token: str, limit: int = 3) -> list[dict]:
    """Search Spotify for tracks matching each genre."""
    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    all_tracks = []

    for genre in genres:
        params = {"q": f"genre:{genre}", "type": "track", "limit": limit}
        resp = requests.get(url, headers=headers, params=params)
        tracks = resp.json().get("tracks", {}).get("items", [])

        if not tracks:
            params["q"] = genre
            resp = requests.get(url, headers=headers, params=params)
            tracks = resp.json().get("tracks", {}).get("items", [])

        for t in tracks:
            all_tracks.append(
                {
                    "id": t["id"],
                    "name": t["name"],
                    "artist": t["artists"][0]["name"],
                    "genre": genre,
                    "url": t["external_urls"]["spotify"],
                }
            )
    return all_tracks


# ── Streamlit UI ────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Tarot Music", page_icon="🔮", layout="centered")

# Center images within columns
st.markdown(
    """
    <style>
    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔮 AI Tarot Music")
st.markdown("*Draw tarot cards, receive a reflective reading, and listen to music that matches your energy.*")

# ── Sidebar – Configuration ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    num_cards = st.slider("Number of cards to draw", 1, 5, 2)
    context = st.selectbox("Life area / context", CONTEXTS)
    st.markdown("---")
    st.caption("Powered by Groq (Llama 3), Tarot API & Spotify")

# ── Main area ────────────────────────────────────────────────────────────────

if st.button("🃏 Draw Cards & Get Reading", type="primary", use_container_width=True):

    # Validate API keys
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not SPOTIFY_CLIENT_ID:
        missing.append("SPOTIFY_CLIENT_ID")
    if not SPOTIFY_CLIENT_SECRET:
        missing.append("SPOTIFY_CLIENT_SECRET")
    if missing:
        st.error(f"Missing environment variables: {', '.join(missing)}. Add them to your `.env` file.")
        st.stop()

    # Draw random cards from the Tarot API
    with st.spinner("Drawing cards..."):
        try:
            drawn_cards = draw_tarot_cards(num_cards)
        except Exception as e:
            st.error(f"Error fetching tarot cards: {e}")
            st.stop()

    card_names = [c["name"] for c in drawn_cards]

    # Display cards with images, names, and meanings
    st.subheader("Your Cards")
    cols = st.columns(num_cards)
    for i, card in enumerate(drawn_cards):
        with cols[i]:
            img_url = f"{TAROT_IMAGE_BASE}/{card['name_short']}.jpg"
            st.image(img_url, caption=card["name"], width=150)
            st.markdown(f"**Upright:** {card['meaning_up']}")
            st.markdown(f"**Reversed:** {card['meaning_rev']}")

    # Get tarot reading from Groq
    with st.spinner("Consulting the cards..."):
        try:
            reading = get_tarot_reading(card_names, context)
        except Exception as e:
            st.error(f"Error getting tarot reading: {e}")
            st.stop()

    st.subheader("📖 Your Reading")
    st.markdown(reading)

    # Extract genres
    with st.spinner("Identifying music genres..."):
        try:
            genres = extract_genres(reading)
        except Exception as e:
            st.error(f"Error extracting genres: {e}")
            st.stop()

    st.subheader("🎵 Recommended Genres")
    st.write(", ".join(genres))

    # Search Spotify and display embedded players
    with st.spinner("Finding tracks on Spotify..."):
        try:
            token = get_spotify_token()
            tracks = search_tracks(genres, token)
        except Exception as e:
            st.error(f"Error searching Spotify: {e}")
            st.stop()

    if tracks:
        st.subheader("🎧 Listen Now")
        for track in tracks:
            embed_url = f"https://open.spotify.com/embed/track/{track['id']}?utm_source=generator&theme=0"
            st.markdown(f"**{track['name']}** — {track['artist']}  `[{track['genre']}]`")
            st.components.v1.html(
                f'<iframe src="{embed_url}" width="100%" height="80" '
                f'frameBorder="0" allow="autoplay; clipboard-write; encrypted-media; '
                f'fullscreen; picture-in-picture" loading="lazy"></iframe>',
                height=90,
            )
    else:
        st.warning("No tracks found on Spotify for the recommended genres.")
