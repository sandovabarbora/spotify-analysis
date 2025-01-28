import os
import requests
import csv
import webbrowser
from urllib.parse import urlencode, urlparse, parse_qs
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8080/callback"
AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE_URL = "https://api.spotify.com/v1"
SCOPES = ["user-top-read", "user-read-recently-played", "user-library-read"]


def create_auth_url():
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES)
    }
    return f"{AUTH_URL}?{urlencode(params)}"


def get_access_token(code):
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    response = requests.post(TOKEN_URL, data=data)
    response.raise_for_status()
    return response.json()


def spotify_get(endpoint, access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(f"{API_BASE_URL}{endpoint}", headers=headers)
    response.raise_for_status()
    return response.json()


def get_top_tracks(access_token, time_range):
    return spotify_get(f"/me/top/tracks?time_range={time_range}&limit=50", access_token)


def get_recently_played(access_token):
    return spotify_get("/me/player/recently-played?limit=50", access_token)


def extract_track_info(track):
    return {
        "name": track["name"],
        "artist": track["artists"][0]["name"],
        "popularity": track["popularity"],
        "duration_ms": track["duration_ms"],
    }


def analyze_top_tracks(tracks):
    return [
        {**extract_track_info(track), "rank": idx + 1}
        for idx, track in enumerate(tracks.get("items", []))
    ]


def analyze_recent_tracks(tracks):
    return [
        {
            **extract_track_info(item["track"]),
            "played_at": item["played_at"],
        }
        for item in tracks.get("items", [])
    ]


def save_to_csv(data, filename):
    if not data:
        print(f"No data to save for {filename}")
        return

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


def authenticate():
    print("Opening browser for authentication...")
    webbrowser.open(create_auth_url())
    code = input("Please enter the code from the redirect URL: ")
    return code


def analyze_listening_habits():
    code = authenticate()
    token_response = get_access_token(code)
    access_token = token_response["access_token"]

    # Fetch and analyze data
    top_tracks_short = analyze_top_tracks(get_top_tracks(access_token, "short_term"))
    top_tracks_medium = analyze_top_tracks(get_top_tracks(access_token, "medium_term"))
    top_tracks_long = analyze_top_tracks(get_top_tracks(access_token, "long_term"))
    recent_tracks = analyze_recent_tracks(get_recently_played(access_token))

    # Save to CSV
    save_to_csv(top_tracks_short, "spotify_top_tracks_short.csv")
    save_to_csv(top_tracks_medium, "spotify_top_tracks_medium.csv")
    save_to_csv(top_tracks_long, "spotify_top_tracks_long.csv")
    save_to_csv(recent_tracks, "spotify_recent_tracks.csv")

    # Basic analysis
    print("\nTop Artists (Last 4 Weeks):")
    artist_counts = {}
    for track in top_tracks_short:
        artist = track["artist"]
        artist_counts[artist] = artist_counts.get(artist, 0) + 1

    sorted_artists = sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)
    for artist, count in sorted_artists[:5]:
        print(f"{artist}: {count}")

    return {
        "top_tracks": {
            "short_term": top_tracks_short,
            "medium_term": top_tracks_medium,
            "long_term": top_tracks_long,
        },
        "recent_tracks": recent_tracks,
    }


if __name__ == "__main__":
    analyze_listening_habits()
