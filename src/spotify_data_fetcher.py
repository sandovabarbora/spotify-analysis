import os
import requests
import json
from datetime import datetime, timezone, timedelta
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode

class SpotifyDataFetcher:
    def __init__(self):
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.redirect_uri = "http://localhost:8080/callback"
        self.scopes = "user-top-read user-read-recently-played user-library-read playlist-read-private"
        self.data_dir = Path('data')
        self.logger = self._setup_logging()
        self.ensure_directories()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('SpotifyDataFetcher')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('spotify_fetcher.log')
        
        # Create formatters and add it to handlers
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        c_format = logging.Formatter(format_str)
        f_format = logging.Formatter(format_str)
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger

    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = ['raw', 'processed', 'temp']
        for dir_name in directories:
            dir_path = self.data_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_last_update_time(self) -> Optional[datetime]:
        """Get the timestamp of the last data update."""
        timestamp_file = self.data_dir / 'temp' / 'last_update.txt'
        if timestamp_file.exists():
            with open(timestamp_file, 'r') as f:
                timestamp_str = f.read().strip()
                return datetime.fromisoformat(timestamp_str)
        return None

    def save_last_update_time(self):
        """Save the current timestamp as the last update time."""
        timestamp_file = self.data_dir / 'temp' / 'last_update.txt'
        with open(timestamp_file, 'w') as f:
            f.write(datetime.now(timezone.utc).isoformat())

    def get_auth_code(self):
        """Start a local server to capture the Spotify authorization code."""
        class SpotifyAuthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if "/callback" in self.path:
                    query = self.path.split("?")[1]
                    code = dict(pair.split("=") for pair in query.split("&"))["code"]
                    self.server.code = code
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"Authorization successful! You can close this window.")

        auth_url = f'''https://accounts.spotify.com/authorize?{urlencode({
            'client_id': self.client_id,
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'scope': self.scopes
        })}'''
        
        print(f"Go to the following URL to authorize:\n{auth_url}")
        server = HTTPServer(("localhost", 8080), SpotifyAuthHandler)
        server.handle_request()
        return server.code

    def get_access_token(self, code: str) -> Dict:
        """Exchange the authorization code for an access token."""
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }
        )
        response.raise_for_status()
        return response.json()

    def spotify_get(self, endpoint: str, access_token: str) -> Dict:
        """Make a GET request to the Spotify API."""
        response = requests.get(
            f"https://api.spotify.com/v1{endpoint}",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        response.raise_for_status()
        return response.json()

    def fetch_recent_data(self, access_token: str) -> Dict:
        """
        Fetch only recent listening data since the last update.
        Returns both the raw API response and processed data.
        """
        last_update = self.get_last_update_time()
        
        try:
            # Fetch recent tracks (Spotify API limits to last 50 tracks)
            recent_data = self.spotify_get(
                "/me/player/recently-played?limit=50", 
                access_token
            )
            
            # Fetch current top tracks
            top_tracks_short = self.spotify_get(
                "/me/top/tracks?time_range=short_term&limit=50",
                access_token
            )
            
            # Process the data
            processed_data = {
                'recent_tracks': [],
                'top_tracks': []
            }
            
            # Process recent tracks
            for item in recent_data.get('items', []):
                played_at = datetime.fromisoformat(item['played_at'].replace('Z', '+00:00'))
                if not last_update or played_at > last_update:
                    track_data = {
                        'ts': played_at.isoformat(),
                        'ms_played': item['track']['duration_ms'],  # Actual played duration not available
                        'master_metadata_track_name': item['track']['name'],
                        'master_metadata_album_artist_name': item['track']['artists'][0]['name'],
                        'master_metadata_album_album_name': item['track']['album']['name'],
                        'spotify_track_uri': item['track']['uri'],
                        'platform': 'API Fetch',
                        'reason_start': 'trackdone',
                        'reason_end': 'trackdone'
                    }
                    processed_data['recent_tracks'].append(track_data)
            
            # Save raw API response
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            raw_data_file = self.data_dir / 'raw' / f'spotify_api_data_{timestamp}.json'
            with open(raw_data_file, 'w') as f:
                json.dump({
                    'recent_tracks': recent_data,
                    'top_tracks_short': top_tracks_short
                }, f, indent=2)
            
            # Save processed data
            if processed_data['recent_tracks']:
                processed_file = self.data_dir / 'processed' / f'recent_tracks_{timestamp}.json'
                with open(processed_file, 'w') as f:
                    json.dump(processed_data['recent_tracks'], f, indent=2)
                
                self.logger.info(f"Saved {len(processed_data['recent_tracks'])} new tracks")
            else:
                self.logger.info("No new tracks to save")
            
            # Update the last update timestamp
            self.save_last_update_time()
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error fetching recent data: {str(e)}")
            raise

def run_daily_update():
    """Function to run the daily update process."""
    fetcher = SpotifyDataFetcher()
    
    try:
        # Get new authentication token
        auth_code = fetcher.get_auth_code()
        token_data = fetcher.get_access_token(auth_code)
        access_token = token_data["access_token"]
        
        # Fetch and save new data
        new_data = fetcher.fetch_recent_data(access_token)
        
        return new_data
    except Exception as e:
        fetcher.logger.error(f"Error in daily update: {str(e)}")
        raise

if __name__ == "__main__":
    # This can be scheduled to run daily
    run_daily_update()