import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
import requests
from typing import Dict, List, Set, Tuple, Optional
import time
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import math
import traceback

class SpotifyDataEnricher:
    def __init__(self, client_id: str, client_secret: str, max_workers: int = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.logger = self._setup_logging()
        self.max_workers = max_workers or max(1, cpu_count() - 1)
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('SpotifyEnricher')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_access_token(self) -> str:
        """Get Spotify API access token."""
        self.logger.info("Requesting new access token")
        auth_url = 'https://accounts.spotify.com/api/token'
        auth_response = requests.post(auth_url, {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
        })
        auth_response.raise_for_status()
        self.logger.info("Successfully obtained new access token")
        return auth_response.json()['access_token']

    @staticmethod
    def _make_request(url: str, headers: Dict, max_retries: int = 3) -> Optional[Dict]:
        """Make a request to Spotify API with retry logic."""
        for attempt in range(max_retries):
            try:
                logging.getLogger('SpotifyEnricher').debug(f"Making request to {url} (attempt {attempt + 1}/{max_retries})")
                response = requests.get(url, headers=headers)
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 1))
                    logging.getLogger('SpotifyEnricher').warning(f"Rate limited, waiting {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                
                if response.status_code == 403:
                    logging.getLogger('SpotifyEnricher').warning("Token expired, need refresh")
                    return {'error': 'token_expired'}
                    
                if response.status_code == 404:
                    logging.getLogger('SpotifyEnricher').warning(f"Resource not found: {url}")
                    return None
                    
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logging.getLogger('SpotifyEnricher').error(f"Request failed: {str(e)}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(1)
        
        return None

    def _process_track_batch(self, batch_data: Tuple[List[str], str, Manager]) -> List[Dict]:
        """Process a batch of tracks with token refresh capability."""
        track_uris, initial_token, token_manager = batch_data
        logger = logging.getLogger('SpotifyEnricher')
        batch_id = track_uris[0] if track_uris else 'empty-batch'  # For logging
        logger.info(f"Starting batch processing for batch {batch_id}")
        
        results = []
        current_token = initial_token
        headers = {'Authorization': f'Bearer {current_token}'}
        
        for idx, track_uri in enumerate(track_uris, 1):
            try:
                if pd.isna(track_uri):
                    logger.debug(f"Skipping NaN track URI")
                    continue
                
                logger.debug(f"Processing track {idx}/{len(track_uris)}: {track_uri}")
                track_id = track_uri.split(':')[-1]
                
                # Get track info
                track_url = f'https://api.spotify.com/v1/tracks/{track_id}'
                logger.debug(f"Fetching track data for {track_id}")
                track_data = self._make_request(track_url, headers)
                
                # Handle token expiration
                if track_data and track_data.get('error') == 'token_expired':
                    logger.info("Token expired during track fetch, refreshing...")
                    if token_manager.token.value:
                        current_token = token_manager.token.value
                        logger.debug("Using token from manager")
                    else:
                        current_token = self.get_access_token()
                        token_manager.token.value = current_token
                        logger.debug("Generated new token")
                    headers = {'Authorization': f'Bearer {current_token}'}
                    track_data = self._make_request(track_url, headers)
                
                if not track_data or isinstance(track_data.get('error'), dict):
                    logger.warning(f"Failed to fetch track data for {track_uri}")
                    continue
                
                # Validate track data
                if not all(key in track_data for key in ['artists', 'album']):
                    logger.warning(f"Missing required fields in track data for {track_uri}")
                    continue
                
                if not track_data['artists']:
                    logger.warning(f"No artists found for track {track_uri}")
                    continue
                
                # Get audio features with detailed logging
                features_url = f'https://api.spotify.com/v1/audio-features/{track_id}'
                logger.debug(f"Fetching audio features for {track_id}")
                features_data = self._make_request(features_url, headers) or {}
                
                if features_data.get('error') == 'token_expired':
                    logger.info("Token expired during features fetch, refreshing...")
                    if token_manager.token.value:
                        current_token = token_manager.token.value
                    else:
                        current_token = self.get_access_token()
                        token_manager.token.value = current_token
                    headers = {'Authorization': f'Bearer {current_token}'}
                    features_data = self._make_request(features_url, headers) or {}
                
                # Get artist data with detailed logging
                artist_id = track_data['artists'][0].get('id')
                artist_data = {}
                if artist_id:
                    artist_url = f'https://api.spotify.com/v1/artists/{artist_id}'
                    logger.debug(f"Fetching artist data for {artist_id}")
                    artist_data = self._make_request(artist_url, headers) or {}
                    
                    if artist_data.get('error') == 'token_expired':
                        logger.info("Token expired during artist fetch, refreshing...")
                        if token_manager.token.value:
                            current_token = token_manager.token.value
                        else:
                            current_token = self.get_access_token()
                            token_manager.token.value = current_token
                        headers = {'Authorization': f'Bearer {current_token}'}
                        artist_data = self._make_request(artist_url, headers) or {}
                
                # Compile track info
                track_info = {
                    'track_id': track_id,
                    'track_uri': track_uri,
                    'name': track_data.get('name'),
                    'artist_id': artist_id,
                    'artist_name': track_data['artists'][0].get('name'),
                    'artist_genres': artist_data.get('genres', []),
                    'album_id': track_data['album'].get('id'),
                    'album_name': track_data['album'].get('name'),
                    'popularity': track_data.get('popularity'),
                    'duration_ms': track_data.get('duration_ms'),
                    'explicit': track_data.get('explicit'),
                    **{k: v for k, v in features_data.items() if not isinstance(v, dict)}
                }
                results.append(track_info)
                logger.debug(f"Successfully processed track {track_uri}")
                
                # Rate limiting
                time.sleep(0.2)  # Increased sleep time for safety
                
            except Exception as e:
                logger.error(f"Error processing track {track_uri}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        logger.info(f"Completed batch processing for batch {batch_id}, processed {len(results)} tracks")
        return results

    def enrich_data(self, data_path: str, output_path: str = None, batch_size: int = 20):  # Reduced batch size
        """Enrich existing listening history with additional track information using multiprocessing."""
        self.logger.info(f"Starting data enrichment process for {data_path}")
        
        try:
            df = pd.read_parquet(data_path)
            self.logger.info(f"Successfully loaded data file with {len(df)} rows")
            
            # Get unique track URIs
            unique_tracks = df['spotify_track_uri'].dropna().unique()
            total_tracks = len(unique_tracks)
            self.logger.info(f"Found {total_tracks} unique tracks to process")
            
            # Get initial access token
            initial_token = self.get_access_token()
            
            # Create a manager for sharing the token across processes
            with Manager() as manager:
                self.logger.info("Initializing multiprocessing manager")
                token_manager = manager.Namespace()
                token_manager.token = manager.Value('s', initial_token)
                
                # Split tracks into batches
                num_batches = math.ceil(total_tracks / batch_size)
                self.logger.info(f"Splitting {total_tracks} tracks into {num_batches} batches")
                track_batches = np.array_split(unique_tracks, num_batches)
                
                # Prepare batch data
                batch_data = [(batch.tolist(), initial_token, token_manager) for batch in track_batches]
                
                # Process batches in parallel
                self.logger.info(f"Starting parallel processing with {self.max_workers} workers")
                process_batch = partial(self._process_track_batch)
                with Pool(self.max_workers) as pool:
                    all_results = []
                    for i, batch_results in enumerate(pool.imap(process_batch, batch_data)):
                        self.logger.info(f"Completed batch {i + 1}/{num_batches}")
                        all_results.extend(batch_results)
                        
            # Process results
            self.logger.info(f"Processing completed. Converting {len(all_results)} results to DataFrame")
            track_info_df = pd.DataFrame(all_results)
            
            if track_info_df.empty:
                self.logger.warning("No results were collected!")
                return None, None
                
            track_info_df.set_index('track_uri', inplace=True)
            
            # Save results
            if output_path:
                self.logger.info("Saving results to files")
                output_dir = Path(output_path).parent
                track_info_df.to_parquet(output_dir / 'track_features.parquet')
                
                enriched_df = df.merge(
                    track_info_df,
                    left_on='spotify_track_uri',
                    right_index=True,
                    how='left'
                )
                enriched_df.to_parquet(output_path)
                
                self.logger.info(f"Successfully saved enriched data to {output_path}")
            
            return enriched_df, track_info_df
            
        except Exception as e:
            self.logger.error(f"Critical error in enrich_data: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

if __name__ == "__main__":
    import os
    
    # Get credentials from environment variables
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not (client_id and client_secret):
        raise ValueError("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables")
    
    enricher = SpotifyDataEnricher(client_id, client_secret)
    
    # Paths
    input_path = "data/processed/merged.parquet"
    output_path = "data/processed/enriched_history.parquet"
    
    # Enrich data
    enriched_df, track_features = enricher.enrich_data(input_path, output_path)