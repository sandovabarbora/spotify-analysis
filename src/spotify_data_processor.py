import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone, timedelta
import os
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import glob

class SpotifyHistoricalDataProcessor:
    def __init__(self, data_dir: str = 'data/'):
        """
        Initialize the Spotify data processor.
        
        Args:
            data_dir (str): Base directory for all data files
        """
        self.data_dir = Path(data_dir)
        self.logger = self._setup_logging()
        self.ensure_directories()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('SpotifyDataProcessor')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('spotify_processor.log')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.INFO)
        
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
        directories = ['raw', 'processed', 'enriched', 'logs']
        for dir_name in directories:
            dir_path = self.data_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Ensured directory exists: {dir_path}")

    def process_multiple_history_files(self, file_pattern: str) -> pd.DataFrame:
        """
        Process multiple streaming history JSON files and combine them.
        
        Args:
            file_pattern (str): Pattern to match history files (e.g., "Streaming_History_Audio_*.json")
            
        Returns:
            pd.DataFrame: Combined and processed historical data
        """
        self.logger.info(f"Processing multiple history files matching pattern: {file_pattern}")
        
        all_data = []
        
        # Get all matching files
        files = glob.glob(file_pattern)
        self.logger.info(f"Found {len(files)} files to process")
        
        for file in sorted(files):
            try:
                self.logger.info(f"Processing file: {file}")
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                all_data.append(df)
                
            except Exception as e:
                self.logger.error(f"Error processing file {file}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No data was successfully processed from any file")
        
        # Combine all DataFrames
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Process the combined DataFrame
        processed_df = self._process_streaming_data(combined_df)
        
        return processed_df

    def _process_streaming_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process streaming data with feature engineering.
        
        Args:
            df (pd.DataFrame): Raw streaming data
            
        Returns:
            pd.DataFrame: Processed data with additional features
        """
        # Convert timestamp and create time-based features
        df['ts'] = pd.to_datetime(df['ts'])
        df['date'] = df['ts'].dt.date
        df['hour'] = df['ts'].dt.hour
        df['day_of_week'] = df['ts'].dt.dayofweek
        df['month'] = df['ts'].dt.month
        df['year'] = df['ts'].dt.year
        df['minutes_played'] = df['ms_played'] / 60000
        
        # Create unique track identifier
        df['track_id'] = df['spotify_track_uri'].fillna(
            df['master_metadata_track_name'] + ' - ' + df['master_metadata_album_artist_name']
        )
        
        # Create listening type categories
        df['listen_type'] = pd.cut(
            df['minutes_played'],
            bins=[-np.inf, 0.5, 1, 2, np.inf],
            labels=['preview', 'partial', 'full', 'extended']
        )
        
        # Extract platform information
        df['device_type'] = df['platform'].str.extract(r'\((.*?)\)').iloc[:, 0]
        df['platform_name'] = df['platform'].str.split().str[0]
        
        # Calculate session information
        df = self._calculate_listening_sessions(df)
        
        # Create artist and track features
        df['artist_track'] = df['master_metadata_album_artist_name'] + ' - ' + df['master_metadata_track_name']
        
        # Sort by timestamp
        df = df.sort_values('ts')
        
        # Drop duplicates while keeping the latest occurrence
        df = df.drop_duplicates(
            subset=['master_metadata_track_name', 'master_metadata_album_artist_name', 'ts'],
            keep='last'
        )
        
        return df

    def _calculate_listening_sessions(self, 
                                   df: pd.DataFrame, 
                                   session_gap: int = 30) -> pd.DataFrame:
        """
        Calculate listening sessions based on time gaps.
        
        Args:
            df (pd.DataFrame): Input DataFrame with timestamp column
            session_gap (int): Minutes gap to consider as new session
            
        Returns:
            pd.DataFrame: DataFrame with session information added
        """
        # Sort by timestamp
        df = df.sort_values('ts')
        
        # Calculate time difference between consecutive tracks
        df['time_diff'] = df['ts'].diff().dt.total_seconds() / 60
        
        # Assign session IDs where time gap is greater than session_gap
        df['new_session'] = (df['time_diff'] > session_gap) | (df['time_diff'].isna())
        df['session_id'] = df['new_session'].cumsum()
        
        # Calculate session metrics
        session_stats = df.groupby('session_id').agg({
            'ts': ['min', 'max'],
            'master_metadata_track_name': 'count'  # Changed from track_id to master_metadata_track_name
        })
        
        session_stats.columns = ['session_start', 'session_end', 'tracks_in_session']
        session_stats['session_duration'] = (
            session_stats['session_end'] - session_stats['session_start']
        ).dt.total_seconds() / 60
        
        # Merge session stats back to main DataFrame
        df = df.merge(session_stats, left_on='session_id', right_index=True)
        
        return df

    def get_listening_statistics(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Generate comprehensive listening statistics.
        
        Args:
            df (pd.DataFrame): Processed streaming data
            
        Returns:
            Dict[str, any]: Dictionary containing various statistics
        """
        stats = {
            'overall': {
                'total_streams': len(df),
                'total_minutes': df['minutes_played'].sum(),
                'unique_tracks': df['master_metadata_track_name'].nunique(),
                'unique_artists': df['master_metadata_album_artist_name'].nunique(),
                'unique_albums': df['master_metadata_album_album_name'].nunique(),
                'date_range': [df['ts'].min(), df['ts'].max()],
            },
            'top_artists': df['master_metadata_album_artist_name'].value_counts().head(20).to_dict(),
            'top_tracks': df['master_metadata_track_name'].value_counts().head(20).to_dict(),
            'yearly_stats': df.groupby('year').agg({
                'master_metadata_track_name': 'count',
                'minutes_played': 'sum',
                'master_metadata_album_artist_name': 'nunique'
            }).to_dict(),
            'platform_usage': df['platform_name'].value_counts().to_dict(),
            'hourly_distribution': df['hour'].value_counts().sort_index().to_dict(),
            'daily_distribution': df['day_of_week'].value_counts().sort_index().to_dict(),
            'listening_types': df['listen_type'].value_counts().to_dict(),
        }
        
        return stats

    def save_processed_data(self, df: pd.DataFrame, filename: str = 'complete_history.parquet'):
        """
        Save processed data to parquet format.
        
        Args:
            df (pd.DataFrame): Processed data to save
            filename (str): Name of the output file
        """
        output_path = self.data_dir / 'processed' / filename
        df.to_parquet(output_path)
        self.logger.info(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    # Example usage
    processor = SpotifyHistoricalDataProcessor()
    
    # Process all streaming history files
    history_df = processor.process_multiple_history_files("data/raw/Streaming_History_Audio_*.json")
    
    # Generate statistics
    stats = processor.get_listening_statistics(history_df)
    
    # Save processed data
    processor.save_processed_data(history_df)
    
    # Print some basic stats
    print("\nBasic Statistics:")
    print(f"Total streams: {stats['overall']['total_streams']:,}")
    print(f"Total minutes: {stats['overall']['total_minutes']:,.2f}")
    print(f"Unique tracks: {stats['overall']['unique_tracks']:,}")
    print(f"Unique artists: {stats['overall']['unique_artists']:,}")
    print("\nDate range:", stats['overall']['date_range'])