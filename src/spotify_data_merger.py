import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import logging
import sys

class SpotifyDataMerger:
    def __init__(self, data_dir: str = 'data/'):
        self.data_dir = Path(data_dir)
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('SpotifyDataMerger')
        logger.setLevel(logging.INFO)
        
        # Create handlers for both console and file
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('spotify_merger.log')
        
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

    def validate_file_paths(self, historical_parquet: str, new_data_json: str):
        """Validate that the input files exist and are accessible."""
        historical_path = Path(historical_parquet)
        new_data_path = Path(new_data_json)
        
        if not historical_path.exists():
            raise FileNotFoundError(f"Historical parquet file not found: {historical_parquet}")
        
        if not new_data_path.exists():
            raise FileNotFoundError(f"New data JSON file not found: {new_data_json}")
        
        return historical_path, new_data_path

    def standardize_api_data(self, api_data: pd.DataFrame) -> pd.DataFrame:
        """Standardize API data to match historical data format."""
        api_data = api_data.copy()
        
        # Add missing columns with default values
        default_values = {
            'conn_country': None,
            'ip_addr': None,
            'episode_name': None,
            'episode_show_name': None,
            'spotify_episode_uri': None,
            'audiobook_title': None,
            'audiobook_uri': None,
            'audiobook_chapter_uri': None,
            'audiobook_chapter_title': None,
            'shuffle': False,
            'skipped': False,
            'offline': False,
            'offline_timestamp': None,
            'incognito_mode': False
        }
        
        for col, default_val in default_values.items():
            if col not in api_data.columns:
                api_data[col] = default_val
        
        return api_data

    def merge_data(self, historical_parquet: str, new_data_json: str, output_path: str = None) -> pd.DataFrame:
        """
        Merge historical parquet data with new JSON data.
        
        Args:
            historical_parquet: Path to historical parquet file
            new_data_json: Path to new JSON data or JSON string
            output_path: Optional path to save merged parquet file
        """
        try:
            # Validate files exist
            historical_path, new_data_path = self.validate_file_paths(historical_parquet, new_data_json)
            
            # Load historical data
            self.logger.info(f"Loading historical data from {historical_path}")
            historical_df = pd.read_parquet(historical_path)
            self.logger.info(f"Loaded {len(historical_df)} historical records")
            
            # Load new data
            self.logger.info(f"Loading new data from {new_data_path}")
            with open(new_data_path, 'r', encoding='utf-8') as f:
                new_data = json.load(f)
            new_df = pd.DataFrame(new_data)
            self.logger.info(f"Loaded {len(new_df)} new records")
            
            # Ensure timestamps are in datetime format
            for df in [historical_df, new_df]:
                if 'ts' in df.columns:
                    df['ts'] = pd.to_datetime(df['ts'])
            
            # Standardize new data format
            new_df = self.standardize_api_data(new_df)
            
            # Identify common columns
            common_columns = list(set(historical_df.columns) & set(new_df.columns))
            self.logger.info(f"Merging on {len(common_columns)} common columns")
            
            # Merge dataframes
            merged_df = pd.concat([
                historical_df[common_columns],
                new_df[common_columns]
            ], ignore_index=True)
            
            # Remove duplicates based on timestamp and track URI
            pre_dedup_count = len(merged_df)
            merged_df = merged_df.drop_duplicates(
                subset=['ts', 'spotify_track_uri', 'master_metadata_track_name'],
                keep='last'
            )
            post_dedup_count = len(merged_df)
            
            self.logger.info(f"Removed {pre_dedup_count - post_dedup_count} duplicate entries")
            
            # Sort by timestamp
            merged_df = merged_df.sort_values('ts')
            
            # Save if output path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Saving merged data to {output_path}")
                merged_df.to_parquet(output_path)
            
            self.logger.info(f"Final merged dataset contains {len(merged_df)} rows")
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error during merge: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        merger = SpotifyDataMerger()
        
        # Define paths
        base_dir = Path('data/processed')
        historical_parquet = base_dir / "complete_history.parquet"
        new_data_json = base_dir / "recent_tracks_20250129_094947.json"  # Removed extra .json
        output_path = base_dir / "merged.parquet"
        
        # Print paths for verification
        print("\nProcessing with paths:")
        print(f"Historical data: {historical_parquet}")
        print(f"New data: {new_data_json}")
        print(f"Output: {output_path}\n")
        
        # Merge data
        merged_df = merger.merge_data(historical_parquet, new_data_json, output_path)
        
        # Print summary
        print("\nMerge completed successfully!")
        print(f"Total tracks: {len(merged_df)}")
        print(f"Date range: {merged_df['ts'].min()} to {merged_df['ts'].max()}")
        
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)