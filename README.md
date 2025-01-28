# Enhanced Spotify Listening Analysis

An advanced music listening analysis tool that provides deep insights into your Spotify listening patterns using machine learning and network analysis.

## Features

### 🎯 Advanced Analytics
- **Listening Pattern Detection**: Uses K-means clustering to identify distinct listening patterns
- **Anomaly Detection**: Identifies unusual listening patterns using Isolation Forest
- **Engagement Scoring**: Calculates sophisticated engagement metrics based on multiple factors
- **Artist Network Analysis**: Analyzes artist relationships and communities using network theory

### 🎵 Smart Recommendations
- Content-based recommendation system using:
  - Track popularity metrics
  - Artist relationships
  - Temporal patterns
  - PCA for dimensionality reduction
  - Cosine similarity for track matching

### 📊 Interactive Dashboard
- Real-time visualization of listening patterns
- Artist network visualization
- Engagement analysis
- Recommendation interface
- Temporal analysis

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- scikit-learn
- networkx
- plotly
- streamlit
- scipy

## Usage

### Data Preparation
Place your Spotify data files in the `data/raw` directory:
- spotify_recent_tracks.csv
- spotify_top_tracks_short.csv
- spotify_top_tracks_medium.csv
- spotify_top_tracks_long.csv

### Running the Analysis
```bash
python src/python/spotify_analyzer.py
```

### Running the Dashboard
```bash
streamlit run src/python/spotify_dashboard.py
```

## Technical Details

### Data Processing
- Robust CSV parsing with error handling
- Automatic feature engineering
- Time-based feature extraction
- Session detection

### Machine Learning Components

#### Pattern Detection
- K-means clustering on temporal and engagement features
- Standardized feature scaling
- Dynamic cluster number selection

#### Recommendation System
1. Feature Engineering:
   - Popularity normalization
   - Artist embeddings
   - PCA dimensionality reduction

2. Similarity Calculation:
   - Cosine similarity matrix
   - Weighted feature importance
   - Temporal context consideration

#### Network Analysis
- Artist collaboration network based on temporal proximity
- Community detection using modularity optimization
- Centrality metrics calculation
- Interactive network visualization

### Engagement Scoring
The engagement score is calculated using multiple factors:
- Play frequency
- Artist diversity
- Time-of-day preferences
- Popularity metrics

## Project Structure
```
spotify-analysis/
├── data/
│   └── raw/
│       ├── spotify_recent_tracks.csv
│       ├── spotify_top_tracks_short.csv
│       ├── spotify_top_tracks_medium.csv
│       └── spotify_top_tracks_long.csv
├── src/
│   └── python/
│       ├── spotify_analyzer.py
│       └── spotify_dashboard.py
└── README.md
```

## API Reference

### EnhancedSpotifyAnalyzer

#### Main Methods
```python
def get_recommendations(track_name, artist, n_recommendations=5)
    """Get sophisticated music recommendations"""

def analyze_listening_patterns()
    """Enhanced analysis of listening patterns using ML"""

def detect_anomalies()
    """Detect unusual listening patterns"""

def analyze_artist_network()
    """Create and analyze artist collaboration network"""

def generate_insights()
    """Generate comprehensive insights using ML"""
```

### EnhancedSpotifyDashboard

#### Main Components
```python
def overview_page()
    """Enhanced overview page with ML insights"""

def pattern_analysis_page()
    """Enhanced pattern analysis page"""

def recommendation_page()
    """Smart recommendation interface"""

def artist_network_page()
    """Artist network visualization and analysis"""
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.