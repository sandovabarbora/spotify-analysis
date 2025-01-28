import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import json
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class EnhancedSpotifyAnalyzer:
    def __init__(self, data_path='../../data/raw/'):
        """Initialize the enhanced analyzer with data path"""
        self.data_path = data_path
        self.load_data()
        
    def load_data(self):
        """Load all Spotify data with enhanced error handling and preprocessing"""
        try:
            # Load current data with robust CSV parsing
            self.recent = pd.read_csv(
                f'{self.data_path}spotify_recent_tracks.csv',
                quoting=1,
                escapechar='\\',
                encoding='utf-8'
            )
            
            # Add unique track identifier
            self.recent['track_id'] = self.recent.apply(
                lambda x: f"{x['name']}|||{x['artist']}", 
                axis=1
            )
            
            # Convert timestamps and add time-based features
            if 'played_at' in self.recent.columns:
                self.recent['played_at'] = pd.to_datetime(self.recent['played_at'])
                self._add_time_features(self.recent)
            
            # Add sophisticated features
            self._add_advanced_features()
            
            # Load top tracks data
            self._load_top_tracks()
            
            # Initialize recommendation system
            self._initialize_recommendation_system()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
            
    def _load_top_tracks(self):
        """Load and process top tracks data"""
        def load_and_process_tracks(filename):
            df = pd.read_csv(f'{self.data_path}{filename}')
            df['track_id'] = df.apply(
                lambda x: f"{x['name']}|||{x['artist']}", 
                axis=1
            )
            return df
            
        self.top_short = load_and_process_tracks('spotify_top_tracks_short.csv')
        self.top_medium = load_and_process_tracks('spotify_top_tracks_medium.csv')
        self.top_long = load_and_process_tracks('spotify_top_tracks_long.csv')
    
    def _add_time_features(self, df):
        """Add time-based features to a dataframe"""
        df['hour'] = df['played_at'].dt.hour
        df['day_of_week'] = df['played_at'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['time_of_day'] = pd.cut(
            df['hour'],
            bins=[-1, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        )
    
    def _add_advanced_features(self):
        """Add sophisticated features for analysis"""
        if 'played_at' in self.recent.columns:
            # Calculate play frequency features
            self.recent['time_diff'] = self.recent['played_at'].diff()
            self.recent['play_frequency'] = self.recent['time_diff'].dt.total_seconds() / 3600
        
        # Add popularity bins for segmentation
        self.recent['popularity_segment'] = pd.qcut(
            self.recent['popularity'], 
            q=5, 
            labels=['Very Niche', 'Niche', 'Moderate', 'Popular', 'Very Popular']
        )
        
        # Calculate artist diversity metrics
        artist_counts = self.recent['artist'].value_counts()
        self.recent['artist_frequency'] = self.recent['artist'].map(artist_counts)
        
        if 'time_diff' in self.recent.columns:
            # Add session features
            self.recent['new_session'] = self.recent['time_diff'] > pd.Timedelta(minutes=30)
            self.recent['session_id'] = self.recent['new_session'].cumsum()
        
        # Calculate engagement scores
        self._calculate_engagement_scores()
    
    def _calculate_engagement_scores(self):
        """Calculate sophisticated engagement metrics"""
        # Base engagement score using play frequency and popularity
        self.recent['engagement_score'] = (
            self.recent['artist_frequency'] * 
            np.log1p(self.recent['popularity'])
        )
        
        if 'hour' in self.recent.columns:
            # Adjust for time of day preferences
            time_weights = self.recent.groupby('hour')['engagement_score'].mean()
            self.recent['time_weighted_engagement'] = (
                self.recent['engagement_score'] * 
                self.recent['hour'].map(time_weights)
            )
        
        # Normalize scores
        scaler = MinMaxScaler()
        self.recent['engagement_score_normalized'] = scaler.fit_transform(
            self.recent[['engagement_score']]
        )
    
    def _initialize_recommendation_system(self):
        """Initialize an advanced recommendation system"""
        try:
            # Combine all track data with unique track IDs
            all_tracks = pd.concat([
                self.recent[['name', 'artist', 'popularity', 'track_id']],
                self.top_short[['name', 'artist', 'popularity', 'track_id']],
                self.top_medium[['name', 'artist', 'popularity', 'track_id']],
                self.top_long[['name', 'artist', 'popularity', 'track_id']]
            ])
            
            # Remove duplicates and reset index
            all_tracks = all_tracks.drop_duplicates('track_id').reset_index(drop=True)
            
            # Create base features DataFrame with proper index
            popularity_features = pd.DataFrame(
                MinMaxScaler().fit_transform(all_tracks[['popularity']]),
                columns=['popularity_normalized'],
                index=all_tracks.index
            )
            
            # Create artist features with same index
            artist_dummies = pd.get_dummies(
                all_tracks['artist'],
                prefix='artist'
            )
            artist_dummies.index = all_tracks.index
            
            # Combine features
            self.track_features = pd.concat(
                [popularity_features, artist_dummies],
                axis=1
            )
            
            # Store track mapping
            self.tracks_list = all_tracks.copy()
            self.track_indices = pd.Series(
                all_tracks.index,
                index=all_tracks['track_id']
            )
            
            # Calculate similarity matrix
            if len(self.track_features) > 1:
                # Use PCA if we have enough features
                n_components = min(50, len(self.track_features.columns))
                if n_components > 0:
                    pca = PCA(n_components=n_components)
                    self.track_features_reduced = pca.fit_transform(self.track_features)
                    self.similarity_matrix = cosine_similarity(self.track_features_reduced)
                else:
                    self.similarity_matrix = cosine_similarity(self.track_features)
            else:
                # If we only have one track, create a 1x1 similarity matrix
                self.similarity_matrix = np.array([[1]])
                
        except Exception as e:
            print(f"Error in recommendation system initialization: {str(e)}")
            raise
            
    def get_recommendations(self, track_name, artist, n_recommendations=5):
        """Get sophisticated music recommendations"""
        try:
            # Create track_id
            track_id = f"{track_name}|||{artist}"
            
            # Get track index
            if track_id not in self.track_indices.index:
                return pd.DataFrame()  # Return empty DataFrame if track not found
                
            idx = self.track_indices[track_id]
            
            # Get similarity scores
            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top similar tracks (excluding the input track)
            sim_scores = [s for s in sim_scores if s[0] != idx][:n_recommendations]
            
            if not sim_scores:
                return pd.DataFrame()
                
            track_indices = [i[0] for i in sim_scores]
            recommendations = self.tracks_list.iloc[track_indices].copy()
            recommendations['similarity_score'] = [i[1] for i in sim_scores]
            
            return recommendations[['name', 'artist', 'popularity', 'similarity_score']]
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return pd.DataFrame()
            
    def get_top_artists(self, n=10):
        """Get top artists by play count"""
        return self.recent['artist'].value_counts().head(n)
        
    def get_popular_tracks(self, n=10):
        """Get most popular tracks"""
        return self.recent.nlargest(n, 'popularity')[['name', 'artist', 'popularity']]
        
    def get_engagement_stats(self):
        """Get engagement statistics"""
        if 'engagement_score_normalized' not in self.recent.columns:
            return {}
            
        return {
            'mean_engagement': self.recent['engagement_score_normalized'].mean(),
            'top_engaged_tracks': self.recent.nlargest(
                5, 
                'engagement_score_normalized'
            )[['name', 'artist', 'engagement_score_normalized']]
        }
        
    def get_listening_patterns(self):
        """Get listening pattern statistics"""
        patterns = {}
        
        if 'hour' in self.recent.columns:
            patterns['hourly'] = self.recent['hour'].value_counts().sort_index()
            
        if 'day_of_week' in self.recent.columns:
            patterns['daily'] = self.recent['day_of_week'].value_counts().sort_index()
            
        if 'time_of_day' in self.recent.columns:
            patterns['time_of_day'] = self.recent['time_of_day'].value_counts()
            
        return patterns
            
    def analyze_artist_network(self):
        """Create and analyze artist collaboration network"""
        if 'played_at' not in self.recent.columns:
            return nx.Graph(), {}
            
        G = nx.Graph()
        
        # Add nodes for each artist
        artists = self.recent['artist'].unique()
        G.add_nodes_from(artists)
        
        # Create edges based on temporal proximity (1 hour window)
        artist_times = {}
        for artist in artists:
            artist_times[artist] = self.recent[
                self.recent['artist'] == artist
            ]['played_at']
        
        for artist1 in artists:
            for artist2 in artists:
                if artist1 != artist2:
                    times1 = artist_times[artist1]
                    times2 = artist_times[artist2]
                    
                    # Check if artists are played within 1 hour of each other
                    for t1 in times1:
                        close_plays = abs(times2 - t1) <= pd.Timedelta(hours=1)
                        if close_plays.any():
                            if G.has_edge(artist1, artist2):
                                G[artist1][artist2]['weight'] += 1
                            else:
                                G.add_edge(artist1, artist2, weight=1)
        
        # Calculate network metrics if the graph is not empty
        if len(G.nodes()) > 0:
            metrics = {
                'centrality': nx.degree_centrality(G),
                'betweenness': nx.betweenness_centrality(G)
            }
            
            # Only calculate communities if we have enough nodes
            if len(G.nodes()) > 2:
                try:
                    metrics['communities'] = list(
                        nx.community.greedy_modularity_communities(G)
                    )
                except:
                    metrics['communities'] = []
        else:
            metrics = {
                'centrality': {},
                'betweenness': {},
                'communities': []
            }
        
        return G, metrics

    def analyze_listening_patterns(self):
            """Enhanced analysis of listening patterns using ML"""
            if len(self.recent) < 10:
                return "Insufficient data for analysis"
            
            # Prepare features for clustering
            features = ['hour', 'day_of_week', 'popularity']
            if 'engagement_score_normalized' in self.recent.columns:
                features.append('engagement_score_normalized')
                
            X = self.recent[features].copy()
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=min(5, len(X)), random_state=42)
            self.recent['listening_cluster'] = kmeans.fit_predict(X_scaled)
            
            # Analyze clusters
            cluster_insights = []
            for cluster in range(len(kmeans.cluster_centers_)):
                cluster_data = self.recent[self.recent['listening_cluster'] == cluster]
                
                insight = {
                    'cluster': cluster,
                    'size': len(cluster_data),
                    'avg_popularity': cluster_data['popularity'].mean(),
                    'peak_hours': cluster_data['hour'].mode().tolist()[:3] if 'hour' in cluster_data.columns else [],
                    'common_artists': cluster_data['artist'].value_counts().head(3).to_dict()
                }
                cluster_insights.append(insight)
            
            return cluster_insights

    def detect_anomalies(self):
        """Detect unusual listening patterns"""
        features = ['popularity']
        if 'engagement_score_normalized' in self.recent.columns:
            features.append('engagement_score_normalized')
        if 'play_frequency' in self.recent.columns:
            features.append('play_frequency')
            
        X = self.recent[features].fillna(0)
        
        if len(X) < 10:
            return pd.DataFrame()  # Return empty DataFrame if insufficient data
            
        # Use Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X)
        
        # Identify anomalous tracks
        self.recent['is_anomaly'] = anomalies == -1
        anomalous_tracks = self.recent[self.recent['is_anomaly']].copy()
        
        return anomalous_tracks[['name', 'artist', 'popularity', 'played_at']]

    def generate_insights(self):
        """Generate comprehensive insights using ML"""
        insights = []
        
        # Analyze listening patterns
        patterns = self.analyze_listening_patterns()
        if patterns != "Insufficient data for analysis":
            insights.append({
                'type': 'listening_patterns',
                'patterns': patterns
            })
        
        # Detect anomalies
        anomalies = self.detect_anomalies()
        if not anomalies.empty:
            insights.append({
                'type': 'anomalies',
                'tracks': anomalies.to_dict('records')
            })
        
        # Calculate diversity metrics
        artist_diversity = len(self.recent['artist'].unique()) / len(self.recent)
        time_diversity = self.recent['hour'].nunique() / 24 if 'hour' in self.recent.columns else 0
        
        insights.append({
            'type': 'diversity_metrics',
            'artist_diversity': artist_diversity,
            'time_diversity': time_diversity
        })
        
        # Generate personalized recommendations
        if len(self.recent) > 0:
            recent_track = self.recent.iloc[0]
            recommendations = self.get_recommendations(
                recent_track['name'], 
                recent_track['artist']
            )
            if not recommendations.empty:
                insights.append({
                    'type': 'recommendations',
                    'based_on': recent_track['name'],
                    'tracks': recommendations.to_dict('records')
                })
        
        return insights

def main():
    analyzer = EnhancedSpotifyAnalyzer()
    
    # Generate comprehensive analysis
    insights = analyzer.generate_insights()
    print("\nKey Insights:")
    for insight in insights:
        print(f"\n{insight['type'].upper()}:")
        print(json.dumps(insight, indent=2))
        
    # Get recommendations for a sample track
    if len(analyzer.recent) > 0:
        sample_track = analyzer.recent.iloc[0]
        print(f"\nRecommendations based on '{sample_track['name']}':")
        recs = analyzer.get_recommendations(sample_track['name'], sample_track['artist'])
        print(recs if not recs.empty else "No recommendations found")

if __name__ == "__main__":
    main()