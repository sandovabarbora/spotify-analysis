import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import networkx as nx
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple
from prophet import Prophet

class SpotifyMLAnalyzer:
    def __init__(self, data_path: str):
        """Initialize with path to enriched Spotify data."""
        self.df = pd.read_parquet(data_path)
        alt.data_transformers.disable_max_rows()  # Allow larger datasets in Altair
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare data for analysis."""
        # Time features
        self.df['ts'] = pd.to_datetime(self.df['ts'])
        self.df['hour'] = self.df['ts'].dt.hour
        self.df['day_name'] = self.df['ts'].dt.day_name()
        self.df['month'] = self.df['ts'].dt.month
        self.df['year'] = self.df['ts'].dt.year
        self.df['day_of_week'] = self.df['ts'].dt.dayofweek
        self.df['minutes_played'] = self.df['ms_played'] / 60000

    def cluster_tracks(self) -> Tuple[Dict, alt.Chart]:
        """Perform ML-based track clustering using audio features."""
        features = ['danceability', 'energy', 'valence', 'tempo', 'instrumentalness', 
                   'acousticness', 'liveness', 'speechiness']
        
        if not all(feat in self.df.columns for feat in features):
            return {}, None
        
        # Prepare feature matrix
        X = self.df[features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Prepare visualization data
        viz_df = pd.DataFrame({
            'PCA1': X_pca[:, 0],
            'PCA2': X_pca[:, 1],
            'Cluster': clusters.astype(str),
            'Track': self.df['master_metadata_track_name'],
            'Artist': self.df['master_metadata_album_artist_name']
        })
        
        # Create Altair visualization
        cluster_chart = alt.Chart(viz_df).mark_circle(size=60).encode(
            x='PCA1:Q',
            y='PCA2:Q',
            color='Cluster:N',
            tooltip=['Track', 'Artist', 'Cluster']
        ).properties(
            width=600,
            height=400,
            title='Track Clusters Based on Audio Features'
        ).interactive()
        
        # Analyze cluster characteristics
        cluster_stats = self.df.copy()
        cluster_stats['cluster'] = clusters
        cluster_profiles = cluster_stats.groupby('cluster')[features].mean()
        
        # Get most representative tracks for each cluster
        cluster_examples = {}
        for i in range(len(cluster_profiles)):
            cluster_tracks = cluster_stats[cluster_stats['cluster'] == i]
            cluster_examples[i] = cluster_tracks[['master_metadata_track_name', 
                                               'master_metadata_album_artist_name']].head(5).to_dict('records')
        
        return {
            'cluster_profiles': cluster_profiles.to_dict(),
            'cluster_examples': cluster_examples,
            'feature_importance': dict(zip(features, abs(pca.components_[0])))
        }, cluster_chart

    def predict_listening_patterns(self) -> Tuple[Dict, alt.Chart]:
        """Predict future listening patterns using Prophet."""
        # Prepare daily listening data
        daily_streams = self.df.groupby(self.df['ts'].dt.date).size().reset_index()
        daily_streams.columns = ['ds', 'y']
        
        # Train Prophet model
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
        model.fit(daily_streams)
        
        # Make predictions for next 30 days
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        # Prepare visualization data
        viz_df = pd.concat([
            daily_streams.assign(type='Historical'),
            pd.DataFrame({
                'ds': forecast.ds.tail(30),
                'y': forecast.yhat.tail(30)
            }).assign(type='Predicted')
        ])
        
        # Create Altair visualization
        forecast_chart = alt.Chart(viz_df).mark_line(point=True).encode(
            x='ds:T',
            y='y:Q',
            color='type:N',
            tooltip=['ds', 'y']
        ).properties(
            width=800,
            height=400,
            title='Listening Pattern Prediction'
        ).interactive()
        
        # Extract insights
        seasonality = {
            'weekly': dict(zip(range(7), model.weekly.effect)),
            'yearly': dict(zip(range(1, 13), model.yearly.effect)),
            'daily': dict(zip(range(24), model.daily.effect))
        }
        
        return {
            'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_dict('records'),
            'seasonality': seasonality
        }, forecast_chart

    def analyze_genre_evolution(self) -> Tuple[Dict, List[alt.Chart]]:
        """Analyze how genre preferences evolve over time."""
        if 'artist_genres' not in self.df.columns:
            return {}, []
            
        # Explode genres and create time-based aggregations
        genres_df = self.df.explode('artist_genres')
        monthly_genres = genres_df.groupby([
            pd.Grouper(key='ts', freq='M'),
            'artist_genres'
        ]).size().reset_index(name='count')
        
        # Get top genres for each month
        top_monthly = (monthly_genres.sort_values('count', ascending=False)
                      .groupby(pd.Grouper(key='ts', freq='M'))
                      .head(5))
        
        # Create genre evolution chart
        evolution_chart = alt.Chart(monthly_genres).mark_area().encode(
            x='ts:T',
            y='count:Q',
            color='artist_genres:N',
            tooltip=['artist_genres', 'count']
        ).properties(
            width=800,
            height=400,
            title='Genre Evolution Over Time'
        ).interactive()
        
        # Create genre diversity chart
        diversity_by_month = genres_df.groupby(pd.Grouper(key='ts', freq='M'))['artist_genres'].nunique()
        diversity_df = pd.DataFrame({
            'Month': diversity_by_month.index,
            'Unique_Genres': diversity_by_month.values
        })
        
        diversity_chart = alt.Chart(diversity_df).mark_line(point=True).encode(
            x='Month:T',
            y='Unique_Genres:Q',
            tooltip=['Month', 'Unique_Genres']
        ).properties(
            width=800,
            height=300,
            title='Genre Diversity Over Time'
        ).interactive()
        
        # Calculate genre transition probabilities
        genre_transitions = []
        for session_id, group in self.df.groupby('session_id'):
            genres = group.explode('artist_genres')['artist_genres'].tolist()
            for i in range(len(genres)-1):
                if genres[i] and genres[i+1]:  # Check for valid genres
                    genre_transitions.append((genres[i], genres[i+1]))
                    
        transition_df = pd.DataFrame(genre_transitions, columns=['from_genre', 'to_genre'])
        transition_probs = (transition_df.groupby('from_genre')['to_genre']
                          .value_counts(normalize=True)
                          .unstack(fill_value=0))
        
        return {
            'top_genres_by_month': top_monthly.to_dict('records'),
            'genre_diversity_trend': diversity_by_month.to_dict(),
            'genre_transitions': transition_probs.to_dict(),
            'emerging_genres': self._find_emerging_genres(monthly_genres)
        }, [evolution_chart, diversity_chart]

    def _find_emerging_genres(self, monthly_genres: pd.DataFrame) -> Dict:
        """Identify emerging genres based on growth rate."""
        # Calculate genre growth rates
        genre_growth = monthly_genres.pivot(index='ts', columns='artist_genres', values='count')
        genre_growth = genre_growth.fillna(0)
        growth_rates = (genre_growth.iloc[-1] - genre_growth.iloc[0]) / genre_growth.iloc[0]
        growth_rates = growth_rates.replace([np.inf, -np.inf], np.nan).dropna()
        
        return {
            'fastest_growing': growth_rates.nlargest(10).to_dict(),
            'declining': growth_rates.nsmallest(10).to_dict()
        }

    def generate_ml_insights(self) -> Dict:
        """Generate comprehensive ML-based insights."""
        cluster_results, cluster_viz = self.cluster_tracks()
        forecast_results, forecast_viz = self.predict_listening_patterns()
        genre_results, genre_viz = self.analyze_genre_evolution()
        
        return {
            'track_clustering': {
                'results': cluster_results,
                'visualization': cluster_viz
            },
            'listening_forecast': {
                'results': forecast_results,
                'visualization': forecast_viz
            },
            'genre_evolution': {
                'results': genre_results,
                'visualizations': genre_viz
            }
        }

if __name__ == "__main__":
    analyzer = SpotifyMLAnalyzer("data/processed/enriched_history.parquet")
    insights = analyzer.generate_ml_insights()
    
    # Print some key insights
    print("\n🎵 Advanced ML-Based Spotify Analysis Report 🎵")
    print("=" * 50)
    
    if insights['track_clustering']['results']:
        print("\nTrack Clusters:")
        for cluster, profile in insights['track_clustering']['results']['cluster_profiles'].items():
            print(f"\nCluster {cluster} characteristics:")
            for feature, value in profile.items():
                print(f"- {feature}: {value:.3f}")
    
    if insights['listening_forecast']['results']:
        print("\nListening Forecast:")
        forecast = insights['listening_forecast']['results']['forecast'][0]
        print(f"Next day prediction: {forecast['yhat']:.0f} tracks")
        print(f"Prediction range: {forecast['yhat_lower']:.0f} - {forecast['yhat_upper']:.0f} tracks")
    
    if insights['genre_evolution']['results']:
        print("\nEmerging Genres:")
        emerging = insights['genre_evolution']['results']['emerging_genres']['fastest_growing']
        for genre, growth in list(emerging.items())[:5]:
            print(f"- {genre}: {growth:.1%} growth")