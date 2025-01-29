import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
import datetime
from spotify_ml_analyzer import SpotifyMLAnalyzer

class SpotifyAnalysisDashboard:
    def __init__(self):
        st.set_page_config(
            page_title="Spotify Listening Analysis",
            page_icon="🎵",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize analyzer
        self.analyzer = SpotifyMLAnalyzer("data/processed/enriched_history.parquet")
        
        # Set up page structure
        self.setup_sidebar()
        self.main_content()

    def setup_sidebar(self):
        """Configure sidebar with filters and navigation."""
        st.sidebar.title("Navigation")
        
        # Page selection
        self.current_page = st.sidebar.radio(
            "Choose Analysis",
            ["Overview", "ML Insights", "Listening Patterns", 
             "Genre Analysis", "Artist Network", "Track Analysis"]
        )
        
        st.sidebar.title("Filters")
        
        # Date range selector
        self.date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(
                self.analyzer.df['ts'].min().date(),
                self.analyzer.df['ts'].max().date()
            ),
            min_value=self.analyzer.df['ts'].min().date(),
            max_value=self.analyzer.df['ts'].max().date()
        )
        
        # Genre filter (if available)
        if 'artist_genres' in self.analyzer.df.columns:
            all_genres = set()
            for genres in self.analyzer.df['artist_genres'].dropna():
                all_genres.update(genres)
            self.selected_genres = st.sidebar.multiselect(
                "Filter by Genres",
                options=sorted(all_genres),
                default=[]
            )
        
        # Minimum popularity filter
        self.min_popularity = st.sidebar.slider(
            "Minimum Track Popularity",
            min_value=0,
            max_value=100,
            value=0
        )

    def main_content(self):
        """Display main content based on selected page."""
        if self.current_page == "Overview":
            self.overview_page()
        elif self.current_page == "ML Insights":
            self.ml_insights_page()
        elif self.current_page == "Listening Patterns":
            self.patterns_page()
        elif self.current_page == "Genre Analysis":
            self.genre_page()
        elif self.current_page == "Artist Network":
            self.artist_network_page()
        elif self.current_page == "Track Analysis":
            self.track_analysis_page()

    def overview_page(self):
        """Display overview statistics and key metrics."""
        st.title("🎵 Your Spotify Listening Overview")
        
        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        filtered_df = self.get_filtered_df()
        
        with col1:
            st.metric(
                "Total Tracks Played",
                f"{len(filtered_df):,}"
            )
            
        with col2:
            unique_tracks = filtered_df['spotify_track_uri'].nunique()
            st.metric(
                "Unique Tracks",
                f"{unique_tracks:,}"
            )
            
        with col3:
            total_time = filtered_df['minutes_played'].sum()
            st.metric(
                "Hours Listened",
                f"{total_time/60:,.1f}"
            )
            
        with col4:
            unique_artists = filtered_df['master_metadata_album_artist_name'].nunique()
            st.metric(
                "Unique Artists",
                f"{unique_artists:,}"
            )
        
        # Listening trends
        st.subheader("Listening Activity Over Time")
        daily_streams = (
            filtered_df.groupby(filtered_df['ts'].dt.date)
            .size()
            .reset_index(name='streams')
        )
        
        trend_chart = alt.Chart(daily_streams).mark_line().encode(
            x=alt.X('ts:T', title='Date'),
            y=alt.Y('streams:Q', title='Tracks Played'),
            tooltip=['ts', 'streams']
        ).properties(
            width=800,
            height=400
        ).interactive()
        
        st.altair_chart(trend_chart, use_container_width=True)
        
        # Top artists and tracks
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Artists")
            top_artists = (
                filtered_df['master_metadata_album_artist_name']
                .value_counts()
                .head(10)
            )
            
            artist_chart = alt.Chart(
                pd.DataFrame({
                    'Artist': top_artists.index,
                    'Plays': top_artists.values
                })
            ).mark_bar().encode(
                y=alt.Y('Artist:N', sort='-x'),
                x='Plays:Q',
                tooltip=['Artist', 'Plays']
            ).properties(
                height=300
            )
            
            st.altair_chart(artist_chart, use_container_width=True)
            
        with col2:
            st.subheader("Top Tracks")
            top_tracks = (
                filtered_df['master_metadata_track_name']
                .value_counts()
                .head(10)
            )
            
            track_chart = alt.Chart(
                pd.DataFrame({
                    'Track': top_tracks.index,
                    'Plays': top_tracks.values
                })
            ).mark_bar().encode(
                y=alt.Y('Track:N', sort='-x'),
                x='Plays:Q',
                tooltip=['Track', 'Plays']
            ).properties(
                height=300
            )
            
            st.altair_chart(track_chart, use_container_width=True)

    def ml_insights_page(self):
        """Display machine learning based insights."""
        st.title("🤖 ML-Powered Insights")
        
        # Track clustering
        st.header("Track Clustering")
        cluster_results, cluster_viz = self.analyzer.cluster_tracks()
        if cluster_viz:
            st.altair_chart(cluster_viz, use_container_width=True)
            
            # Show cluster characteristics
            if cluster_results:
                st.subheader("Cluster Characteristics")
                for cluster, profile in cluster_results['cluster_profiles'].items():
                    with st.expander(f"Cluster {cluster}"):
                        st.write("Audio Features:")
                        for feature, value in profile.items():
                            st.write(f"- {feature}: {value:.3f}")
                        
                        st.write("\nExample Tracks:")
                        for track in cluster_results['cluster_examples'][cluster]:
                            st.write(f"- {track['master_metadata_track_name']} by {track['master_metadata_album_artist_name']}")
        
        # Listening predictions
        st.header("Listening Pattern Predictions")
        forecast_results, forecast_viz = self.analyzer.predict_listening_patterns()
        if forecast_viz:
            st.altair_chart(forecast_viz, use_container_width=True)
            
            with st.expander("Seasonality Insights"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Weekly Pattern")
                    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    weekly = forecast_results['seasonality']['weekly']
                    st.bar_chart(pd.Series(weekly, index=days))
                
                with col2:
                    st.write("Daily Pattern")
                    hours = [f"{h:02d}:00" for h in range(24)]
                    daily = forecast_results['seasonality']['daily']
                    st.bar_chart(pd.Series(daily, index=hours))

    def patterns_page(self):
        """Display listening pattern analysis."""
        st.title("📊 Listening Pattern Analysis")
        
        filtered_df = self.get_filtered_df()
        
        # Hourly patterns
        st.header("Time of Day Patterns")
        hourly_counts = filtered_df['hour'].value_counts().sort_index()
        
        hour_chart = alt.Chart(
            pd.DataFrame({
                'Hour': hourly_counts.index,
                'Plays': hourly_counts.values
            })
        ).mark_line(point=True).encode(
            x=alt.X('Hour:Q', scale=alt.Scale(domain=[0, 23])),
            y='Plays:Q',
            tooltip=['Hour', 'Plays']
        ).properties(
            width=800,
            height=400,
            title="Listening by Hour of Day"
        ).interactive()
        
        st.altair_chart(hour_chart, use_container_width=True)
        
        # Weekly patterns
        st.header("Day of Week Patterns")
        daily_counts = filtered_df['day_name'].value_counts()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_counts = daily_counts.reindex(days)
        
        day_chart = alt.Chart(
            pd.DataFrame({
                'Day': daily_counts.index,
                'Plays': daily_counts.values
            })
        ).mark_bar().encode(
            x='Day:N',
            y='Plays:Q',
            tooltip=['Day', 'Plays']
        ).properties(
            width=800,
            height=400,
            title="Listening by Day of Week"
        )
        
        st.altair_chart(day_chart, use_container_width=True)

    def genre_page(self):
        """Display genre analysis."""
        st.title("🎸 Genre Evolution Analysis")
        
        genre_results, genre_viz = self.analyzer.analyze_genre_evolution()
        
        if genre_viz:
            # Genre evolution chart
            st.header("Genre Trends Over Time")
            st.altair_chart(genre_viz[0], use_container_width=True)
            
            # Genre diversity
            st.header("Genre Diversity")
            st.altair_chart(genre_viz[1], use_container_width=True)
            
            # Emerging genres
            st.header("Emerging Genres")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Fastest Growing")
                emerging = genre_results['emerging_genres']['fastest_growing']
                for genre, growth in list(emerging.items())[:5]:
                    st.write(f"- {genre}: {growth:.1%} growth")
            
            with col2:
                st.subheader("Declining")
                declining = genre_results['emerging_genres']['declining']
                for genre, decline in list(declining.items())[:5]:
                    st.write(f"- {genre}: {decline:.1%} decline")

    def get_filtered_df(self) -> pd.DataFrame:
        """Get DataFrame filtered by sidebar selections."""
        df = self.analyzer.df.copy()
        
        # Date filter
        if len(self.date_range) == 2:
            mask = (df['ts'].dt.date >= self.date_range[0]) & \
                   (df['ts'].dt.date <= self.date_range[1])
            df = df[mask]
        
        # Genre filter
        if hasattr(self, 'selected_genres') and self.selected_genres:
            df = df[df['artist_genres'].apply(
                lambda x: any(genre in x for genre in self.selected_genres)
            )]
        
        # Popularity filter
        if 'popularity' in df.columns:
            df = df[df['popularity'] >= self.min_popularity]
        
        return df

if __name__ == "__main__":
    dashboard = SpotifyAnalysisDashboard()