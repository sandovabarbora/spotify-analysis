import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler
import networkx as nx
from spotify_analyzer import EnhancedSpotifyAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSpotifyDashboard:
    def __init__(self):
        """Initialize the enhanced dashboard"""
        st.set_page_config(
            page_title="Enhanced Spotify Analysis Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize analyzer and state
        self.analyzer = EnhancedSpotifyAnalyzer()
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'time_range' not in st.session_state:
            st.session_state.time_range = 'short_term'
        if 'selected_features' not in st.session_state:
            st.session_state.selected_features = ['popularity', 'engagement_score']

    def create_plotly_figure(self, data, chart_type, **kwargs):
        """Create Plotly figure with error handling"""
        try:
            if isinstance(data, pd.DataFrame) and data.empty:
                return go.Figure()

            fig = None
            
            if chart_type == "line":
                fig = px.line(data, **kwargs)
            elif chart_type == "bar":
                fig = px.bar(data, **kwargs)
            elif chart_type == "scatter":
                fig = px.scatter(data, **kwargs)
            elif chart_type == "histogram":
                fig = px.histogram(data, **kwargs)
            else:
                logger.error(f"Unknown chart type: {chart_type}")
                return go.Figure()
            
            # Update layout
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating chart: {str(e)}")
            return go.Figure()
    
    def overview_page(self):
        """Enhanced overview page with ML insights"""
        st.header("🎵 Advanced Music Analysis")
        
        # Get insights
        insights = self.analyzer.generate_insights()
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_tracks = len(self.analyzer.recent)
            st.metric("Total Tracks", total_tracks)
            
        with col2:
            unique_artists = self.analyzer.recent['artist'].nunique()
            st.metric("Unique Artists", unique_artists)
            
        with col3:
            if 'engagement_score_normalized' in self.analyzer.recent.columns:
                avg_engagement = self.analyzer.recent['engagement_score_normalized'].mean()
                st.metric("Avg Engagement", f"{avg_engagement:.2f}")
            
        with col4:
            diversity_metrics = next(
                (i for i in insights if i['type'] == 'diversity_metrics'), 
                None
            )
            if diversity_metrics:
                st.metric(
                    "Music Diversity", 
                    f"{diversity_metrics['artist_diversity']:.1%}"
                )
        
        # Listening Patterns
        st.subheader("🎧 Listening Pattern Analysis")
        patterns = next(
            (i for i in insights if i['type'] == 'listening_patterns'), 
            None
        )
        
        if patterns and patterns['patterns']:
            # Create DataFrame for visualization
            pattern_data = pd.DataFrame(patterns['patterns'])
            
            fig = self.create_plotly_figure(
                pattern_data,
                "scatter",
                x="avg_popularity",
                y="size",
                size="size",
                color="cluster",
                title="Listening Clusters Analysis",
                labels={
                    "avg_popularity": "Average Popularity",
                    "size": "Number of Tracks"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top Artists Chart
        st.subheader("👨‍🎤 Top Artists")
        top_artists = self.analyzer.get_top_artists()
        if not top_artists.empty:
            artist_data = pd.DataFrame({
                'artist': top_artists.index,
                'plays': top_artists.values
            })
            
            fig = self.create_plotly_figure(
                artist_data,
                "bar",
                x="plays",
                y="artist",
                orientation='h',
                title="Most Played Artists",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Add download button for data
        st.subheader("📥 Download Data")
        csv_data = self.analyzer.recent.to_csv(index=False)
        st.download_button(
            label="Download Recent Tracks Data as CSV",
            data=csv_data,
            file_name="recent_tracks.csv",
            mime="text/csv"
        )
    
    def pattern_analysis_page(self):
        """Enhanced pattern analysis page"""
        st.header("📊 Listening Patterns Deep Dive")
        
        patterns = self.analyzer.get_listening_patterns()
        
        if patterns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Hourly distribution
                if 'hourly' in patterns:
                    hourly_data = pd.DataFrame({
                        'hour': patterns['hourly'].index,
                        'plays': patterns['hourly'].values
                    })
                    
                    fig = self.create_plotly_figure(
                        hourly_data,
                        "line",
                        x="hour",
                        y="plays",
                        title="Hourly Listening Pattern"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Daily distribution
                if 'daily' in patterns:
                    daily_data = pd.DataFrame({
                        'day': patterns['daily'].index,
                        'plays': patterns['daily'].values
                    })
                    
                    fig = self.create_plotly_figure(
                        daily_data,
                        "bar",
                        x="day",
                        y="plays",
                        title="Daily Listening Pattern"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Engagement Analysis
        st.subheader("🎯 Engagement Analysis")
        engagement_stats = self.analyzer.get_engagement_stats()
        
        if engagement_stats:
            if 'top_engaged_tracks' in engagement_stats:
                top_engaged = engagement_stats['top_engaged_tracks']
                if not top_engaged.empty:
                    st.write("Top Engaged Tracks:")
                    st.dataframe(top_engaged)
    
    def recommendation_page(self):
        """Enhanced recommendation page"""
        st.header("🎵 Smart Music Recommendations")
        
        # Track selector
        if not self.analyzer.recent.empty:
            # Create track selection with both name and artist
            track_data = self.analyzer.recent[['name', 'artist']].drop_duplicates()
            track_options = [
                f"{row['name']} - {row['artist']}" 
                for _, row in track_data.iterrows()
            ]
            
            selected = st.selectbox(
                "Select a track:",
                options=track_options
            )
            
            if selected:
                # Split the selection back into name and artist
                selected_track, selected_artist = selected.split(" - ", 1)
                
                # Get recommendations
                recommendations = self.analyzer.get_recommendations(
                    selected_track,
                    selected_artist
                )
                
                if not recommendations.empty:
                    st.subheader("Recommended Tracks")
                    
                    # Create visualization
                    fig = self.create_plotly_figure(
                        recommendations,
                        "bar",
                        x="name",
                        y="similarity_score",
                        color="popularity",
                        title="Track Recommendations",
                        labels={
                            "similarity_score": "Similarity Score",
                            "name": "Track Name"
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display detailed recommendations
                    st.dataframe(recommendations)
                else:
                    st.info("No recommendations found for this track.")
        else:
            st.info("No tracks available for recommendations.")
    
    def artist_network_page(self):
        """Artist network analysis page"""
        st.header("🎸 Artist Connections")
        
        # Create network visualization
        artist_counts = self.analyzer.get_top_artists()
        if not artist_counts.empty:
            # Create network
            G = nx.Graph()
            
            # Add nodes
            for artist, count in artist_counts.items():
                G.add_node(artist, size=count)
            
            # Add edges based on co-occurrence
            for artist1 in G.nodes():
                for artist2 in G.nodes():
                    if artist1 < artist2:  # Avoid duplicate edges
                        # Count co-occurrences within same session
                        if 'session_id' in self.analyzer.recent.columns:
                            cooccurrences = len(
                                self.analyzer.recent[
                                    self.analyzer.recent['session_id'].isin(
                                        self.analyzer.recent[
                                            self.analyzer.recent['artist'] == artist1
                                        ]['session_id']
                                    ) &
                                    (self.analyzer.recent['artist'] == artist2)
                                ]
                            )
                            
                            if cooccurrences > 0:
                                G.add_edge(artist1, artist2, weight=cooccurrences)
            
            # Create positions for visualization
            pos = nx.spring_layout(G)
            
            # Create traces for visualization
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Add edges
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += tuple([x0, x1, None])
                edge_trace['y'] += tuple([y0, y1, None])
            
            # Create node trace
            node_trace = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode='markers+text',
                hoverinfo='text',
                marker=dict(
                    size=[],
                    color=[],
                    colorscale='Viridis',
                    line_width=2
                )
            )
            
            # Add nodes
            for node in G.nodes():
                x, y = pos[node]
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])
                node_trace['text'] += tuple([node])
                node_trace['marker']['size'] += tuple([G.nodes[node]['size'] * 2])
            
            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Artist Connection Network',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No artist network data available.")
    
    def run(self):
        """Main dashboard execution"""
        st.sidebar.title("Navigation")
        
        # Navigation
        page = st.sidebar.radio(
            "Select Page",
            ["Overview", "Pattern Analysis", "Recommendations", "Artist Network"]
        )
        
        # Time range selector
        st.sidebar.subheader("Time Range")
        time_range = st.sidebar.selectbox(
            "Select time range",
            ["short_term", "medium_term", "long_term"],
            key="time_range"
        )
        
        # Feature selector
        st.sidebar.subheader("Analysis Features")
        available_features = ['popularity']
        if 'engagement_score' in self.analyzer.recent.columns:
            available_features.append('engagement_score')
        if 'play_frequency' in self.analyzer.recent.columns:
            available_features.append('play_frequency')
            
        selected_features = st.sidebar.multiselect(
            "Select features to analyze",
            available_features,
            default=st.session_state.selected_features,
            key="selected_features"
        )
        
        # Page routing
        if page == "Overview":
            self.overview_page()
        elif page == "Pattern Analysis":
            self.pattern_analysis_page()
        elif page == "Recommendations":
            self.recommendation_page()
        else:
            self.artist_network_page()

if __name__ == "__main__":
    dashboard = EnhancedSpotifyDashboard()
    dashboard.run()
