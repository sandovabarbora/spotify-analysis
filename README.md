# Spotify Data Analysis

This project provides tools to analyze Spotify listening habits using **Clojure** and **Python**. It enables the extraction of user data from Spotify's API, processing and saving it for further analysis and visualization.

---

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Setup](#setup)
4. [Clojure Component](#clojure-component)
   - [Configuration](#configuration)
   - [Core Logic](#core-logic)
5. [Python Component](#python-component)
   - [Scripts](#scripts)
   - [Dependencies](#dependencies)
   - [Usage](#usage)
6. [Data Files](#data-files)
7. [License](#license)

---

## Overview
The Spotify Data Analysis project uses Spotify's API to retrieve user data and perform analysis on:
- Recently played tracks.
- Top tracks over short, medium, and long time ranges.

The processed data is saved in CSV format and can be visualized using Python dashboards.

---

## Features
- **Authentication**: OAuth 2.0 flow to connect with Spotify's API.
- **Data Retrieval**: Fetches recently played tracks and top tracks for various time ranges.
- **Data Analysis**: Analyzes popularity, frequency, and other key metrics.
- **Visualization**: Python scripts for creating dashboards and charts.

---

## Setup

### Prerequisites
- Spotify Developer Account: [Spotify Developer](https://developer.spotify.com/dashboard/)
- [Leiningen](https://leiningen.org/) for Clojure setup.
- Python 3.8 or higher installed.
- Required Python libraries (see `requirements.txt`).

### Environment Variables
Create a `.env` file in the root directory and include the following keys:


### Installation
1. **Clojure Setup**:
   - Install [Leiningen](https://leiningen.org/).
   - Run `lein deps` to install dependencies.

2. **Python Setup**:
   - Install Python dependencies:
     ```bash
     pip install -r requirements.txt
     ```

---

## Clojure Component

### Configuration
- **File**: `config.clj`
- Loads environment variables using the `dotenv` library.
- Sets default paths and checks for Spotify API credentials.

### Core Logic
- **File**: `core.clj`
- Key Functions:
  - `create-auth-url`: Generates a URL for Spotify API authentication.
  - `get-access-token`: Handles OAuth2 token exchange.
  - `get-top-tracks`: Fetches top tracks for a user.
  - `analyze-top-tracks`: Extracts and processes track details.
  - `save-to-csv`: Saves processed data as CSV files.

### Running the App
To fetch and process Spotify data:
```bash
lein run

The processed data will be saved in the directory specified by `OUTPUT_DIR`. Ensure this is correctly configured in the `.env` file before running the project.

---

## Python Component

### Scripts

1. **`spotify_analyzer.py`**:
   - This script is designed to process Spotify data that has been fetched and saved in CSV format by the Clojure component.
   - It performs detailed analysis, including:
     - Identifying the top artists based on frequency of plays.
     - Calculating track popularity metrics.
     - Analyzing listening trends over time.
   - The script reads the input CSV files (`spotify_recent_tracks.csv`, `spotify_top_tracks_short.csv`, etc.), processes the data, and outputs derived insights or aggregates for further use.

2. **`spotify_dashboard.py`**:
   - This script is designed to build interactive dashboards that visualize Spotify listening habits.
   - Key features include:
     - Displaying trends in recently played tracks.
     - Comparing top tracks across different time ranges.
     - Highlighting the most frequently played artists and tracks.
   - The script uses Python libraries like Streamlit and Plotly to create a user-friendly interface.

Run the dashboard app with:
```bash
streamlit run spotify_dashboard.py

