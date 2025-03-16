# Fantasy Baseball Prediction API

A machine learning-powered API for fantasy baseball player recommendations and projections.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Database Setup](#database-setup)
- [Data Preparation](#data-preparation)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
- [Training the Model](#training-the-model)
- [Data Files](#data-files)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

## Overview

This Fantasy Baseball API uses machine learning to provide fantasy baseball managers with data-driven recommendations and projections. It analyzes historical player statistics to predict fantasy point production and identifies players who best match your team's needs.

The system uses a RandomForest model to make predictions based on a comprehensive set of baseball metrics, and provides an easy-to-use API for integrating these predictions into your fantasy baseball strategy.

## System Architecture

The system consists of three main components:

1. **FastAPI Application** (`app.py`): The main API service that handles requests and serves predictions.
2. **DuckDB Database**: A lightweight, file-based database that stores player information and statistics.
3. **Data Loading Script** (`load_data_to_duckdb.py`): Processes CSV data and populates the database.

## Key Features

- **Player Projections**: Predict fantasy points for any player in the database
- **Team Analysis**: Identify statistical strengths and weaknesses in your roster
- **Player Recommendations**: Get personalized player suggestions based on your team's needs
- **Data Aggregation**: Properly handle players who played for multiple teams in a season
- **Custom Scoring**: Adapt to different fantasy league scoring formats

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Install Dependencies

```bash
pip install fastapi uvicorn pandas numpy scikit-learn duckdb requests
```

### Clone or Download Project Files

Ensure you have the following files:
- `app.py`: Main API application
- `load_data_to_duckdb.py`: Data loading script

## Database Setup

The database will be automatically created when you run the data loading script. By default, it will be named `fantasy_baseball.duckdb` and stored in the project directory.

## Data Preparation

### Required Data Files

You need two CSV files for training and using the system:

1. **player_data.csv**: Contains player statistics (features for the model)
2. **performance_data.csv**: Contains fantasy points (target for the model)

### Data File Specifications

#### player_data.csv

```
player_id,name,team,position,season,avg,obp,slg,ops,hr,r,rbi,sb,k_rate,bb_rate,babip,iso,wrc_plus,era,whip,k_9,bb_9,hr9,fip,xfip,war
```

#### performance_data.csv

```
player_id,name,season,fantasy_points,games_played,points_per_game,league_id
```

### Data Processing

If your raw data doesn't match the required format, use the included data processing utilities to convert it:

1. Put your raw baseball statistics in the project directory
2. Use the field calculation script to generate the required files:

```python
from field_calculations import calculate_missing_fields, split_data_for_training

# Load your raw data
raw_data = pd.read_csv("your_baseball_stats.csv")

# Calculate missing statistics
processed_data = calculate_missing_fields(raw_data)

# Split into required formats
player_data, performance_data = split_data_for_training(processed_data)

# Save to CSV files
player_data.to_csv("processed_data/player_data.csv", index=False)
performance_data.to_csv("processed_data/performance_data.csv", index=False)
```

## Running the API

### Start the Server

```bash
# Start the API server
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

### Load Data and Train Model

In a separate terminal, run:

```bash
# Load data and train model
python load_data_to_duckdb.py
```

## API Endpoints

### Get Player List

```
GET /players
```

Optional query parameters:
- `position`: Filter by position (e.g., "1B", "OF")
- `team`: Filter by team (e.g., "NYY", "LAD")
- `min_points`: Minimum projected fantasy points
- `top`: Limit to top N players

Example: `GET /players?position=OF&top=10`

### Predict Player Performance

```
POST /predict/player/{player_id}
```

Returns projected fantasy points for the specified player.

Example: `POST /predict/player/P105`

### Get Recommendations

```
POST /recommend
```

Request body:
```json
{
  "team_id": "my_team",
  "players": ["P101", "P102", "P105"],
  "league_settings": {
    "position_limits": {"1B": 1, "2B": 1, "SS": 1, "3B": 1, "C": 1, "OF": 3, "P": 5},
    "league_averages": {"hr": 150, "r": 450, "rbi": 450, "sb": 50},
    "position_weight": 5,
    "stat_weights": {"hr": 2, "sb": 2, "avg": 10}
  },
  "num_recommendations": 3
}
```

Returns recommended players based on your team's needs.

### Import Players

```
POST /import/players
```

Request body: Array of player objects

### Train Model

```
POST /train
```

Request body:
```json
{
  "player_data_csv": "processed_data/player_data.csv",
  "performance_data_csv": "processed_data/performance_data.csv"
}
```

Trains the prediction model with the specified data files.

## Training the Model

The model can be trained in two ways:

1. **Via the API**: Send a POST request to `/train` with the CSV file paths
2. **Via the loading script**: Uncomment the `train_model()` line in `load_data_to_duckdb.py`

Training creates two files:
- `model/fantasy_model.pkl`: The trained RandomForest model
- `model/scaler.pkl`: The StandardScaler for feature normalization

## Data Files

### Input Data Structure

For detailed specifications of input data files, see [Data Preparation](#data-preparation).

### Handling Duplicates

The system automatically handles players who played for multiple teams in a season by:
- Aggregating counting stats (HR, RBI, etc.)
- Calculating weighted averages for rate stats (AVG, ERA, etc.)
- Combining all positions played
- Using the most recent team as the player's team

## Advanced Usage

### Custom Fantasy Scoring

To adapt the system to your league's scoring format, modify the `calculate_fantasy_points()` function in `field_calculations.py`:

```python
def calculate_fantasy_points(df):
    """
    Calculate fantasy points based on your league's scoring system
    """
    # Start with zero points
    points = pd.Series(0.0, index=df.index)
    
    # Calculate for batters (non-pitchers)
    batter_mask = ~df['pos'].str.contains('P', na=False)
    # Example scoring formula for batters in points league
    points[batter_mask] = (
        1.0 * df.loc[batter_mask, 'r'] + 
        1.0 * df.loc[batter_mask, 'rbi'] + 
        4.0 * df.loc[batter_mask, 'hr'] + 
        2.0 * df.loc[batter_mask, 'sb'] - 
        0.5 * df.loc[batter_mask, 'so']
    )
    
    # Calculate for pitchers
    pitcher_mask = df['pos'].str.contains('P', na=False)
    # Example scoring formula for pitchers in points league
    points[pitcher_mask] = (
        4.0 * df.loc[pitcher_mask, 'w'] + 
        2.0 * df.loc[pitcher_mask, 'sv'] + 
        1.0 * df.loc[pitcher_mask, 'so'] - 
        2.0 * df.loc[pitcher_mask, 'er'] - 
        1.0 * df.loc[pitcher_mask, 'bb']
    )
    
    return points
```

### Team Analysis

The loading script includes utility functions for analyzing teams and players:

```python
# Get top players at a position
get_top_players_by_position("OF", 10)

# Analyze a team's player statistics
analyze_team_stats("NYY")
```

## Troubleshooting

### Common Issues

#### Duplicate Player Entries

**Issue**: `ConstraintException: Duplicate key violates primary key constraint`

**Solution**: The system should automatically handle duplicates through aggregation. If you still encounter this error, check for inconsistent player_id values or manually aggregate the data.

#### Missing Fields

**Issue**: `KeyError: ['field_name'] not in index`

**Solution**: The script will now automatically add missing fields with default values. If you want more accurate values, modify your data processing to calculate these fields properly.

#### Database Connection Issues

**Issue**: `Error: unable to open database file`

**Solution**: Check file permissions and path. The DuckDB file should be writable by the application.

## Future Enhancements

Potential improvements for future versions:

1. **Web Interface**: Add a frontend for easier interaction
2. **Advanced Models**: Implement more sophisticated prediction algorithms
3. **Real-time Updates**: Add integration with live statistics APIs
4. **Positional Scarcity**: Incorporate position scarcity into player valuations
5. **Draft Assistant**: Add functionality for draft-day recommendations
6. **Multi-Year Projections**: Extend model to provide multi-year player projections
7. **Custom Scoring System Builder**: Interface for easily defining custom scoring formats
8. **Player Comparison Tool**: Visualize head-to-head player comparisons

## License

This project is open source and available under the MIT License.

---

*For additional questions or suggestions, feel free to open an issue or contribute to the project.*
