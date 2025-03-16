from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os
import json

# Initialize FastAPI app
app = FastAPI(
    title="Fantasy Baseball API",
    description="API for fantasy baseball player recommendations and projections",
)


# Define data models
class Player(BaseModel):
    player_id: str
    name: str
    team: str
    position: List[str]
    stats: Dict[str, float]
    projected_points: float = 0.0


class TeamRoster(BaseModel):
    team_id: str
    players: List[str]  # List of player_ids
    league_settings: Dict[str, Union[float, str, Dict]]


class FantasyRecommendation(BaseModel):
    recommended_players: List[Player]
    reasoning: str


# Database setup
import sqlite3
from contextlib import contextmanager

# Database path
DB_PATH = "fantasy_baseball.db"


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """Initialize the database with required tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Create players table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            team TEXT NOT NULL,
            position TEXT NOT NULL,
            stats TEXT NOT NULL,
            projected_points REAL DEFAULT 0.0
        )
        """)

        # Create teams table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            team_id TEXT PRIMARY KEY,
            players TEXT NOT NULL,
            league_settings TEXT NOT NULL
        )
        """)

        conn.commit()


# Model storage
MODEL = None
SCALER = None
FEATURES = [
    "avg",
    "obp",
    "slg",
    "ops",
    "hr",
    "r",
    "rbi",
    "sb",
    "k_rate",
    "bb_rate",
    "babip",
    "iso",
    "wrc_plus",
    "era",
    "whip",
    "k_9",
    "bb_9",
    "hr9",
    "fip",
    "xfip",
    "war",
]


# Load model
def load_model():
    global MODEL, SCALER
    try:
        if os.path.exists("model/fantasy_model.pkl"):
            MODEL = pickle.load(open("model/fantasy_model.pkl", "rb"))
            SCALER = pickle.load(open("model/scaler.pkl", "rb"))
            print("Model loaded successfully")
        else:
            # Create a simple model if none exists
            print("No model found, initializing with dummy model")
            MODEL = RandomForestRegressor(n_estimators=100, random_state=42)
            SCALER = StandardScaler()
    except Exception as e:
        print(f"Error loading model: {e}")


# Train model function
def train_model(player_data, performance_data):
    global MODEL, SCALER

    # Prepare features and target
    X = player_data[FEATURES].fillna(0)
    y = performance_data["fantasy_points"]

    # Scale features
    SCALER = StandardScaler()
    X_scaled = SCALER.fit_transform(X)

    # Train model
    MODEL = RandomForestRegressor(n_estimators=100, random_state=42)
    MODEL.fit(X_scaled, y)

    # Save model
    os.makedirs("model", exist_ok=True)
    pickle.dump(MODEL, open("model/fantasy_model.pkl", "wb"))
    pickle.dump(SCALER, open("model/scaler.pkl", "wb"))

    return "Model trained successfully"


# Initialize API
@app.on_event("startup")
async def startup_event():
    # Initialize database
    init_database()
    # Load ML model
    load_model()


# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to Fantasy Baseball API", "status": "online"}


@app.get("/players", response_model=List[Player])
def get_players(
    position: Optional[str] = None,
    team: Optional[str] = None,
    min_points: Optional[float] = None,
    top: Optional[int] = None,
):
    """Get list of players with optional filtering"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Build query
        query = "SELECT * FROM players"
        params = []

        # Add filters
        conditions = []
        if position:
            conditions.append("position LIKE ?")
            params.append(f"%{position}%")
        if team:
            conditions.append("team = ?")
            params.append(team)
        if min_points is not None:
            conditions.append("projected_points >= ?")
            params.append(min_points)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Add sorting
        query += " ORDER BY projected_points DESC"

        # Add limit
        if top:
            query += " LIMIT ?"
            params.append(top)

        # Execute query
        cursor.execute(query, params)
        rows = cursor.fetchall()

        # Convert to Player objects
        players = []
        for row in rows:
            player_data = dict(row)
            # Convert JSON strings to Python objects
            player_data["position"] = json.loads(player_data["position"])
            player_data["stats"] = json.loads(player_data["stats"])
            players.append(Player(**player_data))

        return players


@app.post("/predict/player/{player_id}")
def predict_player_performance(player_id: str):
    """Predict fantasy points for a single player"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Get player from database
        cursor.execute("SELECT * FROM players WHERE player_id = ?", (player_id,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Player not found")

        # Convert to Player object
        player_data = dict(row)
        player_data["position"] = json.loads(player_data["position"])
        player_data["stats"] = json.loads(player_data["stats"])
        player = Player(**player_data)

        # Extract features
        features = np.array([[player.stats.get(f, 0) for f in FEATURES]])

        # Scale features
        scaled_features = SCALER.transform(features)

        # Make prediction
        prediction = MODEL.predict(scaled_features)[0]

        # Update player's projected points in database
        cursor.execute(
            "UPDATE players SET projected_points = ? WHERE player_id = ?",
            (float(prediction), player_id),
        )
        conn.commit()

        return {"player_id": player_id, "projected_points": prediction}


@app.post("/recommend", response_model=FantasyRecommendation)
def recommend_players(team: TeamRoster, num_recommendations: int = 3):
    """Recommend players based on team needs"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Save team to database if it doesn't exist
        cursor.execute("SELECT * FROM teams WHERE team_id = ?", (team.team_id,))
        if not cursor.fetchone():
            cursor.execute(
                "INSERT INTO teams (team_id, players, league_settings) VALUES (?, ?, ?)",
                (
                    team.team_id,
                    json.dumps(team.players),
                    json.dumps(team.league_settings),
                ),
            )
            conn.commit()

        # Get current roster players
        roster_players = []
        if team.players:
            placeholders = ",".join(["?"] * len(team.players))
            cursor.execute(
                f"SELECT * FROM players WHERE player_id IN ({placeholders})",
                team.players,
            )
            for row in cursor.fetchall():
                player_data = dict(row)
                player_data["position"] = json.loads(player_data["position"])
                player_data["stats"] = json.loads(player_data["stats"])
                roster_players.append(Player(**player_data))

        # Get all available players (not in team)
        available_players = []
        if team.players:
            placeholders = ",".join(["?"] * len(team.players))
            cursor.execute(
                f"SELECT * FROM players WHERE player_id NOT IN ({placeholders})",
                team.players,
            )
        else:
            cursor.execute("SELECT * FROM players")

        for row in cursor.fetchall():
            player_data = dict(row)
            player_data["position"] = json.loads(player_data["position"])
            player_data["stats"] = json.loads(player_data["stats"])
            available_players.append(Player(**player_data))

        # Analyze team needs based on roster composition
        needs = analyze_team_needs(roster_players, team.league_settings)

        # Score available players based on team needs
        scored_players = []
        for player in available_players:
            score = calculate_player_score(player, needs, team.league_settings)
            scored_players.append((player, score))

        # Sort by score
        scored_players.sort(key=lambda x: x[1], reverse=True)

        # Get top recommendations
        recommendations = [p for p, _ in scored_players[:num_recommendations]]

        # Generate reasoning
        reasoning = generate_recommendation_reasoning(recommendations, needs)

        return FantasyRecommendation(
            recommended_players=recommendations, reasoning=reasoning
        )


@app.post("/import/players")
def import_players(players: List[Player]):
    """Import or update player data"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        count = 0
        for player in players:
            # Convert Python objects to JSON strings for storage
            position_json = json.dumps(player.position)
            stats_json = json.dumps(player.stats)

            # Check if player exists and update or insert accordingly
            cursor.execute(
                "SELECT 1 FROM players WHERE player_id = ?", (player.player_id,)
            )
            if cursor.fetchone():
                # Update existing player
                cursor.execute(
                    """
                    UPDATE players 
                    SET name = ?, team = ?, position = ?, stats = ?, projected_points = ?
                    WHERE player_id = ?
                """,
                    (
                        player.name,
                        player.team,
                        position_json,
                        stats_json,
                        player.projected_points,
                        player.player_id,
                    ),
                )
            else:
                # Insert new player
                cursor.execute(
                    """
                    INSERT INTO players (player_id, name, team, position, stats, projected_points)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        player.player_id,
                        player.name,
                        player.team,
                        position_json,
                        stats_json,
                        player.projected_points,
                    ),
                )
            count += 1

        conn.commit()

    return {"message": f"Imported {count} players successfully"}


@app.post("/train")
def api_train_model(player_data_csv: str, performance_data_csv: str):
    """Train the model with new data"""
    try:
        player_data = pd.read_csv(player_data_csv)
        performance_data = pd.read_csv(performance_data_csv)
        result = train_model(player_data, performance_data)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# Helper functions
def analyze_team_needs(roster, settings):
    """Analyze team needs based on current roster"""
    # This would be more sophisticated in a real implementation
    positions = {}
    stat_totals = {}

    # Count positions
    for player in roster:
        for pos in player.position:
            positions[pos] = positions.get(pos, 0) + 1

        # Sum up stats
        for stat, value in player.stats.items():
            stat_totals[stat] = stat_totals.get(stat, 0) + value

    # Determine position needs
    position_needs = {}
    for pos, count in settings.get("position_limits", {}).items():
        current = positions.get(pos, 0)
        if current < count:
            position_needs[pos] = count - current

    # Determine stat needs based on league averages (mock implementation)
    stat_needs = {}
    league_averages = settings.get("league_averages", {})
    for stat, avg in league_averages.items():
        team_stat = stat_totals.get(stat, 0)
        if team_stat < avg:
            stat_needs[stat] = (avg - team_stat) / max(1, len(roster))

    return {"position_needs": position_needs, "stat_needs": stat_needs}


def calculate_player_score(player, needs, settings):
    """Calculate how well a player addresses team needs"""
    score = player.projected_points

    # Position need bonus
    position_needs = needs.get("position_needs", {})
    for pos in player.position:
        if pos in position_needs and position_needs[pos] > 0:
            score += settings.get("position_weight", 5)

    # Stat need bonus
    stat_needs = needs.get("stat_needs", {})
    for stat, need in stat_needs.items():
        if stat in player.stats:
            contribution = min(player.stats[stat], need)
            score += contribution * settings.get("stat_weights", {}).get(stat, 1)

    return score


def generate_recommendation_reasoning(recommendations, needs):
    """Generate explanation for recommendations"""
    position_needs = needs.get("position_needs", {})
    stat_needs = needs.get("stat_needs", {})

    reasoning = "Recommendations based on: "

    # Add position needs
    if position_needs:
        reasoning += f"Position needs: {', '.join([f'{pos} ({count})' for pos, count in position_needs.items()])}. "

    # Add stat needs
    if stat_needs:
        reasoning += f"Statistical needs: {', '.join([f'{stat} (deficit: {value:.2f})' for stat, value in stat_needs.items()])}. "

    # Add player specific reasoning
    reasoning += "\n\nPlayer highlights:\n"
    for player in recommendations:
        reasoning += f"- {player.name} ({'/'.join(player.position)}): "
        reasoning += f"Projected {player.projected_points:.2f} pts. "

        # Highlight key stats
        top_stats = sorted(player.stats.items(), key=lambda x: x[1], reverse=True)[:3]
        reasoning += f"Strong in {', '.join([f'{stat} ({value:.2f})' for stat, value in top_stats])}.\n"

    return reasoning


# Run with: uvicorn main:app --reload
