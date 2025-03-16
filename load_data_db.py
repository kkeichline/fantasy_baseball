import duckdb
import pandas as pd
import os
import json
import requests
from contextlib import contextmanager

# Database path
DB_PATH = "fantasy_baseball.duckdb"


@contextmanager
def get_db_connection():
    """Context manager for DuckDB database connections"""
    conn = duckdb.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """Initialize the DuckDB database with required tables"""
    with get_db_connection() as conn:
        # Create players table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            team VARCHAR NOT NULL,
            position VARCHAR NOT NULL,
            stats VARCHAR NOT NULL,
            projected_points DOUBLE DEFAULT 0.0
        )
        """)

        # Create teams table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            team_id VARCHAR PRIMARY KEY,
            players VARCHAR NOT NULL,
            league_settings VARCHAR NOT NULL
        )
        """)

        print("Database initialized successfully.")


def aggregate_player_stats(player_data_df):
    """
    Aggregates player statistics across multiple teams/entries

    Args:
        player_data_df: DataFrame with player data, potentially containing duplicates

    Returns:
        DataFrame with one row per player_id, with aggregated stats
    """
    # Group by player_id
    grouped = player_data_df.groupby("player_id")

    # Create a new DataFrame for the aggregated data
    aggregated_data = []

    for player_id, group in grouped:
        if len(group) == 1:
            # No aggregation needed
            aggregated_data.append(group.iloc[0].to_dict())
        else:
            # Need to aggregate stats across multiple teams
            first_row = group.iloc[0].to_dict()

            # Keep non-stat fields from the most recent team
            # Sort by season (if it exists) to get the most recent team
            if "season" in group.columns:
                latest_idx = group["season"].idxmax()
            else:
                # Just take the last row otherwise
                latest_idx = group.index[-1]

            latest_row = group.loc[latest_idx].to_dict()

            # Use the last team they played for, or "Multiple" if we can't determine that
            first_row["team"] = latest_row.get("team", "Multiple")

            # Combine positions from all entries (remove duplicates)
            all_positions = []
            for _, row in group.iterrows():
                if pd.notna(row["position"]):
                    if "," in str(row["position"]):
                        all_positions.extend(
                            [pos.strip() for pos in str(row["position"]).split(",")]
                        )
                    else:
                        all_positions.append(str(row["position"]).strip())

            first_row["position"] = ",".join(sorted(set(all_positions)))

            # Aggregate numeric stats
            numeric_stats = [
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

            # Aggregate counting stats by summing
            count_stats = ["hr", "r", "rbi", "sb"]
            for stat in count_stats:
                if stat in group.columns:
                    first_row[stat] = group[stat].sum()

            # For rate stats, take weighted averages if games_played is available
            rate_stats = [
                "avg",
                "obp",
                "slg",
                "ops",
                "era",
                "whip",
                "k_9",
                "bb_9",
                "hr9",
                "fip",
                "xfip",
            ]

            if "games_played" in group.columns:
                weights = group["games_played"]
                for stat in rate_stats:
                    if stat in group.columns:
                        first_row[stat] = (group[stat] * weights).sum() / weights.sum()
            else:
                # If no games_played, just use a simple average
                for stat in rate_stats:
                    if stat in group.columns:
                        first_row[stat] = group[stat].mean()

            # For WAR, sum it up
            if "war" in group.columns:
                first_row["war"] = group["war"].sum()

            aggregated_data.append(first_row)

    return pd.DataFrame(aggregated_data)


def load_csvs_to_duckdb(player_data_path, performance_data_path):
    """
    Load the CSV files into DuckDB and prepare the data for the model

    Args:
        player_data_path: Path to player_data.csv
        performance_data_path: Path to performance_data.csv
    """
    # Read CSVs into pandas DataFrames
    player_data = pd.read_csv(player_data_path)
    performance_data = pd.read_csv(performance_data_path)

    print(f"Loaded {len(player_data)} player records from {player_data_path}")
    print(f"Found {player_data['player_id'].nunique()} unique players in player data")
    print(
        f"Loaded {len(performance_data)} performance records from {performance_data_path}"
    )

    # Check for duplicates in player_data
    duplicate_players = player_data["player_id"].duplicated(keep=False)
    duplicate_count = duplicate_players.sum()

    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate player entries in player_data.csv")
        print("Aggregating statistics for players who played on multiple teams...")

        # Aggregate player stats to resolve duplicates
        player_data = aggregate_player_stats(player_data)
        print(f"After aggregation: {len(player_data)} unique player records")

    # Similarly, check for duplicates in performance_data and aggregate if needed
    if "player_id" in performance_data.columns:
        duplicate_performance = performance_data["player_id"].duplicated(keep=False)
        dup_perf_count = duplicate_performance.sum()

        if dup_perf_count > 0:
            print(
                f"Found {dup_perf_count} duplicate player entries in performance_data.csv"
            )

            # Group by player_id and aggregate
            performance_grouped = (
                performance_data.groupby("player_id")
                .agg(
                    {
                        "name": "first",
                        "season": "first",
                        "fantasy_points": "sum",
                        "games_played": "sum",
                        "points_per_game": lambda x: x.mean(),
                        "league_id": "first",
                    }
                )
                .reset_index()
            )

            # Recalculate points_per_game after aggregation
            performance_grouped["points_per_game"] = (
                performance_grouped["fantasy_points"]
                / performance_grouped["games_played"]
            ).round(2)

            performance_data = performance_grouped
            print(
                f"After aggregation: {len(performance_data)} unique performance records"
            )

    # Merge data to get complete player information
    merged_data = pd.merge(
        player_data,
        performance_data[["player_id", "fantasy_points", "games_played"]],
        on="player_id",
        how="left",
    )

    # Initialize the database
    init_database()

    # Convert the data to the format needed for the database
    players_to_insert = []

    for _, row in merged_data.iterrows():
        # Create stats dictionary from all statistical columns
        stats = {}
        for feature in [
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
        ]:
            if feature in row and not pd.isna(row[feature]):
                stats[feature] = float(row[feature])
            else:
                stats[feature] = 0.0

        # Convert position to proper format (string to list for JSON storage)
        if pd.isna(row["position"]):
            positions = []
        elif "," in str(row["position"]):
            positions = [pos.strip() for pos in str(row["position"]).split(",")]
        else:
            positions = [str(row["position"]).strip()]

        # Get projected points (fantasy_points)
        projected_points = 0.0
        if "fantasy_points" in row and not pd.isna(row["fantasy_points"]):
            projected_points = float(row["fantasy_points"])

        # Create player dictionary
        player = {
            "player_id": row["player_id"],
            "name": row["name"],
            "team": str(row["team"]),
            "position": json.dumps(positions),
            "stats": json.dumps(stats),
            "projected_points": projected_points,
        }

        players_to_insert.append(player)

    # Insert data into DuckDB
    with get_db_connection() as conn:
        # Clear existing data (optional)
        conn.execute("DELETE FROM players")

        # Insert new data
        for player in players_to_insert:
            conn.execute(
                """
                INSERT INTO players (player_id, name, team, position, stats, projected_points)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    player["player_id"],
                    player["name"],
                    player["team"],
                    player["position"],
                    player["stats"],
                    player["projected_points"],
                ),
            )

        print(f"Inserted {len(players_to_insert)} players into the database")

        # Verify the data was inserted correctly
        result = conn.execute("SELECT COUNT(*) FROM players").fetchone()
        print(f"Database now contains {result[0]} players")


def train_model(
    player_data_path, performance_data_path, api_url="http://localhost:8000"
):
    """
    Train the model using the prepared CSV files

    Args:
        player_data_path: Path to player_data.csv
        performance_data_path: Path to performance_data.csv
        api_url: URL of the Fantasy Baseball API
    """
    try:
        # Send training request to API
        response = requests.post(
            f"{api_url}/train",
            json={
                "player_data_csv": player_data_path,
                "performance_data_csv": performance_data_path,
            },
        )

        if response.status_code == 200:
            print("Model trained successfully!")
            print(response.json())
        else:
            print(f"Error training model: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Exception while training model: {e}")
        print("Make sure the API server is running at the specified URL")


# Example usage
if __name__ == "__main__":
    # Set paths to your processed CSV files
    player_data_path = "processed_data/player_data.csv"
    performance_data_path = "processed_data/performance_data.csv"

    # Load the data into DuckDB
    load_csvs_to_duckdb(player_data_path, performance_data_path)

    # Train the model (make sure the API is running first)
    # Uncomment the line below when ready to train
    # train_model(player_data_path, performance_data_path)
