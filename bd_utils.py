import duckdb
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


# Example of how to use DuckDB for JSON operations
# DuckDB has built-in JSON functions that can be very powerful


def get_players_by_position_with_json(position):
    """Example of using DuckDB's JSON capabilities"""
    with get_db_connection() as conn:
        # Query using DuckDB's JSON functions
        result = conn.execute(
            """
            SELECT *
            FROM players
            WHERE json_contains(position, ?)
            ORDER BY projected_points DESC
        """,
            [f'"{position}"'],
        )

        # Return results
        return result.fetchall()


def get_players_with_specific_stat(stat_name, min_value):
    """Another example showcasing DuckDB's JSON capabilities"""
    with get_db_connection() as conn:
        # Query using DuckDB's JSON extraction
        result = conn.execute(
            """
            SELECT player_id, name, team,
                   json_extract(stats, ?) as stat_value
            FROM players
            WHERE CAST(json_extract(stats, ?) AS DOUBLE) >= ?
            ORDER BY CAST(json_extract(stats, ?) AS DOUBLE) DESC
        """,
            [f"$.{stat_name}", f"$.{stat_name}", min_value, f"$.{stat_name}"],
        )

        return result.fetchall()


# Additional DuckDB-specific functionality examples


def complex_player_analytics():
    """Demonstrate some of DuckDB's analytics capabilities"""
    with get_db_connection() as conn:
        # Calculate team averages for a specific stat
        team_averages = conn.execute("""
            SELECT team, 
                   AVG(CAST(json_extract(stats, '$.hr') AS DOUBLE)) as avg_hr,
                   AVG(CAST(json_extract(stats, '$.avg') AS DOUBLE)) as avg_batting,
                   COUNT(*) as player_count
            FROM players
            GROUP BY team
            ORDER BY avg_hr DESC
        """).fetchall()

        # Find players who are above average in their team for a specific stat
        above_average_players = conn.execute("""
            WITH team_avgs AS (
                SELECT team, 
                       AVG(CAST(json_extract(stats, '$.hr') AS DOUBLE)) as team_avg_hr
                FROM players
                GROUP BY team
            )
            SELECT p.player_id, p.name, p.team, 
                   CAST(json_extract(p.stats, '$.hr') AS DOUBLE) as hr,
                   ta.team_avg_hr
            FROM players p
            JOIN team_avgs ta ON p.team = ta.team
            WHERE CAST(json_extract(p.stats, '$.hr') AS DOUBLE) > ta.team_avg_hr
            ORDER BY p.team, hr DESC
        """).fetchall()

        return {
            "team_averages": team_averages,
            "above_average_players": above_average_players,
        }


def export_data_to_parquet():
    """Export player data to Parquet format (DuckDB handles this natively)"""
    with get_db_connection() as conn:
        # Export to Parquet
        conn.execute("""
            COPY (
                SELECT 
                    player_id,
                    name,
                    team,
                    json_extract(stats, '$.avg') as batting_avg,
                    json_extract(stats, '$.hr') as home_runs,
                    json_extract(stats, '$.rbi') as rbis,
                    projected_points
                FROM players
            ) TO 'player_stats.parquet' (FORMAT PARQUET)
        """)

        return {"message": "Data exported to player_stats.parquet"}
