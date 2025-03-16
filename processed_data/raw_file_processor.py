import pandas as pd
import numpy as np
import os

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
# wrc_plus may be named differently in the raw data files, so change it to 'wrc_plus' in the processed data
WRC_PLUS_ALIASES = ["wrc+", "wrc_plus", "wrcplus"]
# similar: ['avg', 'k_rate', 'bb_rate', 'babip', 'iso', 'wrc_plus', 'k_9', 'bb_9', 'hr_9', 'xfip']
AVG_ALIASES = ["avg", "batting_avg", "batting_average", "ba"]
K_RATE_ALIASES = ["k_rate", "k%", "k_pct"]
BB_RATE_ALIASES = ["bb_rate", "bb%", "bb_pct"]
BABIP_ALIASES = ["babip"]
ISO_ALIASES = ["iso"]
WRC_PLUS_ALIASES = ["wrc+", "wrc_plus", "wrcplus"]
K_9_ALIASES = ["k_9", "k/9"]
BB_9_ALIASES = ["bb_9", "bb/9"]
HR_9_ALIASES = ["hr_9", "hr/9", "hr9"]
XFIP_ALIASES = ["xfip"]

ALIAS_MAP = {
    "avg": AVG_ALIASES,
    "k_rate": K_RATE_ALIASES,
    "bb_rate": BB_RATE_ALIASES,
    "babip": BABIP_ALIASES,
    "iso": ISO_ALIASES,
    "wrc_plus": WRC_PLUS_ALIASES,
    "k_9": K_9_ALIASES,
    "bb_9": BB_9_ALIASES,
    "hr9": HR_9_ALIASES,
    "xfip": XFIP_ALIASES,
}


def process_raw_files():
    # Process the raw data files into a single file
    # Read in the raw data files and derive the year from the filename
    file_list = os.listdir("raw_data")
    dataframes = []
    for file in file_list:
        year = file.split("_")[0]
        df = pd.read_csv(os.path.join("raw_data", file))
        # Add the year as a new column
        df["year"] = year
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    # reorder the columns to match the desired output
    processed_df = combined_df[
        ["year", "Player-additional"]
        + [
            col
            for col in combined_df.columns
            if col not in ["year", "Player-additional"]
        ]
    ]
    # rename the 'Player-additional' column to 'player_id'
    processed_df.rename(columns={"Player-additional": "player_id"}, inplace=True)
    # rename all columns to lowercase
    processed_df.columns = processed_df.columns.str.lower()
    # rename features to match the required feature names
    processed_df = rename_features(processed_df)
    # calculate missing fields required for the training data format
    processed_df = calculate_missing_fields(processed_df)
    #  save the processed data to CSV files
    processed_df.to_csv("processed_data.csv", index=False)
    # check if all required features are present in the processed data
    check_required_features(processed_df)
    # split the data into player_data.csv and performance_data.csv format
    player_data, performance_data = split_data_for_training(processed_df)
    # save the processed data to CSV files
    player_data.to_csv("player_data.csv", index=False)
    performance_data.to_csv("performance_data.csv", index=False)


def rename_features(df):
    # Rename features to match the required feature names
    for feature, aliases in ALIAS_MAP.items():
        for alias in aliases:
            if alias in df.columns:
                df.rename(columns={alias: feature}, inplace=True)
                break
    return df


def check_required_features(df):
    # Check if all required features are present in the processed data
    missing_features = [feature for feature in FEATURES if feature not in df.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
    else:
        print("All required features present in the processed data.")


def calculate_missing_fields(df):
    """
    Calculate missing fields required for the training data format

    Args:
        df: DataFrame with your existing baseball statistics

    Returns:
        DataFrame with all required fields
    """
    # Create a copy to avoid modifying the original
    result = df.copy()

    # Convert numeric fields that might be strings
    numeric_cols = [
        "pa",
        "ab",
        "h",
        "bb",
        "so",
        "hr",
        "2b",
        "3b",
        "tb",
        "hbp",
        "sf",
        "ip",
        "bf",
    ]
    for col in numeric_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 1. Calculate k_rate (strikeout rate)
    if "so" in result.columns and "pa" in result.columns:
        result["k_rate"] = result["so"] / result["pa"]

    # 2. Calculate bb_rate (walk rate)
    if "bb" in result.columns and "pa" in result.columns:
        result["bb_rate"] = result["bb"] / result["pa"]

    # 3. Calculate babip (Batting Average on Balls In Play)
    # BABIP = (H - HR) / (AB - K - HR + SF)
    if all(col in result.columns for col in ["h", "hr", "ab", "so", "sf"]):
        # If SF is missing, use 0
        if "sf" not in result.columns:
            result["sf"] = 0

        denominator = result["ab"] - result["so"] - result["hr"] + result["sf"]
        # Avoid division by zero
        result["babip"] = np.where(
            denominator > 0, (result["h"] - result["hr"]) / denominator, np.nan
        )

    # 4. Calculate iso (Isolated Power)
    # ISO = SLG - AVG
    if "slg" in result.columns and "avg" in result.columns:
        result["iso"] = result["slg"] - result["avg"]

    # 5. Calculate k_9 (strikeouts per 9 innings) if not already present
    if "so9" in result.columns:
        result["k_9"] = result["so9"]
    elif "so" in result.columns and "ip" in result.columns:
        result["k_9"] = 9 * result["so"] / result["ip"]

    # 5b. Calculate bb_9 (walks per 9 innings) if not already present
    if "bb9" in result.columns:
        result["bb_9"] = result["bb9"]
    elif "bb" in result.columns and "ip" in result.columns:
        result["bb_9"] = 9 * result["bb"] / result["ip"]

    # 6. Calculate xfip (Expected Fielding Independent Pitching)
    # This is a complex metric - we'll use a simplified version or approximate
    # If missing, we can estimate as FIP + small constant
    if "fip" in result.columns:
        # Simple approximation: xFIP ≈ FIP ± 0.2
        result["xfip"] = result["fip"].apply(
            lambda x: x + np.random.uniform(-0.2, 0.2) if pd.notnull(x) else np.nan
        )

    # 7. wrc_plus (Weighted Runs Created Plus)
    # This is complex and requires league averages
    # If you have ops+ (similar concept), we can use it as an approximation
    if "ops+" in result.columns:
        # Approximate wRC+ using OPS+
        result["wrc_plus"] = result["ops+"]
    else:
        # If no OPS+, we can rough estimate from OPS
        # This is very approximate
        if "ops" in result.columns:
            # Very rough approximation
            result["wrc_plus"] = (
                result["ops"] * 100 / 0.710
            ).round()  # 0.710 is approximately league average OPS

    # 8. Format the position field if needed
    if "pos" in result.columns:
        # Remove spaces, standardize format
        result["position"] = result["pos"].str.replace(" ", "").str.replace("/", ",")

    # 9. Handle war (Wins Above Replacement)
    # WAR is usually already in your dataset, but make sure it's properly signed
    # For pitchers, WAR is typically reported as positive when good
    # The model might expect negative WAR for pitchers - check your model requirements

    # 10. Calculate fantasy_points (will depend on your league scoring)
    # This is a placeholder - you'll need to replace with your league's actual formula
    result["fantasy_points"] = calculate_fantasy_points(result)

    return result


def calculate_fantasy_points(df):
    """
    Calculate fantasy points based on standard 5x5 rotisserie scoring.
    This is highly league-dependent - replace with your actual scoring formula.

    For illustration, we'll create a simple formula:
    - Batters: R + RBI + 2*HR + 2*SB - SO/4
    - Pitchers: Win*5 + Save*5 + K - ER - BB
    """
    # Start with zero points
    points = pd.Series(0, index=df.index)

    # Determine if row is pitcher or position player
    # Simplified check - you may need something more robust
    is_pitcher = df["pos"].str.contains("P", na=False)

    # Calculate for batters
    batter_mask = ~is_pitcher
    if (
        "r" in df.columns
        and "rbi" in df.columns
        and "hr" in df.columns
        and "sb" in df.columns
        and "so" in df.columns
    ):
        points[batter_mask] = (
            df.loc[batter_mask, "r"]
            + df.loc[batter_mask, "rbi"]
            + 2 * df.loc[batter_mask, "hr"]
            + 2 * df.loc[batter_mask, "sb"]
            - df.loc[batter_mask, "so"] / 4
        )

    # Calculate for pitchers
    pitcher_mask = is_pitcher
    if all(col in df.columns for col in ["w", "sv", "so", "er"]):
        # Add bb if it exists, otherwise use 0
        bb = df.loc[pitcher_mask, "bb"] if "bb" in df.columns else 0

        points[pitcher_mask] = (
            5 * df.loc[pitcher_mask, "w"]
            + 5 * df.loc[pitcher_mask, "sv"]
            + df.loc[pitcher_mask, "so"]
            - df.loc[pitcher_mask, "er"]
            - bb
        )

    # Scale points to a reasonable fantasy range (e.g., 300-600 points per season)
    # This scaling is arbitrary and should be adjusted based on your league's actual scoring
    points = 300 + (points - points.min()) / (points.max() - points.min()) * 300

    return points.round(1)


def split_data_for_training(df):
    """
    Split the data into player_data.csv and performance_data.csv format

    Args:
        df: DataFrame with all calculated fields

    Returns:
        Tuple of (player_data_df, performance_data_df)
    """
    # Ensure all required fields exist in the DataFrame
    required_fields = [
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

    # Add any missing columns with default value of 0
    for field in required_fields:
        if field not in df.columns:
            print(f"Adding missing column: {field}")
            df[field] = 0

    # Fields for player_data.csv
    player_fields = [
        "player_id",
        "player",
        "team",
        "position",
        "year",
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

    # Rename columns to match expected format
    rename_dict = {"player": "name", "year": "season"}

    # Create player_data DataFrame
    player_data = df[player_fields].copy()
    player_data = player_data.rename(columns=rename_dict)

    # Fields for performance_data.csv
    performance_fields = ["player_id", "player", "year", "fantasy_points", "g"]

    # Create performance_data DataFrame
    performance_data = df[performance_fields].copy()
    performance_data = performance_data.rename(
        columns={"player": "name", "year": "season", "g": "games_played"}
    )

    # Calculate points_per_game
    performance_data["points_per_game"] = (
        performance_data["fantasy_points"] / performance_data["games_played"]
    ).round(2)

    # Add a default league_id
    performance_data["league_id"] = "standard"

    return player_data, performance_data


if __name__ == "__main__":
    process_raw_files()
