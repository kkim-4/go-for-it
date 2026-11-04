"""Export the NFL play-by-play data for the 4th-down MDP project.

Data: nflverse play-by-play (produced by nflfastR), loaded with nflreadpy.
License: CC-BY 4.0 (https://github.com/nflverse/nflverse-data)

Usage:
    pip3 install nflreadpy polars pyarrow
    python3 export_data.py                       # 2015-2024, writes CSVs to ./data
    python3 export_data.py --start 1999 --end 2024
    python3 export_data.py --parquet             # also write Parquet copies

Outputs (in --out):
    plays_raw.csv      every play, only the columns we need (no filtering)
    plays_clean.csv    filtered plays used to estimate the MDP
    fourth_downs.csv   real 4th-down decisions, for comparing with coaches
"""
import argparse
import os

import nflreadpy as nfl
import polars as pl

# Columns we need, grouped by what they are used for.
COLUMNS = [
    # identifiers
    "game_id", "play_id", "season", "week", "posteam", "defteam",
    # state
    "down", "ydstogo", "yardline_100",
    # action
    "play_type",
    # outcomes of a run/pass play
    "yards_gained", "touchdown", "interception", "fumble_lost", "safety",
    "fourth_down_converted", "fourth_down_failed",
    # field goals
    "kick_distance", "field_goal_result",
    # punts
    "return_yards", "touchback", "punt_blocked",
    # filtering
    "qtr", "half_seconds_remaining", "score_differential", "penalty",
    # validation only: nflfastR's expected points before the play
    "ep",
]


def load(start, end):
    seasons = list(range(start, end + 1))
    print(f"Downloading play-by-play for {start}-{end} (first run takes ~1 min)...")
    pbp = nfl.load_pbp(seasons)
    return pbp.select(COLUMNS)


def clean(plays, max_score_diff=16, min_half_seconds=120):
    """Keep normal scrimmage plays where expected points drive decisions."""
    return (
        plays.filter(
            pl.col("down").is_not_null()
            & pl.col("play_type").is_in(["run", "pass", "punt", "field_goal"])
            & (pl.col("penalty") != 1)
            & (pl.col("qtr") <= 4)                                    # no overtime
            & (pl.col("half_seconds_remaining") > min_half_seconds)   # no end-of-half
            & (pl.col("score_differential").abs() <= max_score_diff)  # no blowouts
        )
        .with_columns(
            pl.col("down").cast(pl.Int8),
            pl.col("ydstogo").cast(pl.Int16),
            pl.col("yardline_100").cast(pl.Int16),
            # the action, in MDP terms
            pl.when(pl.col("play_type").is_in(["run", "pass"])).then(pl.lit("go"))
              .otherwise(pl.col("play_type")).alias("action"),
            # turnover on the play (interception or lost fumble)
            ((pl.col("interception") == 1) | (pl.col("fumble_lost") == 1))
              .cast(pl.Int8).alias("turnover"),
        )
    )


def summarize(raw, plays, fourth):
    print(f"\nRaw plays:        {raw.height:,}")
    print(f"Clean plays:      {plays.height:,}")
    print(f"Real 4th downs:   {fourth.height:,}")
    print("\n4th-down decisions by season (share of 4th downs):")
    by_season = (
        fourth.group_by("season")
        .agg(
            pl.len().alias("n"),
            (pl.col("action") == "go").mean().round(3).alias("go"),
            (pl.col("action") == "punt").mean().round(3).alias("punt"),
            (pl.col("action") == "field_goal").mean().round(3).alias("field_goal"),
        )
        .sort("season")
    )
    print(by_season)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--out", default="data")
    ap.add_argument("--parquet", action="store_true", help="also write Parquet copies")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    raw = load(args.start, args.end)
    plays = clean(raw)
    fourth = plays.filter(pl.col("down") == 4)

    outputs = {"plays_raw": raw, "plays_clean": plays, "fourth_downs": fourth}
    for name, df in outputs.items():
        df.write_csv(os.path.join(args.out, f"{name}.csv"))
        if args.parquet:
            df.write_parquet(os.path.join(args.out, f"{name}.parquet"))

    summarize(raw, plays, fourth)
    print(f"\nSaved to {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()