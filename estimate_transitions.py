"""Estimate the MDP's transition probabilities from the play-by-play CSVs (MLE).

Run export_data.py first, then:
    python3 estimate_transitions.py

Reads   data/plays_clean.csv
Writes  transitions/play_summary.csv     per (down group, distance): P(turnover), P(first down), mean gain
        transitions/yards_gained.csv     P(yards gained | down group, distance), no-turnover plays
        transitions/field_goal.csv       P(make | kick distance), from a logistic regression
        transitions/punt_outcomes.csv    P(opponent's start yardline | where the punt was taken)

Every estimate is a maximum-likelihood estimate:
- categorical outcomes -> empirical frequencies (count / total)
- field goal make/miss -> logistic regression (Bernoulli likelihood)
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATA = "data/plays_clean.csv"
OUT = "transitions"

# Yards-to-go buckets: short distances get their own bucket because they matter most on 4th down.
TOGO_BINS = [0, 1, 2, 3, 6, 10, 99]
TOGO_LABELS = ["1", "2", "3", "4-6", "7-10", "11+"]

GAIN_MIN, GAIN_MAX = -20, 99   # clip extreme gains/losses
PUNT_BIN = 5                   # bin punting position into 5-yard groups


def load():
    df = pd.read_csv(DATA, low_memory=False)
    print(f"Loaded {len(df):,} plays from {DATA}")
    return df


def play_outcomes(df):
    """Run/pass plays: turnover rate and yards-gained distribution.

    We pool 3rd and 4th downs (down_group = '3-4') because real 4th-down
    attempts are rare and selective; this is a modeling assumption to test later.
    """
    plays = df[df["action"] == "go"].copy()
    plays["down_group"] = np.where(plays["down"] <= 2, "1-2", "3-4")
    plays["togo"] = pd.cut(plays["ydstogo"], TOGO_BINS, labels=TOGO_LABELS)
    plays["first_down"] = (plays["yards_gained"] >= plays["ydstogo"]) & (plays["turnover"] == 0)

    summary = (
        plays.groupby(["down_group", "togo"], observed=True)
        .agg(n=("play_id", "size"),
             p_turnover=("turnover", "mean"),
             p_first_down=("first_down", "mean"),
             mean_gain=("yards_gained", "mean"))
        .round(4)
        .reset_index()
    )

    clean = plays[plays["turnover"] == 0].copy()
    clean["gain"] = clean["yards_gained"].clip(GAIN_MIN, GAIN_MAX).astype(int)
    gains = (
        clean.groupby(["down_group", "togo", "gain"], observed=True).size()
        .rename("count").reset_index()
    )
    gains["prob"] = gains["count"] / gains.groupby(["down_group", "togo"], observed=True)["count"].transform("sum")
    return summary, gains


def field_goals(df):
    """Logistic regression for P(make | kick distance)."""
    fg = df[(df["action"] == "field_goal") & df["kick_distance"].notna()]
    X = fg[["kick_distance"]].to_numpy()
    y = (fg["field_goal_result"] == "made").astype(int).to_numpy()
    model = LogisticRegression(C=1e6).fit(X, y)  # large C = essentially no regularization = plain MLE

    distances = np.arange(18, 71)
    table = pd.DataFrame({
        "kick_distance": distances,
        "p_make": model.predict_proba(distances.reshape(-1, 1))[:, 1].round(4),
    })
    print(f"Field goals: {len(fg):,} attempts, "
          f"intercept={model.intercept_[0]:.3f}, slope={model.coef_[0][0]:.4f}")
    return table


def punts(df):
    """Distribution of where the opponent starts after a punt, by punting position.

    opp_yardline_100 is the opponent's yardline_100 on their first play, i.e. how far
    they are from OUR end zone: 100 - (where the ball ends up, in our yardline_100).
    Bins near the opponent's goal line have few punts, so treat those estimates with care.
    """
    p = df[(df["action"] == "punt") & (df["punt_blocked"] != 1)
           & (df["fumble_lost"] != 1) & df["kick_distance"].notna()].copy()
    ball_ends = p["yardline_100"] - p["kick_distance"] + p["return_yards"].fillna(0)
    p["opp_yardline_100"] = (100 - ball_ends).round().clip(1, 99).astype(int)
    p.loc[p["touchback"] == 1, "opp_yardline_100"] = 80   # touchback: opponent at their 20
    p["punt_from"] = ((p["yardline_100"] - 1) // PUNT_BIN) * PUNT_BIN + 1   # 1-5, 6-10, ...

    dist = (
        p.groupby(["punt_from", "opp_yardline_100"]).size()
        .rename("count").reset_index()
    )
    dist["prob"] = dist["count"] / dist.groupby("punt_from")["count"].transform("sum")
    print(f"Punts: {len(p):,} used")
    return dist, p


def sanity_checks(summary, fg, punt_rows):
    print("\nSanity checks")
    short = summary[(summary["down_group"] == "3-4") & (summary["togo"] == "1")]
    print(f"  3rd/4th & 1 conversion rate:   {short['p_first_down'].iloc[0]:.1%}  (expect ~65-75%)")
    for d in (30, 40, 50, 60):
        print(f"  FG make prob at {d} yards:     {fg.loc[fg['kick_distance'] == d, 'p_make'].iloc[0]:.1%}")
    own30 = punt_rows[punt_rows["yardline_100"].between(68, 72)]
    print(f"  Punt from own 30 -> opponent starts at their own "
          f"{100 - own30['opp_yardline_100'].median():.0f} (median)")


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load()

    summary, gains = play_outcomes(df)
    fg = field_goals(df)
    punt_dist, punt_rows = punts(df)

    summary.to_csv(f"{OUT}/play_summary.csv", index=False)
    gains.to_csv(f"{OUT}/yards_gained.csv", index=False)
    fg.to_csv(f"{OUT}/field_goal.csv", index=False)
    punt_dist.to_csv(f"{OUT}/punt_outcomes.csv", index=False)

    print("\nPlay outcomes by down group and distance:")
    print(summary.to_string(index=False))
    sanity_checks(summary, fg, punt_rows)
    print(f"\nSaved tables to {os.path.abspath(OUT)}/")


if __name__ == "__main__":
    main()
