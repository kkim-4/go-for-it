"""Team-level 4th-down efficiency and state-by-state coach agreement heatmap.

Complements q_learning_vs_coaches.py (which covers overall agreement, confusion
matrix, yards-to-go buckets, season trends, and costliest individual plays).

Run after export_data.py, estimate_transitions.py, and value_iteration.py:
    python compare_coaches.py

Reads:
    data/fourth_downs.csv                 real 4th-down plays (2015-2024)
    results/values.csv                    MDP V(s), Q(s,a), and optimal action

Writes:
    results/coach_by_team.csv             all 32 teams ranked by mean EP lost
    figures/coach_team_rankings.png       horizontal bar chart of team EP lost
    figures/coach_agreement_heatmap.png   continuous agreement rate by state
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOURTH = "data/fourth_downs.csv"
VALUES = "results/values.csv"
ACTIONS = ["go", "punt", "field_goal"]
Q_COL = {"go": "Q_go", "punt": "Q_punt", "field_goal": "Q_fg"}
MAX_TOGO = 20
FG_MAX_YL = 50  # matches MDP: field goals allowed up to 67 yards

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"


def load_and_merge():
    """Join real 4th-down plays with MDP values using the same filters as q_learning_vs_coaches.py."""
    plays = pd.read_csv(FOURTH, low_memory=False)
    plays = plays[plays["action"].isin(ACTIONS)].copy()
    plays["yardline_100"] = plays["yardline_100"].astype(int)
    plays["togo_state"] = plays["ydstogo"].clip(lower=1, upper=MAX_TOGO).astype(int)
    plays = plays[plays["togo_state"] <= plays["yardline_100"]]
    plays = plays[~((plays["action"] == "field_goal") & (plays["yardline_100"] > FG_MAX_YL))]

    vals4 = pd.read_csv(VALUES)
    vals4 = vals4[vals4["down"] == 4].copy()

    df = plays.merge(
        vals4,
        left_on=["down", "togo_state", "yardline_100"],
        right_on=["down", "ydstogo", "yardline_100"],
        how="inner",
        suffixes=("", "_mdp"),
    )

    q_coach = np.full(len(df), np.nan)
    for act, col in Q_COL.items():
        mask = df["action"] == act
        q_coach[mask] = df.loc[mask, col].to_numpy()

    df["Q_coach"] = q_coach
    df["agree"] = df["action"] == df["best"]
    df["ep_loss"] = np.maximum(0.0, df["V"] - df["Q_coach"])
    return df


def compute_team_table(df):
    """Rank all 32 NFL teams by mean expected points lost per 4th down."""
    by_team = (
        df.groupby("posteam")
        .agg(
            n=("play_id", "size"),
            agree_rate=("agree", "mean"),
            coach_go=("action", lambda s: (s == "go").mean()),
            optimal_go=("best", lambda s: (s == "go").mean()),
            mean_ep_loss=("ep_loss", "mean"),
            total_ep_loss=("ep_loss", "sum"),
        )
        .reset_index()
    )

    go_when_opt = (
        df[df["best"] == "go"]
        .groupby("posteam")["agree"]
        .mean()
        .rename("go_when_opt_go")
        .reset_index()
    )

    by_team = (
        by_team.merge(go_when_opt, on="posteam", how="left")
        .sort_values("mean_ep_loss", ascending=True)
        .reset_index(drop=True)
    )
    return by_team


def plot_agreement_heatmap(df):
    """Continuous heatmap of coach-MDP agreement rate over (yardline_100, ydstogo <= 10)."""
    sub = df[df["togo_state"] <= 10]
    stats = (
        sub.groupby(["togo_state", "yardline_100"])["agree"]
        .agg(["mean", "size"])
        .reset_index()
    )

    grid = np.full((10, 99), np.nan)
    for _, row in stats.iterrows():
        if row["size"] >= 3:  # hide sparse cells
            grid[int(row["togo_state"]) - 1, int(row["yardline_100"]) - 1] = row["mean"]

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#f0f0f0")
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        extent=[0.5, 99.5, 0.5, 10.5],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_xlim(99.5, 0.5)
    ax.set_xlabel("Yards from opponent's end zone")
    ax.set_ylabel("Yards to go")
    ax.set_title("Coach agreement rate with optimal 4th-down policy (n >= 3)", loc="left")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Agreement rate")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/coach_agreement_heatmap.png", dpi=150)
    plt.close(fig)


def plot_team_rankings(by_team):
    """Horizontal bar chart of all teams sorted by mean EP lost per 4th down."""
    ordered = by_team.sort_values("mean_ep_loss", ascending=False)
    colors = [
        GREEN if i >= len(ordered) - 5 else (ORANGE if i < 5 else BLUE)
        for i in range(len(ordered))
    ]

    fig, ax = plt.subplots(figsize=(7.0, 7.5))
    ax.barh(ordered["posteam"], ordered["mean_ep_loss"], color=colors, alpha=0.9)
    ax.set_xlabel("Mean expected points lost per 4th down")
    ax.set_title("NFL teams by 4th-down decision efficiency (2015-2024)", loc="left")
    ax.grid(axis="x", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("figures/coach_team_rankings.png", dpi=150)
    plt.close(fig)


def print_summary(by_team):
    best, worst = by_team.iloc[0], by_team.iloc[-1]
    spread = worst["mean_ep_loss"] - best["mean_ep_loss"]

    print(f"Team efficiency spread: {best['posteam']} ({best['mean_ep_loss']:.3f}) "
          f"vs {worst['posteam']} ({worst['mean_ep_loss']:.3f}) "
          f"— gap of {spread:.3f} EP/play")

    print("\nTop 5 teams (lowest mean EP lost):")
    for _, r in by_team.head(5).iterrows():
        print(f"  {r['posteam']:4s}  EP lost {r['mean_ep_loss']:.3f}  "
              f"agree {r['agree_rate']:.1%}  go|opt=go {r['go_when_opt_go']:.1%}  "
              f"coach go {r['coach_go']:.1%}  (n={int(r['n'])})")

    print("\nBottom 5 teams (highest mean EP lost):")
    for _, r in by_team.tail(5).iloc[::-1].iterrows():
        print(f"  {r['posteam']:4s}  EP lost {r['mean_ep_loss']:.3f}  "
              f"agree {r['agree_rate']:.1%}  go|opt=go {r['go_when_opt_go']:.1%}  "
              f"coach go {r['coach_go']:.1%}  (n={int(r['n'])})")


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    df = load_and_merge()
    by_team = compute_team_table(df)
    by_team.round(4).to_csv("results/coach_by_team.csv", index=False)

    plot_agreement_heatmap(df)
    plot_team_rankings(by_team)
    print_summary(by_team)
    print("\nSaved results/coach_by_team.csv, figures/coach_team_rankings.png, "
          "figures/coach_agreement_heatmap.png")


if __name__ == "__main__":
    main()