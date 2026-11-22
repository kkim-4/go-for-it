"""Compare NFL coaches' real 4th-down calls with the value-iteration and Q-learning policies.

Run after value_iteration.py and q_learning.py:
    python3 q_learning_vs_coaches.py

Reads   data/fourth_downs.csv, results/values.csv, results/q_learned_values.csv
Writes  results/coaches_summary.csv       action shares and agreement, overall and by yards to go
        results/coaches_by_season.csv     go rates and agreement by season
        results/coaches_confusion.csv     value-iteration call x coach call (counts)
        results/costliest_decisions.csv   the 20 real decisions that gave up the most expected points
        figures/policy_vs_coaches.png     coaches' most common call vs value iteration vs Q-learning
        figures/go_rate_by_season.png     how often coaches went for it vs how often VI says to

Each real 4th down is mapped to the MDP state (4, yards to go capped at 20, yards from the
opponent's end zone) and looked up in each policy. Expected points lost = Q(best action) minus
Q(coach's action), from the value-iteration Q-values. Q-learning only has a policy where its data
contained at least one legal action; elsewhere it makes no call ("no data").
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ACTIONS = ["go", "punt", "field_goal"]
MAX_TOGO = 20
FG_MAX_YL = 50                                   # the MDP only allows field goals up to 67 yards
TOGO_BINS = [0, 1, 2, 3, 6, 10, 99]
TOGO_LABELS = ["1", "2", "3", "4-6", "7-10", "11+"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]       # go, punt, field goal (as in the other heatmaps)
MIN_COACH_N = 5                                  # decisions needed for a cell in the coach heatmap
Q_COLS = ["Q_go", "Q_punt", "Q_fg"]


def load_coach_decisions():
    d = pd.read_csv("data/fourth_downs.csv",
                    usecols=["season", "posteam", "ydstogo", "yardline_100", "action"]).dropna()
    d = d[d["action"].isin(ACTIONS)].copy()
    d["yardline_100"] = d["yardline_100"].astype(int)
    d["togo"] = d["ydstogo"].clip(upper=MAX_TOGO).astype(int)    # the MDP caps yards to go at 20
    d = d[d["togo"] <= d["yardline_100"]]                        # keep states the MDP considers legal
    d = d[~((d["action"] == "field_goal") & (d["yardline_100"] > FG_MAX_YL))]
    d["coach"] = d["action"].map(ACTIONS.index)
    return d


def load_vi():
    """Value-iteration best action and Q-values in every 4th-down state."""
    v = pd.read_csv("results/values.csv")
    v = v[v["down"] == 4].rename(columns={"ydstogo": "togo"})
    v["vi"] = v["best"].map(ACTIONS.index)
    return v[["togo", "yardline_100", "vi"] + Q_COLS]


def load_ql():
    """Q-learning's best action in every 4th-down state; -1 where it has no data."""
    q = pd.read_csv("results/q_learned_values.csv")
    q = q[q["down"] == 4].rename(columns={"ydstogo": "togo", "yard_line100": "yardline_100"})
    q["ql"] = q["best"].map({a: i for i, a in enumerate(ACTIONS)}).fillna(-1).astype(int)
    return q[["togo", "yardline_100", "ql"]]


def shares(s):
    return [(s == i).mean() for i in range(3)]


def summarize(d, key, label):
    rows = []
    for bucket, g in d.groupby(key, observed=True):
        covered = g[g["ql"] >= 0]
        row = dict(group=label, bucket=str(bucket), n=len(g),
                   agree_vi=(g["coach"] == g["vi"]).mean(),
                   ep_lost=g["ep_lost"].mean(),
                   ql_coverage=len(covered) / len(g),
                   agree_ql=(covered["coach"] == covered["ql"]).mean() if len(covered) else np.nan)
        for name, s in (("coach", g["coach"]), ("vi", g["vi"]), ("ql", covered["ql"])):
            for a, x in zip(["go", "punt", "fg"], shares(s)):
                row[f"{name}_{a}"] = x
        rows.append(row)
    return pd.DataFrame(rows)


def grid(cells):
    """(yards to go, yardline) -> 10 x 99 grid of actions; blank where there is no call."""
    g = np.full((10, 99), np.nan)
    for (t, y), a in cells.items():
        if t <= 10 and a >= 0:
            g[t - 1, y - 1] = a
    return g


def plot_policies(d, vi, ql, path):
    counts = (d.groupby(["togo", "yardline_100"])["coach"].value_counts().unstack(fill_value=0)
              .reindex(columns=[0, 1, 2], fill_value=0))
    coach = counts.idxmax(axis=1).where(counts.sum(axis=1) >= MIN_COACH_N).dropna().astype(int)
    panels = [("Coaches' most common call", coach),
              ("Optimal policy (value iteration)", vi.set_index(["togo", "yardline_100"])["vi"]),
              ("Offline Q-learning", ql.set_index(["togo", "yardline_100"])["ql"])]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.6), sharex=True)
    for ax, (title, cells) in zip(axes, panels):
        ax.imshow(grid(cells), aspect="auto", origin="lower", extent=[0.5, 99.5, 0.5, 10.5],
                  cmap=ListedColormap(COLORS), vmin=-0.5, vmax=2.5, interpolation="nearest")
        ax.set_xlim(99.5, 0.5)
        ax.set_ylabel("Yards to go")
        ax.set_title(title, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[-1].set_xlabel("Yards from opponent's end zone")
    axes[-1].legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS], ["Go for it", "Punt", "Field goal"],
                    ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    fig.text(0.01, 0.005, f"Blank: coach cells with fewer than {MIN_COACH_N} real decisions; "
             "Q-learning cells with no data.", fontsize=8, color="#6b6a65")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_seasons(by_season, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for col, label, color in (("vi_go", "Value iteration says go", COLORS[0]),
                              ("coach_go", "Coaches went for it", COLORS[1])):
        ax.plot(by_season["season"], by_season[col], color=color, lw=2, marker="o", ms=5, label=label)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Share of real 4th downs")
    ax.set_title("Go-for-it rate: optimal policy vs coaches", loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="center right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    vi, ql = load_vi(), load_ql()
    d = (load_coach_decisions()
         .merge(vi, on=["togo", "yardline_100"], how="inner")
         .merge(ql, on=["togo", "yardline_100"], how="inner"))
    q = d[Q_COLS].to_numpy(dtype=float)
    d["ep_lost"] = np.nanmax(q, axis=1) - q[np.arange(len(d)), d["coach"].to_numpy()]
    d["togo_bin"] = pd.cut(d["togo"], TOGO_BINS, labels=TOGO_LABELS)
    d["all"] = "all 4th downs"

    summary = pd.concat([summarize(d, "all", "overall"), summarize(d, "togo_bin", "yards to go")],
                        ignore_index=True)
    summary.round(4).to_csv("results/coaches_summary.csv", index=False)

    by_season = (d.assign(coach_go=d["coach"] == 0, vi_go=d["vi"] == 0, agree_vi=d["coach"] == d["vi"])
                 .groupby("season")
                 .agg(n=("coach", "size"), coach_go=("coach_go", "mean"), vi_go=("vi_go", "mean"),
                      agree_vi=("agree_vi", "mean"), ep_lost=("ep_lost", "mean"))
                 .reset_index())
    by_season.round(4).to_csv("results/coaches_by_season.csv", index=False)

    confusion = pd.crosstab(d["vi"].map(dict(enumerate(ACTIONS))), d["coach"].map(dict(enumerate(ACTIONS))),
                            rownames=["value iteration"], colnames=["coach"])
    confusion.to_csv("results/coaches_confusion.csv")

    worst = (d.sort_values("ep_lost", ascending=False).head(20)
             .assign(coach=lambda x: x["coach"].map(dict(enumerate(ACTIONS))),
                     best=lambda x: x["vi"].map(dict(enumerate(ACTIONS))))
             [["season", "posteam", "togo", "yardline_100", "coach", "best", "ep_lost"]])
    worst.round(3).to_csv("results/costliest_decisions.csv", index=False)

    # ---- printed results
    a = summary.iloc[0]
    print(f"{len(d):,} real 4th downs (2015-2024)\n")
    print(f"{'':18s}{'go':>8s}{'punt':>8s}{'FG':>8s}")
    print(f"{'Coaches':18s}{a['coach_go']:8.1%}{a['coach_punt']:8.1%}{a['coach_fg']:8.1%}")
    print(f"{'Value iteration':18s}{a['vi_go']:8.1%}{a['vi_punt']:8.1%}{a['vi_fg']:8.1%}")
    print(f"{'Q-learning':18s}{a['ql_go']:8.1%}{a['ql_punt']:8.1%}{a['ql_fg']:8.1%}   (where it has data)")
    print(f"\nCoaches make the value-iteration call in {a['agree_vi']:.1%} of decisions "
          f"(always punting would match {a['coach_punt']:.1%}).")
    print(f"Average expected points given up per 4th down: {a['ep_lost']:.3f}")
    print(f"Q-learning has a policy for {a['ql_coverage']:.1%} of these decisions and matches the coach "
          f"in {a['agree_ql']:.1%} of them.")
    print("\nValue iteration (rows) vs coach (columns):")
    print(confusion.to_string())
    print("\nBy yards to go:")
    by = summary[summary["group"] == "yards to go"]
    print(by[["bucket", "n", "coach_go", "vi_go", "agree_vi", "ep_lost"]]
          .rename(columns={"bucket": "yards to go"})
          .to_string(index=False, formatters={c: "{:.1%}".format for c in ("coach_go", "vi_go", "agree_vi")}
                     | {"ep_lost": "{:.3f}".format}))
    print("\nBy season:")
    print(by_season.to_string(index=False, formatters={c: "{:.1%}".format for c in ("coach_go", "vi_go", "agree_vi")}
                              | {"n": "{:.0f}".format, "season": "{:.0f}".format, "ep_lost": "{:.3f}".format}))

    plot_policies(d, vi, ql, "figures/policy_vs_coaches.png")
    plot_seasons(by_season, "figures/go_rate_by_season.png")
    print("\nSaved results/coaches_*.csv, results/costliest_decisions.csv, "
          "figures/policy_vs_coaches.png, figures/go_rate_by_season.png")


if __name__ == "__main__":
    main()
