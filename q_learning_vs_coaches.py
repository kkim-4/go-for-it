"""How often does the offline Q-learning policy make the same 4th-down call as NFL coaches?

Run after q_learning.py:
    python3 q_learning_vs_coaches.py

Reads   data/fourth_downs.csv, results/q_learned_values.csv
Writes  results/ql_vs_coaches.csv    action shares and agreement, overall and by yards to go
        figures/ql_vs_coaches.png    coaches' most common action vs Q-learning's action

Each real 4th down is mapped to the MDP state (4, yards to go capped at 20, yards from the
opponent's end zone) and looked up in the Q-learning policy.
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


def load_coach_decisions():
    d = pd.read_csv("data/fourth_downs.csv", usecols=["ydstogo", "yardline_100", "action"]).dropna()
    d = d[d["action"].isin(ACTIONS)].copy()
    d["yardline_100"] = d["yardline_100"].astype(int)
    d["togo"] = d["ydstogo"].clip(upper=MAX_TOGO).astype(int)    # the MDP caps yards to go at 20
    d = d[d["togo"] <= d["yardline_100"]]                        # keep states the MDP considers legal
    d = d[~((d["action"] == "field_goal") & (d["yardline_100"] > FG_MAX_YL))]
    d["coach"] = d["action"].map(ACTIONS.index)
    return d


def load_q_policy():
    """Q-learning's best action in every 4th-down state, and whether all its actions were ever updated."""
    q = pd.read_csv("results/q_learned_values.csv")
    q = q[q["down"] == 4].rename(columns={"ydstogo": "togo", "yard_line100": "yardline_100"})
    values = q[["Q_go", "Q_punt", "Q_fg"]].to_numpy(dtype=float)
    values[:, 2] = np.where(q["yardline_100"] <= FG_MAX_YL, values[:, 2], np.nan)   # no field goal past the 50
    q["ql"] = np.nanargmax(values, axis=1)
    # Q starts at 0, and an action nobody took in a state is never updated: it keeps Q = 0.
    q["untried"] = (values == 0).any(axis=1)
    return q[["togo", "yardline_100", "ql", "untried"]]


def summarize(d, key, label):
    g = d.groupby(key, observed=True)
    out = pd.DataFrame({"n": g.size(), "agree": g["agree"].mean()})
    for name, col in (("coach", "coach"), ("ql", "ql")):
        for i, a in enumerate(["go", "punt", "fg"]):
            out[f"{name}_{a}"] = g[col].apply(lambda s, i=i: (s == i).mean())
    out = out.reset_index().rename(columns={key: "bucket"})
    out.insert(0, "group", label)
    out["bucket"] = out["bucket"].astype(str)
    return out


def plot(d, q, path):
    counts = (d.groupby(["togo", "yardline_100"])["coach"].value_counts().unstack(fill_value=0)
              .reindex(columns=[0, 1, 2], fill_value=0))
    coach = counts.idxmax(axis=1).where(counts.sum(axis=1) >= MIN_COACH_N).dropna().astype(int)

    def grid(cells):                              # (yards to go, yardline) -> 10 x 99 grid of actions
        g = np.full((10, 99), np.nan)
        for (t, y), a in cells.items():
            if t <= 10:
                g[t - 1, y - 1] = a
        return g

    ql = q.set_index(["togo", "yardline_100"])["ql"]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True)
    for ax, title, cells in zip(axes, ["Coaches' most common action", "Q-learning"], [coach, ql]):
        ax.imshow(grid(cells), aspect="auto", origin="lower", extent=[0.5, 99.5, 0.5, 10.5],
                  cmap=ListedColormap(COLORS), vmin=-0.5, vmax=2.5, interpolation="nearest")
        ax.set_xlim(99.5, 0.5)
        ax.set_ylabel("Yards to go")
        ax.set_title(title, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[-1].set_xlabel("Yards from opponent's end zone")
    axes[-1].legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS], ["Go for it", "Punt", "Field goal"],
                    ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    fig.text(0.01, 0.005, f"Coach cells need at least {MIN_COACH_N} real decisions; blank = fewer.",
             fontsize=8, color="#6b6a65")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    d = load_coach_decisions().merge(load_q_policy(), on=["togo", "yardline_100"], how="inner")
    d["agree"] = d["coach"] == d["ql"]
    d["togo_bin"] = pd.cut(d["togo"], TOGO_BINS, labels=TOGO_LABELS)
    d["all"] = "all 4th downs"

    summary = pd.concat([summarize(d, "all", "overall"), summarize(d, "togo_bin", "yards to go")],
                        ignore_index=True)
    summary.round(4).to_csv("results/ql_vs_coaches.csv", index=False)

    # ---- printed results
    n = len(d)
    all_row = summary.iloc[0]
    print(f"{n:,} real 4th downs (2015-2024), each looked up in the Q-learning policy\n")
    print(f"{'':14s}{'go':>8s}{'punt':>8s}{'FG':>8s}")
    print(f"{'Coaches':14s}{all_row['coach_go']:8.1%}{all_row['coach_punt']:8.1%}{all_row['coach_fg']:8.1%}")
    print(f"{'Q-learning':14s}{all_row['ql_go']:8.1%}{all_row['ql_punt']:8.1%}{all_row['ql_fg']:8.1%}")
    print(f"\nQ-learning makes the coach's call in {all_row['agree']:.1%} of decisions "
          f"(always punting would match {all_row['coach_punt']:.1%}).")

    full = d[~d["untried"]]
    print(f"In {d['untried'].mean():.0%} of these states some action was never updated (Q still 0); where every "
          f"action was updated ({len(full):,} decisions), agreement is {full['agree'].mean():.1%}.")

    by = summary[summary["group"] == "yards to go"]
    print("\nBy yards to go:")
    print(by[["bucket", "n", "coach_go", "ql_go", "agree"]]
          .rename(columns={"bucket": "yards to go", "coach_go": "coach go", "ql_go": "Q-learning go"})
          .to_string(index=False, formatters={c: "{:.1%}".format for c in ("coach go", "Q-learning go", "agree")}))

    plot(d, load_q_policy(), "figures/ql_vs_coaches.png")
    print("\nSaved results/ql_vs_coaches.csv, figures/ql_vs_coaches.png")


if __name__ == "__main__":
    main()
