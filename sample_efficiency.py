"""Sample-efficiency comparison: MLE + value iteration vs offline Q-learning.

Run after export_data.py:
    python3 sample_efficiency.py
    python3 sample_efficiency.py --plot-only     # redraw the figure from results/

For each budget (1k, 5k, 25k, 100k plays, and all plays) we sample whole games, seeded, until
the budget is reached, then run both methods on that same subset:
  - MLE + VI:   estimate_transitions' estimators -> value_iteration.build_mdp -> value_iteration
  - Q-learning: q_learning.extract_transitions -> q_learning (alpha 0.1, 100 epochs)

Metrics, on 4th-down states with yards to go <= 10 (945 states):
  agree_ref   share of states whose action matches the full-data MLE + VI policy
  agree_self  share matching the same method's own full-data policy
  ep_gap      mean |V(1st & 10) - nflfastR EP| over yardlines 10-99 (EP from all plays)
  go_share    share of states where "go" is chosen
  coverage    share of states where the method has enough data to choose any action
              (always 100% for MLE + VI, which pools data across states; Q-learning only
              decides where the data contains at least one of the legal actions)

Reads   data/plays_clean.csv
Writes  results/sample_efficiency_runs.csv     one row per method x budget x seed
        results/sample_efficiency_summary.csv  mean (and sd) per method x budget
        figures/sample_efficiency.png          gap vs nflfastR and policy agreement vs plays
"""
import argparse
import contextlib
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import estimate_transitions as et
import value_iteration as vi
import q_learning as ql

SIZES = [1_000, 5_000, 25_000, 100_000, None]   # None = all plays
SEEDS = [0, 1, 2]
METHODS = ["MLE + VI", "Q-learning"]
COLORS = {"MLE + VI": "#2a78d6", "Q-learning": "#eb6834"}
FOURTH = [vi.idx(4, t, y) for t in range(1, 11) for y in range(t, 100)]
YS = np.arange(10, 100)
RUNS = "results/sample_efficiency_runs.csv"
SUMMARY = "results/sample_efficiency_summary.csv"
FIG = "figures/sample_efficiency.png"


def sample_games(df, n, seed):
    """Whole games in random order until at least n plays"""
    if n is None:
        return df
    games = df["game_id"].drop_duplicates().sample(frac=1, random_state=seed).to_numpy()
    sizes = df.groupby("game_id").size().reindex(games).to_numpy()
    k = int(np.searchsorted(np.cumsum(sizes), n)) + 1
    return df[df["game_id"].isin(set(games[:k]))]


def mle_tables(df):
    """In-memory estimate_transitions.main + value_iteration.load_tables"""
    with contextlib.redirect_stdout(io.StringIO()):
        summary, gains = et.play_outcomes(df)
        try:
            fg = et.field_goals(df).set_index("kick_distance")["p_make"]
        except ValueError: # only one class in the sample
            kicks = df[df["action"] == "field_goal"]
            # no kicks in the sample: use the make rate over all plays (6,302 of 7,307 = 86.2%)
            p = (kicks["field_goal_result"] == "made").mean() if len(kicks) else 0.862
            fg = pd.Series(p, index=np.arange(18, 71))
        punt_dist, _ = et.punts(df)

    play = {}
    for _, row in summary.iterrows():
        g = gains[(gains["down_group"] == row["down_group"]) & (gains["togo"] == row["togo"])]
        if len(g):
            play[(row["down_group"], row["togo"])] = (row["p_turnover"], g["gain"].to_numpy(),
                                                      g["prob"].to_numpy())
    for grp in ("1-2", "3-4"): # empty bucket -> nearest bucket
        for i, lab in enumerate(vi.TOGO_LABELS):
            if (grp, lab) not in play:
                near = [(abs(j - i), (grp, l)) for j, l in enumerate(vi.TOGO_LABELS) if (grp, l) in play]
                play[(grp, lab)] = play[min(near or [(0, k) for k in play])[1]]
    punt = {b: (g["opp_yardline_100"].to_numpy(), g["prob"].to_numpy())
            for b, g in punt_dist.groupby("punt_from")}
    return play, fg, punt


def run_mle(df):
    R, M = vi.build_mdp(*mle_tables(df))
    V, Q, _ = vi.value_iteration(R, M)
    return V, Q.argmax(axis=0)


def run_ql(df):
    Q, _ = ql.q_learning(ql.extract_transitions(df), alpha=0.1, gamma=1.0, epochs=100)
    return ql.get_values(Q), ql.get_policy(Q)


def task(job):
    method, n, seed, df = job
    t0 = time.time()
    V, pi = (run_mle if method == "MLE + VI" else run_ql)(df)
    return dict(method=method, budget=n, seed=seed, plays=len(df),
                V1=np.array([V[vi.first_down(y)] for y in YS]), pi4=pi[FOURTH], secs=time.time() - t0)


# experiment
def run_experiment():
    df = pd.read_csv("data/plays_clean.csv", low_memory=False)
    ep = (df[(df["down"] == 1) & (df["ydstogo"] == 10)]
          .groupby("yardline_100")["ep"].mean().reindex(YS).to_numpy())

    jobs = [(m, n, seed, sub)
            for n in SIZES for seed in (SEEDS if n is not None else [0])
            for sub in [sample_games(df, n, seed)] for m in METHODS]
    jobs.sort(key=lambda j: -len(j[3]) * (50 if j[0] == "Q-learning" else 1))   # longest first

    workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    print(f"{len(jobs)} runs on {workers} workers")
    results = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(task, jobs):
            results.append(r)
            print(f"  {r['method']:<10}  budget {str(r['budget'] or 'all'):>6}  seed {r['seed']}  "
                  f"{r['plays']:>7,} plays  {r['secs']:5.1f}s")

    ref = {m: next(r["pi4"] for r in results if r["method"] == m and r["budget"] is None) for m in METHODS}
    raw = pd.DataFrame([dict(method=r["method"], budget=r["budget"] or "all", seed=r["seed"],
                             plays=r["plays"],
                             ep_gap=float(np.nanmean(np.abs(r["V1"] - ep))),
                             agree_ref=float((r["pi4"] == ref["MLE + VI"]).mean()),
                             agree_self=float(((r["pi4"] == ref[r["method"]]) & (r["pi4"] >= 0)).mean()),
                             go_share=float((r["pi4"] == vi.PLAY).mean()),
                             coverage=float((r["pi4"] >= 0).mean()),
                             secs=r["secs"]) for r in results])
    order = {b: i for i, b in enumerate([1000, 5000, 25000, 100000, "all"])}
    raw = raw.sort_values(["method", "budget", "seed"],
                          key=lambda s: s.map(order) if s.name == "budget" else s)
    raw.to_csv(RUNS, index=False)
    return raw


def summarize(raw):
    summ = (raw.groupby(["method", "budget"], sort=False)
            .agg(plays=("plays", "mean"), ep_gap=("ep_gap", "mean"), ep_gap_sd=("ep_gap", "std"),
                 agree_ref=("agree_ref", "mean"), agree_self=("agree_self", "mean"),
                 go_share=("go_share", "mean"), coverage=("coverage", "mean"),
                 secs=("secs", "mean"))
            .reset_index())
    summ.round(4).to_csv(SUMMARY, index=False)
    return summ


# ------------------------------------------------------------------ figure
def plot(raw):
    panels = [("ep_gap", "Gap vs nflfastR EP (points)", "Value error, 1st & 10", None),
              ("agree_ref", "Agreement with full-data VI policy", "4th-down policy agreement", (0, 1.02))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (col, ylab, title, ylim) in zip(axes, panels):
        for m in METHODS:
            g = raw[raw["method"] == m].groupby("budget", sort=False).agg(
                x=("plays", "mean"), mean=(col, "mean"), lo=(col, "min"), hi=(col, "max"))
            ax.fill_between(g["x"], g["lo"], g["hi"], color=COLORS[m], alpha=0.15, lw=0)
            ax.plot(g["x"], g["mean"], color=COLORS[m], lw=2, marker="o", ms=6,
                    markeredgecolor="white", markeredgewidth=1.5, label=m)
        ax.set_xscale("log")
        ax.set_xticks([1e3, 5e3, 2.5e4, 1e5, 283613])
        ax.set_xticklabels(["1k", "5k", "25k", "100k", "all\n(284k)"])
        ax.set_xlim(800, 4e5)
        ax.minorticks_off()
        if ylim:
            ax.set_ylim(*ylim)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        else:
            ax.set_ylim(0, None)
        ax.set_xlabel("Plays used (whole games, log scale)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", alpha=0.3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].legend(frameon=False, loc="lower right")
    fig.text(0.01, 0.01, "Lines: mean of 3 seeds (1 run at all plays). Bands: min-max across seeds.\n"
             "Policy agreement over 945 4th-down states with yards to go <= 10; "
             "states where Q-learning has no data count as disagreement.",
             fontsize=8, color="#6b6a65")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true", help=f"redraw {FIG} from {RUNS}")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)

    raw = pd.read_csv(RUNS) if args.plot_only else run_experiment()
    summ = summarize(raw)
    plot(raw)
    print()
    print(summ.round(3).to_string(index=False))
    print(f"\nSaved {RUNS}, {SUMMARY}, {FIG}")


if __name__ == "__main__":
    main()