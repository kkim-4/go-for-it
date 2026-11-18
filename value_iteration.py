"""Build the 4th-down MDP from the MLE tables and solve it with value iteration.

Run after export_data.py and estimate_transitions.py:
    python3 value_iteration.py
    python3 value_iteration.py --gamma-poss 0.9     # ablation: discount the opponent's future

Reads   transitions/*.csv, data/plays_clean.csv (only the 'ep' column, for validation)
Writes  results/values.csv            V(s), Q(s,a) and best action for every state
        results/vi_convergence.csv    max |V_new - V_old| per sweep
        figures/value_vs_nflfastr.png V(1st & 10) vs nflfastR expected points
        figures/policy_heatmap.png    optimal 4th-down action by yards to go x field position

MDP
  State   s = (down 1-4, yards to go 1-20, yardline_100 1-99); yardline_100 = yards from
          the opponent's end zone. Yards to go can't exceed yardline_100.
  Actions downs 1-3: PLAY only. 4th down: PLAY (go for it), PUNT, FG (if yardline_100 <= 50).
  Reward  +7 touchdown, +3 field goal, -2 safety, 0 otherwise.
  Zero-sum: when the opponent gets the ball, our value is MINUS their value from their
          new state, so V(s) = expected (our points - opponent points) = expected net points.

  Bellman update:  Q(s,a) = sum_s' P(s'|s,a) * [ r + sign(s') * gamma * V(s') ]
          sign = +1 if we keep the ball, -1 if the opponent gets it.
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

T = "transitions"
PLAY, PUNT, FG = 0, 1, 2
ACTIONS = ["go", "punt", "field_goal"]
MAX_TOGO = 20
N_STATES = 4 * MAX_TOGO * 99
KICKOFF_YL = 75          # after a score the other team starts around its own 25
SAFETY_YL = 60           # after a safety free kick, the scored-on team kicks; receiver starts ~own 40
FG_MAX_YL = 50           # longest attempt considered: 67 yards
TOGO_BINS = [0, 1, 2, 3, 6, 10, 99]
TOGO_LABELS = ["1", "2", "3", "4-6", "7-10", "11+"]


# ------------------------------------------------------------------ state indexing
def idx(down, togo, yl):
    return ((down - 1) * MAX_TOGO + (togo - 1)) * 99 + (yl - 1)


def first_down(yl):
    """1st & 10 (or 1st & goal inside the 10)."""
    return idx(1, min(10, yl), yl)


KICKOFF = first_down(KICKOFF_YL)


def legal_actions(down, togo, yl):
    if togo > yl:
        return []                      # impossible state
    if down < 4:
        return [PLAY]
    return [PLAY, PUNT] + ([FG] if yl <= FG_MAX_YL else [])


# ------------------------------------------------------------------ load MLE tables
def load_tables():
    summary = pd.read_csv(f"{T}/play_summary.csv", dtype={"togo": str})
    gains = pd.read_csv(f"{T}/yards_gained.csv", dtype={"togo": str})
    play = {}
    for _, row in summary.iterrows():
        key = (row["down_group"], row["togo"])
        g = gains[(gains["down_group"] == key[0]) & (gains["togo"] == key[1])]
        play[key] = (row["p_turnover"], g["gain"].to_numpy(), g["prob"].to_numpy())

    fg = pd.read_csv(f"{T}/field_goal.csv").set_index("kick_distance")["p_make"]

    punts = pd.read_csv(f"{T}/punt_outcomes.csv")
    punt = {b: (g["opp_yardline_100"].to_numpy(), g["prob"].to_numpy())
            for b, g in punts.groupby("punt_from")}
    return play, fg, punt


def play_key(down, togo):
    group = "1-2" if down <= 2 else "3-4"
    bucket = TOGO_LABELS[np.searchsorted(TOGO_BINS, togo) - 1]
    return group, bucket


def punt_bin(yl, punt):
    b = ((yl - 1) // 5) * 5 + 1
    return punt[b] if b in punt else punt[min(punt, key=lambda k: abs(k - b))]


# ------------------------------------------------------------------ transitions
def outcomes(down, togo, yl, a, play, fg, punt):
    """List of (prob, sign, next_state, reward) for taking action a in state s."""
    out = []

    def opponent_ball(opp_yl, p):
        out.append((p, -1, first_down(int(min(max(opp_yl, 1), 99))), 0.0))

    if a == PLAY:
        p_to, gains, probs = play[play_key(down, togo)]
        opponent_ball(100 - yl, p_to)                         # interception / fumble
        for g, pg in zip(gains, probs):
            p = (1 - p_to) * pg
            new = yl - g
            if new <= 0:                                      # touchdown
                out.append((p, -1, KICKOFF, 7.0))
            elif new >= 100:                                  # safety
                out.append((p, -1, first_down(SAFETY_YL), -2.0))
            elif g >= togo:                                   # first down
                out.append((p, +1, first_down(new), 0.0))
            elif down < 4:                                    # next down
                out.append((p, +1, idx(down + 1, min(togo - g, MAX_TOGO), new), 0.0))
            else:                                             # turnover on downs
                opponent_ball(100 - new, p)
    elif a == PUNT:
        opp, probs = punt_bin(yl, punt)
        for o, p in zip(opp, probs):
            opponent_ball(o, p)
    elif a == FG:
        p_make = float(fg.loc[yl + 17])
        out.append((p_make, -1, KICKOFF, 3.0))
        opponent_ball(100 - max(yl + 7, 20), 1 - p_make)     # miss: ball at spot of kick
    return out


def build_mdp(play, fg, punt, gamma_poss=1.0):
    """R[a, s] = expected immediate reward; M[a] = sparse matrix so that Q_a = R[a] + M[a] @ V."""
    R = np.full((3, N_STATES), -np.inf)
    M = []
    for a in range(3):
        rows, cols, vals = [], [], []
        for d in range(1, 5):
            for t in range(1, MAX_TOGO + 1):
                for y in range(1, 100):
                    if a not in legal_actions(d, t, y):
                        continue
                    s = idx(d, t, y)
                    R[a, s] = 0.0
                    for p, sign, nxt, r in outcomes(d, t, y, a, play, fg, punt):
                        R[a, s] += p * r
                        rows.append(s); cols.append(nxt)
                        vals.append(p * sign * (1.0 if sign > 0 else gamma_poss))
        M.append(sp.csr_matrix((vals, (rows, cols)), shape=(N_STATES, N_STATES)))
    return R, M


# ------------------------------------------------------------------ solvers
def q_values(R, M, V):
    Q = np.full_like(R, -np.inf)
    for a in range(3):
        ok = np.isfinite(R[a])
        Q[a, ok] = R[a, ok] + (M[a] @ V)[ok]
    return Q


def value_iteration(R, M, tol=1e-6, max_iter=5000):
    valid = np.isfinite(R).any(axis=0)
    V = np.zeros(N_STATES)
    deltas = []
    for _ in range(max_iter):
        V_new = np.where(valid, q_values(R, M, V).max(axis=0), 0.0)
        deltas.append(np.abs(V_new - V).max())
        V = V_new
        if deltas[-1] < tol:
            break
    return V, q_values(R, M, V), deltas


def policy_iteration(R, M, max_iter=50):
    """Check on value iteration: exact policy evaluation with a sparse linear solve."""
    valid = np.isfinite(R).any(axis=0)
    pi = np.where(np.isfinite(R[PLAY]), PLAY, PUNT)
    I = sp.identity(N_STATES, format="csr")
    for it in range(1, max_iter + 1):
        r = np.where(valid, R[pi, np.arange(N_STATES)], 0.0)
        P = sp.csr_matrix((N_STATES, N_STATES))
        for a in range(3):
            P = P + sp.diags(((pi == a) & valid).astype(float)) @ M[a]
        V = spla.spsolve((I - P).tocsc(), r)
        new_pi = np.where(valid, q_values(R, M, V).argmax(axis=0), pi)
        if (new_pi[valid] == pi[valid]).all():
            return V, pi, it
        pi = new_pi
    return V, pi, max_iter


# ------------------------------------------------------------------ outputs
def save_tables(V, Q, deltas):
    rows = []
    for d in range(1, 5):
        for t in range(1, MAX_TOGO + 1):
            for y in range(t, 100):
                s = idx(d, t, y)
                q = Q[:, s]
                rows.append(dict(down=d, ydstogo=t, yardline_100=y, V=V[s],
                                 Q_go=q[PLAY], Q_punt=q[PUNT] if np.isfinite(q[PUNT]) else np.nan,
                                 Q_fg=q[FG] if np.isfinite(q[FG]) else np.nan,
                                 best=ACTIONS[int(np.argmax(q))]))
    pd.DataFrame(rows).round(4).to_csv("results/values.csv", index=False)
    pd.DataFrame({"sweep": range(1, len(deltas) + 1), "max_change": deltas}).to_csv(
        "results/vi_convergence.csv", index=False)


def plot_value_vs_ep(V):
    plays = pd.read_csv("data/plays_clean.csv", usecols=["down", "ydstogo", "yardline_100", "ep"])
    ep = plays[(plays["down"] == 1) & (plays["ydstogo"] == 10)].groupby("yardline_100")["ep"].mean()
    ys = np.arange(10, 100)
    ours = np.array([V[first_down(y)] for y in ys])
    gap = np.abs(ours - ep.reindex(ys).to_numpy())
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(ys, ours, color="#2a78d6", lw=2, label="Our MDP (value iteration)")
    ax.plot(ep.index, ep.values, color="#eb6834", lw=2, ls="--", label="nflfastR expected points")
    ax.set_xlim(99, 10); ax.set_xlabel("Yards from opponent's end zone"); ax.set_ylabel("Expected points")
    ax.set_title("Value of 1st & 10 by field position", loc="left")
    ax.grid(axis="y", alpha=0.3); ax.legend(frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig("figures/value_vs_nflfastr.png", dpi=150); plt.close(fig)
    return float(np.nanmean(gap))


def plot_policy(Q):
    grid = np.full((10, 99), np.nan)
    for t in range(1, 11):
        for y in range(t, 100):
            grid[t - 1, y - 1] = np.argmax(Q[:, idx(4, t, y)])
    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.imshow(grid, aspect="auto", origin="lower", extent=[0.5, 99.5, 0.5, 10.5],
              cmap=ListedColormap(["#2a78d6", "#eb6834", "#1baf7a"]), vmin=-0.5, vmax=2.5,
              interpolation="nearest")
    ax.set_xlim(99.5, 0.5); ax.set_xlabel("Yards from opponent's end zone"); ax.set_ylabel("Yards to go")
    ax.set_title("Optimal 4th-down decision (value iteration)", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#2a78d6", "#eb6834", "#1baf7a"]]
    ax.legend(handles, ["Go for it", "Punt", "Field goal"], ncol=3, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.25))
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig("figures/policy_heatmap.png", dpi=150); plt.close(fig)
    return float(np.nanmean(grid == PLAY))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma-poss", type=float, default=1.0,
                    help="discount on the opponent's future value (1.0 = none)")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)

    play, fg, punt = load_tables()
    R, M = build_mdp(play, fg, punt, args.gamma_poss)
    V, Q, deltas = value_iteration(R, M)
    V_pi, pi, pi_iters = policy_iteration(R, M)

    valid = np.isfinite(R).any(axis=0)
    agree = (Q.argmax(axis=0)[valid] == pi[valid]).mean()
    save_tables(V, Q, deltas)
    gap = plot_value_vs_ep(V)
    go_share = plot_policy(Q)

    print(f"Legal states:                {valid.sum():,}")
    print(f"Value iteration:             converged in {len(deltas)} sweeps (max change {deltas[-1]:.1e})")
    print(f"Policy iteration check:      {pi_iters} iterations, same action in {agree:.1%} of states,"
          f" max value gap {np.abs(V - V_pi)[valid].max():.1e}")
    print(f"Mean gap vs nflfastR EP:     {gap:.2f} points (1st & 10, yardlines 10-99)")
    spots = {99: "own 1", 75: "own 25", 50: "midfield", 25: "opp 25", 10: "opp 10"}
    print("V(1st & 10):                 " + ", ".join(
        f"{name} {V[first_down(y)]:+.2f}" for y, name in spots.items()))
    print(f"4th-down states where go is optimal (togo <= 10): {go_share:.1%}")
    print("Saved results/values.csv, results/vi_convergence.csv, figures/*.png")


if __name__ == "__main__":
    main()