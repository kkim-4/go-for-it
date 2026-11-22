"""In a model-free approach, find the optimal state/action value function using (offline) Q-learning without MLE transition
probability estimates and construct the optimal policy.

Run after export_data.py 
    python3 q_learning.py 

Reads data/plays_clean.csv (only the 'ep' column for validation)

Writes  figures/q_learn_vs_nflfastr.png V(1st & 10) vs nflfastR expected points
        figures/q_policy_heatmap.png    optimal 4th-down action by yards to go x field position

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


DATA = "data/plays_clean.csv"

PLAY, PUNT, FG = 0, 1, 2
ACTIONS = ["go", "punt", "field_goal"]
MAX_TOGO = 20
N_STATES = 4 * MAX_TOGO * 99 # 4 downs possible, 20 values of yards to go possible, 99 yards to opponenent endzone possible
KICKOFF_YL = 75          # after a score the other team starts around its own 25
SAFETY_YL = 60           # after a safety free kick, the scored-on team kicks; receiver starts ~own 40
FG_MAX_YL = 50           # longest attempt considered: 67 yards
TOGO_BINS = [0, 1, 2, 3, 6, 10, 99]
TOGO_LABELS = ["1", "2", "3", "4-6", "7-10", "11+"]

GAMMA = 1.0
ALPHA = 0.1

NUM_EPOCHS = 100

# ------------------------------------------------------------------ loading cleaned data

def load():
    df = pd.read_csv(DATA, low_memory=False)
    print(f"Loaded {len(df):,} plays from {DATA}")
    return df

# ------------------------------------------------------------------ Extracting states and transitions from observations

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

def get_state(row):
    """
    Converts a data row for a play into one of the possible state indexes. 

    Returns:
        int: State index
    """

    if(row["play_type"] == "kickoff"):
        return KICKOFF
    
    down = int(row["down"])
    togo = min(int(row["ydstogo"]), MAX_TOGO)
    yl = int(row["yardline_100"])

    if not (1 <= down <= 4):
        return None

    if not (1 <= togo <= MAX_TOGO):
        return None

    if not (1 <= yl <= 99):
        return None

    if togo > yl:
        return None

    return idx(down, togo, yl)

def get_action(row):
    """
    Converts a play decision into an action index.

    Returns:
        int: Action index
    """

    action = row["action"]

    return ACTIONS.index(action)


def get_reward(row):
    """
    Determines the immediate reward from a play. +7 if touchdown, +3 if field goal, -2 if safety, and 0 otherwise.

    Returns:
        float
    """

    if(row["touchdown"]):
        # who scored? (checked against nflfastR's td_team: exact on all 10,272 TDs in 2015-2024)
        if row["action"] == "field_goal":                 # blocked/missed kick returned for a TD
            return -7.0
        if row["action"] == "punt":                       # return TD, unless the receiving team
            return 7.0 if row["turnover"] == 1 else -7.0  # muffed it and the punting team scored
        return -7.0 if row["turnover"] == 1 else 7.0      # interception/fumble return TD
    if(row["field_goal_result"] == "made"):
        return 3.0
    if(row["safety"]):
        return -2.0
    return 0.0


def get_next_state(row, next_row):
    """
    Determine the next state after a play in the data given the two rows. Analogous to outcomes in valueiteration.py.

    Returns tuple:
        (sign, state)
    """

    #Handle changes in possession 
    # if ACTIONS.index(row["action"]) == 0:
    #     if row["touchdown"] or row["safety"] or row["turnover"] or (row["down"] == 4 and row["row["ydstogo"]:
    #         return (-1, get_state(next_row), get_reward(next_row))
    #     return (1, get_state(next_row), get_reward(next_row))
    # elif ACTIONS.index(row["action"]) == 1 or ACTIONS.index(row["action"]) == 2:
    #     return (-1, get_state(next_row), get_reward(next_row))

    if(row["posteam"] != next_row["posteam"]):
        return(-1, get_state(next_row))
    
    return(1, get_state(next_row))


def extract_transitions(df):
    """
    Extract states, actions, rewards, and transitions from play-by-play data.

    Returns list[tuple]:
        Each transition is (sign, state, action, reward, next state, For the next state: down, ydstogo, yardline100)
    
    sign = +1 if possession is retained after the play
    sign = -1 if possession changes after the play
    """

    transitions = []

    df = df.reset_index(drop=True)

    for game_id, game in df.groupby("game_id", sort=False):
        plays = game.to_dict("records")

        for i, row in enumerate(plays):
            end_state = (i == len(plays)-1) #this indicates whether the current state is terminal
            state = get_state(row)
            action = get_action(row)
            reward = get_reward(row)
            

            if not end_state:
                if state is None:
                    continue

                next_row = plays[i+1]      
                down = next_row["down"]
                ydstogo = next_row["ydstogo"]
                ydline = next_row["yardline_100"]              
                #state = get_state(row)
                next_state  = get_next_state(row, next_row)

                if next_state[1] is None:
                    continue

                # skip halftime: the next row starts a new half, not a result of this play
                if row["qtr"] == 2 and next_row["qtr"] == 3:
                    continue

                # skip gaps left by filtering (e.g. a dropped penalty play): same possession, but the
                # next play does not start where this run/pass ended
                if next_state[0] == 1 and row["action"] == "go" and not row["touchdown"]:
                    if abs((row["yardline_100"] - row["yards_gained"]) - next_row["yardline_100"]) > 0.5:
                        continue

                #action = get_action(row)
                

                
                # if action not in legal_actions(
                #     int(row["down"]),
                #     min(int(row["ydstogo"]), MAX_TOGO),
                #     int(row["yardline_100"]))
                #     continue

                
                
                transitions.append((next_state[0], state, action, reward, next_state[1], down, ydstogo, ydline))
            else:
                if state is None:
                    continue
                transitions.append((1, state, action, reward, None, None, None, None))

    return transitions


# ------------------------------------------------------------------ solvers
def q_learning(transitions, alpha=0.1, gamma=1.0, epochs=100, tol=1e-6):
    """gamma discounts the opponent's future value only (same meaning as --gamma-poss in value_iteration.py).

    Each game is treated as an episode of observations. We update the Q-value for each step in
    the game (a play) until the game ends. We repeat for all games, for multiple epochs.

    Returns ndarray(3,N_STATES), list[float]:
        Q, deltas
    """
    Q = np.zeros((3, N_STATES))

    # Offline data only contains the actions coaches took. A (state, action) pair that never
    # appears keeps its initial Q = 0, which would look better than any negative value
    # (e.g. punting from your own 20). So we only ever compare actions that were observed.
    seen = np.zeros((3, N_STATES), dtype=bool)
    for play in transitions:
        seen[play[2], play[1]] = True

    deltas = []

    # for epoch in range(epochs):

    #     #np.random.shuffle(transitions)
    for epoch in range(epochs):
        max_delta = 0.0
        
        for i, play in enumerate(transitions):
            sign = play[0]
            state = play[1]
            action = play[2] #Since we do not know transition probabilities, we fix the action to the historical action
            new_state = play[4]
            reward = play[3]

            if(new_state is not None):
                next_actions = [a for a in legal_actions(
                    int(play[5]),
                    min(int(play[6]), MAX_TOGO),
                    int(play[7])
                ) if seen[a, new_state]]                      # legal AND observed
                next_value = max(Q[a, new_state] for a in next_actions) if next_actions else 0.0
                disc = gamma if sign < 0 else 1.0           # discount only the opponent's future, as in value_iteration
                delta = reward + sign*disc*next_value - Q[action, state]
            else:
                delta = reward - Q[action, state]

            Q[action, state] = Q[action, state] + alpha*delta #Q learning update

            max_delta = max(max_delta, abs(alpha*delta))

        deltas.append(max_delta)

    Q[~seen] = np.nan        # never observed: no estimate
    return Q, deltas


def get_values(Q):
    """
    Get the state value function from the Q values passed as input.

    Returns ndarray(N_STATES)
        V
    """

    V = np.zeros(N_STATES)

    for down in range(1,5):
        for togo in range(1, MAX_TOGO+1):
            for yl in range(1,100):
                state = idx(down,togo,yl)
                actions = legal_actions(down,togo,yl)

                actions = [a for a in actions if np.isfinite(Q[a, state])]
                V[state] = max(Q[a, state] for a in actions) if actions else np.nan

    return V

def get_policy(Q):
    """
    Get the policy from the Q values passed as input.

    Returns ndarray(N_STATES)
        pi
    """
    pi = np.full(N_STATES, -1)

    for down in range(1,5):
        for togo in range(1, MAX_TOGO+1):
            for yl in range(1,100):
                state = idx(down,togo,yl)
                actions = legal_actions(down,togo,yl)

                actions = [a for a in actions if np.isfinite(Q[a, state])]
                if actions:
                    pi[state] = max(actions, key=lambda a: Q[a, state])   # -1 = no data
    return pi

# ------------------------------------------------------------------ outputs

def save_tables(Q, deltas):
    V = get_values(Q)
    pi = get_policy(Q)
    rows = []
    for d in range (1,5):
        for t in range(1, MAX_TOGO+1):
            for y in range(1,100):
                s = idx(d,t,y)
                q=Q[:,s]
                rows.append(dict(down=d, ydstogo=t, yard_line100=y, V=V[s], Q_go=q[PLAY], Q_punt=q[PUNT], Q_fg = q[FG], best = ACTIONS[pi[s]] if pi[s] != -1 else None))
    
    pd.DataFrame(rows).round(4).to_csv("results/q_learned_values.csv", index=False)
    pd.DataFrame({"sweep": range(1, len(deltas) + 1), "max_change": deltas}).to_csv("results/ql_convergence.csv", index=False)

def plot_value_vs_ep(Q):
    V = get_values(Q)
    plays = pd.read_csv("data/plays_clean.csv", usecols=["down", "ydstogo", "yardline_100", "ep"])
    ep = plays[(plays["down"] == 1) & (plays["ydstogo"] == 10)].groupby("yardline_100")["ep"].mean()
    ys = np.arange(10, 100)
    ours = np.array([V[first_down(y)] for y in ys])
    gap = np.abs(ours - ep.reindex(ys).to_numpy())
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(ys, ours, color="#2a78d6", lw=2, label="Our Offline Q-learning")
    ax.plot(ep.index, ep.values, color="#eb6834", lw=2, ls="--", label="nflfastR expected points")
    ax.set_xlim(99, 10); ax.set_xlabel("Yards from opponent's end zone"); ax.set_ylabel("Expected points")
    ax.set_title("Value of 1st & 10 by field position", loc="left")
    ax.grid(axis="y", alpha=0.3); ax.legend(frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig("figures/value_vs_nflfastr_ql.png", dpi=150); plt.close(fig)
    return float(np.nanmean(gap))


def plot_policy(Q):
    pi = get_policy(Q)
    grid = np.full((10, 99), np.nan)
    for t in range(1, 11):
        for y in range(t, 100):
            # grid[t - 1, y - 1] = np.argmax(Q[:, idx(4, t, y)])
            # state = idx(4, t, y)
            # actions = legal_actions(4, t, y)

            # if actions:
            #     grid[t - 1, y - 1] = max(
            #         actions,
            #         key=lambda a: Q[a, state]
            #     )

            p = pi[idx(4,t,y)]
            grid[t-1,y-1] = p if p >= 0 else np.nan    # blank where there is no data

    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.imshow(grid, aspect="auto", origin="lower", extent=[0.5, 99.5, 0.5, 10.5],
              cmap=ListedColormap(["#2a78d6", "#eb6834", "#1baf7a"]), vmin=-0.5, vmax=2.5,
              interpolation="nearest")
    ax.set_xlim(99.5, 0.5); ax.set_xlabel("Yards from opponent's end zone"); ax.set_ylabel("Yards to go")
    ax.set_title("Optimal 4th-down decision (Q-learning)", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#2a78d6", "#eb6834", "#1baf7a"]]
    ax.legend(handles, ["Go for it", "Punt", "Field goal"], ncol=3, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.25))
    for s in ax.spines.values(): s.set_visible(False)
    fig.tight_layout(); fig.savefig("figures/policy_heatmap_ql.png", dpi=150); plt.close(fig)
    return float(np.mean(grid[~np.isnan(grid)] == PLAY))   # share of 4th-down states with data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma-poss", type=float, default=1.0,
                    help="discount on the opponent's future value (1.0 = none)")
    ap.add_argument("--alpha", type=float, default=0.1, 
                    help="Q-learning step size (0.1 = none)")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Number of epochs of training (100 = none)")
    args = ap.parse_args()
    os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)

    plays_df = load()
    transitions = extract_transitions(plays_df)
    
    Q, deltas = q_learning(transitions, args.alpha, args.gamma_poss, args.epochs)
    V = get_values(Q)

    save_tables(Q, deltas)

    gap = plot_value_vs_ep(Q)
    go_share = plot_policy(Q)

    print(f"Q-learning:             completed training in {len(deltas)} epochs (max change {deltas[-1]:.1e})")
    print(f"Mean gap vs nflfastR EP:     {gap:.2f} points (1st & 10, yardlines 10-99)")
    spots = {99: "own 1", 75: "own 25", 50: "midfield", 25: "opp 25", 10: "opp 10"}
    print("V(1st & 10):                 " + ", ".join(
        f"{name} {V[first_down(y)]:+.2f}" for y, name in spots.items()))
    print(f"4th-down states where go is optimal (togo <= 10): {go_share:.1%}")
    print("Saved results/q_learned_values.csv, results/ql_convergence.csv, figures/*.png")


if __name__ == "__main__":
    main()