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
from tqdm import tqdm


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
        return 7.0
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

                #action = get_action(row)
                

                
                # if action not in legal_actions(
                #     int(row["down"]),
                #     min(int(row["ydstogo"]), MAX_TOGO),
                #     int(row["yardline_100"]))
                #     continue

                
                
                transitions.append((next_state[0], state, action, reward, next_state[1], down, ydstogo, ydline))
            else:
                transitions.append((1, state, action, reward, None, None, None, None))

    return transitions


# ------------------------------------------------------------------ solvers
def q_learning(df, transitions, alpha=0.1, gamma=1.0, epochs=100, tol=1e-6):
    """
    Each game is treated as an episode of observations. We update the Q-value for each step in
    the game (a play) until the game ends. We repeat for all games, for multiple epochs.

    Returns ndarray(3,N_STATES):
        Q
    """
    Q = np.zeros((3, N_STATES))

    deltas = []

    # for epoch in range(epochs):

    #     #np.random.shuffle(transitions)
    for epoch in range(epochs):
        for i, play in enumerate(transitions):
            sign = play[0]
            state = play[1]
            action = play[2] #Since we do not know transition probabilities, we fix the action to the historical action
            new_state = play[4]
            reward = play[3]

            if(new_state is not None):
                next_actions = legal_actions(
                    int(play[5]),
                    min(int(play[6]), MAX_TOGO),
                    int(play[7])
                )
                delta = reward + sign*gamma*max(Q[a, new_state] for a in next_actions) - Q[action, state] #Only maximize Q for the next state for legal actions
            else:
                delta = reward - Q[action, state]

            Q[action, state] = Q[action, state] + alpha*delta #Q learning update

    return Q


def get_values(Q):
    """
    Get the state value function from the Q values passed as input.

    Returns ndarray(N_STATES)
        V
    """

    V = max(Q, axis=0)

    V = np.zeros(N_STATES)

    for down in range(1,5):
        for togo in range(1, MAX_TOGO+1):
            for yl in range(1,100):
                state = idx(down,togo,yl)
                actions = legal_actions(down,togo,yl)

                if actions:
                    V[state] = max(Q[a, state] for a in actions)

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

                if actions:
                    pi[state] = max(actions, key=lambda a: Q[a, state])
    return pi

# ------------------------------------------------------------------ outputs

def save_tables(Q, delta):
    pass

def plot_value_vs_ep():
    pass

def plot_policy():
    pass

def main():
    pass

if __name__ == "__main__":
    main()