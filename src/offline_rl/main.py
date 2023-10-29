from ast import literal_eval
from enum import Enum
from os.path import exists

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from offline_rl.data import RLDataset
from offline_rl.models import get_model_type

app = typer.Typer(add_completion=False)

HISTORY_PATH = "src/offline_rl/data/data.csv"
ENVIRONMENT = "CartPole-v1"


class ModelType(str, Enum):
    double_dqn = "double_dqn"
    clipped_dqn = "clipped_dqn"
    multi_dqn = "multi_dqn"
    qr_dqn = "qr_dqn"
    rem = "rem"


def create_history() -> pd.DataFrame:
    env = gym.make(ENVIRONMENT)
    num_episodes = 5000
    episodes, states, actions, next_states, terminals = [], [], [], [], []
    for episode in tqdm(range(num_episodes)):
        state, *_ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, _, done, *_ = env.step(action)
            episodes.append(episode)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            terminals.append(int(done))
            state = next_state
    env.close()
    df = pd.DataFrame(
        {
            "episode": episodes,
            "state": np.vstack(states).tolist(),
            "action": actions,
            "next_state": np.vstack(next_states).tolist(),
            "terminal": terminals,
        }
    )
    df.to_csv(HISTORY_PATH, index=False)
    return df


@app.command()
def main(
    model_type: str = typer.Argument(..., help="Type of model to train."),
) -> None:
    if not exists(HISTORY_PATH):
        print("History does not exist. Creating history.")
        history = create_history()
    else:
        print("Loading history.")
        history = pd.read_csv(HISTORY_PATH).assign(
            state=lambda df: df["state"].apply(literal_eval),
            next_state=lambda df: df["next_state"].apply(literal_eval),
        )
    history = history.assign(reward=1)  # Reward is always 1 for CartPole
    dataset = RLDataset(history)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=dataset.collate)

    print("Training model.")
    model = get_model_type(model_type)()
    model.fit(dataloader)

    print("Evaluating model.")
    env = gym.make(ENVIRONMENT)
    num_episodes = 1000
    max_score = 500

    new_scores = []
    for _ in range(num_episodes):
        state, *_ = env.reset()
        done = False
        score = 0
        while not done and score < max_score:
            action = model.get_action(torch.tensor(state, dtype=torch.float).unsqueeze(0))[0].item()
            state, _, done, *_ = env.step(action)
            score += 1
        new_scores.append(score)
    env.close()
    historical_rewards = history.groupby("episode")["reward"].sum()
    print(f"History average score: {np.mean(historical_rewards)}")
    print(f"New average score: {np.mean(new_scores)}")

    print("Saving results.")
    bins = np.linspace(0, 500, 50)
    plt.hist(historical_rewards, bins, label="random")
    plt.hist(new_scores, bins, label=f"{model_type}")
    plt.legend(loc="upper right")
    plt.savefig(f"src/offline_rl/results/{model_type}.png")
