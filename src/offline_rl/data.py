import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class RLDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self._verify_df(df)
        self.df = df
        self._state_idx: int = df.columns.get_loc("state")
        self._action_idx: int = df.columns.get_loc("action")
        self._reward_idx: int = df.columns.get_loc("reward")
        self._terminal_idx: int = df.columns.get_loc("terminal")
        self._next_state_idx: int = df.columns.get_loc("next_state")
        self.state_size = len(df["state"][0])
        self.num_actions = df["action"].nunique()

    def _verify_df(self, df: pd.DataFrame) -> None:
        if not {"state", "action", "reward", "terminal", "next_state"}.issubset(df.columns):
            raise ValueError("Missing columns in dataframe")

    def collate(self, batch: list[np.ndarray]) -> dict[str, Tensor]:
        batch = np.array(batch)
        return {
            "state": Tensor(np.vstack(batch[:, self._state_idx])),
            "action": Tensor(np.vstack(batch[:, self._action_idx]).flatten()).to(torch.int64),
            "reward": Tensor(np.vstack(batch[:, self._reward_idx]).flatten()),
            "terminal": Tensor(np.vstack(batch[:, self._terminal_idx]).flatten()).to(torch.bool),
            "next_state": Tensor(np.vstack(batch[:, self._next_state_idx])),
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.df.iloc[idx].values
