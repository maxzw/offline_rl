from abc import ABC
from abc import abstractmethod

import torch
from torch import Tensor

from offline_rl.data import RLDataset


class BaseModel(ABC):
    """Base class for offline RL models."""

    @property
    def _device(self) -> str:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def fit(self, dataset: RLDataset) -> None:
        pass

    def get_action(self, x: Tensor) -> Tensor:
        q_values = self.get_q_values(x.to(self._device))
        actions = q_values.argmax(1)
        return actions

    @abstractmethod
    def get_q_values(self, x: Tensor) -> Tensor:
        pass
