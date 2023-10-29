from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from offline_rl.encoder import VectorEncoder
from offline_rl.models.base import BaseModel


class MultiDQN(BaseModel):
    """Offline Multi Deep Q-Network.

    See paper:
        Exploring deep reinforcement learning with multi q-learning.
        Duryea, E., Ganger, M., & Hu, W. (2016).
    """
    models: list[VectorEncoder]

    def __init__(self, num_encoders: int = 4, hidden_layers: Sequence[int] = (8, 8)) -> None:
        self.num_encoders = num_encoders
        self.hidden_layers = hidden_layers

    def _initialize(self, state_size: int, num_actions: int) -> None:
        kwargs = {
            "n_input": state_size,
            "n_output": num_actions,
            "hidden_layers": self.hidden_layers
        }
        self.models = [VectorEncoder(**kwargs) for _ in range(self.num_encoders)]

    def fit(self, dataloader: DataLoader, epochs=20, gamma=0.9, lr=1e-3) -> None:
        self._initialize(
            state_size=dataloader.dataset.state_size,
            num_actions=dataloader.dataset.num_actions
        )
        optimizers = [torch.optim.Adam(model.parameters(), lr=lr) for model in self.models]

        for _ in tqdm(range(epochs)):
            for batch in dataloader:
                self._update(batch, optimizers, gamma)

    def _update(self, batch: dict[str, Tensor], optimizers: list[torch.optim.Optimizer], gamma: float) -> None:
        # Move batch to device
        for key in batch:
            batch[key] = batch[key].to(self._device)

        # 1) Calculate Q-values for the current state and current action using sampled model
        model_idx = np.random.randint(0, self.num_encoders)
        q_hat = self.models[model_idx](batch["state"]).gather(1, batch["action"].unsqueeze(1))

        # 2) Calculate max Q-values for the next state
        next_q = torch.stack([self._return_max_q(model, batch) for model in self.models]).mean(dim=0)

        # 3) Calculate target Q-values
        target_q = (batch["reward"] + gamma * next_q).float().unsqueeze(1).detach()

        # 4) Calculate loss
        loss = F.mse_loss(q_hat, target_q)

        # 5) Update policy model
        optimizer = optimizers[model_idx]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _return_max_q(self, model: VectorEncoder, batch: dict[str, Tensor]) -> Tensor:
        q_values = torch.zeros(len(batch["next_state"]), dtype=torch.float)
        q_values[~batch["terminal"]] = model(batch["next_state"][~batch["terminal"]]).max(1)[0]
        return q_values

    def get_q_values(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            q_values = torch.stack([model(x.to(self._device)) for model in self.models]).mean(dim=0)
        return q_values
