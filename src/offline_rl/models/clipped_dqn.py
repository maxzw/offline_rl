from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from offline_rl.encoder import VectorEncoder
from offline_rl.models.base import BaseModel


class ClippedDQN(BaseModel):
    """Offline Clipped Deep Q-Network.

    See paper: https://arxiv.org/pdf/1802.09477.pdf
    """

    model_1: VectorEncoder
    model_2: VectorEncoder

    def __init__(self, hidden_layers: Sequence[int] = (8, 8)) -> None:
        self.hidden_layers = hidden_layers

    def _initialize(self, state_size: int, num_actions: int) -> None:
        kwargs = {
            "n_input": state_size,
            "n_output": num_actions,
            "hidden_layers": self.hidden_layers
        }
        self.model_1 = VectorEncoder(**kwargs).to(self._device)
        self.model_2 = VectorEncoder(**kwargs).to(self._device)

    def fit(self, dataloader: DataLoader, epochs=20, gamma=0.9, lr=1e-3,) -> None:
        self._initialize(
            state_size=dataloader.dataset.state_size,
            num_actions=dataloader.dataset.num_actions
        )
        optimizer_1 = torch.optim.Adam(self.model_1.parameters(), lr=lr)
        optimizer_2 = torch.optim.Adam(self.model_2.parameters(), lr=lr)

        for _ in tqdm(range(epochs)):
            for batch in dataloader:
                self._update(batch, optimizer_1, optimizer_2, gamma)

    def _update(
        self,
        batch: dict[str, Tensor],
        optimizer_1: torch.optim.Optimizer,
        optimizer_2: torch.optim.Optimizer,
        gamma: float,
    ) -> None:
        # Move batch to device
        for key in batch:
            batch[key] = batch[key].to(self._device)

        # 1) Calculate Q-values for the current state and current action
        q_hat_1 = self.model_1(batch["state"]).gather(1, batch["action"].unsqueeze(1))
        q_hat_2 = self.model_2(batch["state"]).gather(1, batch["action"].unsqueeze(1))

        # 2) Calculate max Q-values for the next state
        next_q_1 = torch.zeros(len(batch["next_state"]), dtype=torch.float)
        next_q_1[~batch["terminal"]] = self.model_1(batch["next_state"][~batch["terminal"]]).max(1)[0]
        next_q_2 = torch.zeros(len(batch["next_state"]), dtype=torch.float)
        next_q_2[~batch["terminal"]] = self.model_2(batch["next_state"][~batch["terminal"]]).max(1)[0]
        next_q = torch.min(next_q_1, next_q_2)

        # 3) Calculate target Q-values
        target_q = (batch["reward"] + gamma * next_q).float().unsqueeze(1).detach()

        # 4) Calculate loss
        loss_1 = F.mse_loss(q_hat_1, target_q)
        loss_2 = F.mse_loss(q_hat_2, target_q)

        # 5) Update policy model
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        loss_1.backward()
        loss_2.backward()
        optimizer_1.step()
        optimizer_2.step()

    def get_q_values(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            q_values_1 = self.model_1(x.to(self._device))
            q_values_2 = self.model_2(x.to(self._device))
            q_values = (q_values_1 + q_values_2) / 2
        return q_values
