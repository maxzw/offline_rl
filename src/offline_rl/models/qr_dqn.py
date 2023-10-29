from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from offline_rl.encoder import VectorEncoder
from offline_rl.models.double_dqn import DoubleDQN


class QuantileHuberLoss(nn.Module):
    def __init__(self, quantiles, delta=1.0):
        super().__init__()
        self.quantiles = quantiles
        self.delta = delta
        self.hl = nn.HuberLoss(reduction="none", delta=delta)

    def forward(self, preds: Tensor, target: Tensor) -> Tensor:
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            h_loss = self.hl(preds[:, i], target)
            weight = torch.abs(q - (errors.detach() < 0).float()).unsqueeze(1)
            losses.append(weight * h_loss)
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

class QRDQN(DoubleDQN):
    """Offline Quantile Regression DQN.

    See paper: https://arxiv.org/pdf/1710.10044.pdf
    """
    policy_model: VectorEncoder
    target_model: VectorEncoder

    def __init__(
        self,
        hidden_layers: Sequence[int] = (8, 8),
        quantiles: Sequence[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
    ) -> None:
        self.hidden_layers = hidden_layers
        self.num_quantiles = len(quantiles)
        self.huber_loss = QuantileHuberLoss(quantiles)

    def _initialize(self, state_size: int, num_actions: int) -> None:
        kwargs = {
            "n_input": state_size,
            "n_output": num_actions * self.num_quantiles,
            "hidden_layers": self.hidden_layers
        }
        self.policy_model = VectorEncoder(**kwargs).to(self._device)
        self.target_model = VectorEncoder(**kwargs).to(self._device)
        self.target_model.load_state_dict(self.policy_model.state_dict())

    def fit(
        self,
        dataloader: DataLoader,
        epochs=20,
        gamma=0.9,
        lr=1e-3,
        update_target_every: int = 10,
        update_percentage: float = 0.01,
    ) -> None:
        self._initialize(
            state_size=dataloader.dataset.state_size,
            num_actions=dataloader.dataset.num_actions
        )
        optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=lr)

        iteration = 0
        for _ in tqdm(range(epochs)):
            for batch in dataloader:
                self._update_policy_model(batch, optimizer, gamma)

                if iteration % update_target_every == 0:
                    self._update_target_model(update_percentage)

                iteration += 1

    def _update_policy_model(self, batch: dict[str, Tensor], optimizer: torch.optim.Optimizer, gamma: float) -> None:
        # Move batch to device
        for key in batch:
            batch[key] = batch[key].to(self._device)

        # 1) Calculate Q-values for the current state and current action
        q_values = self.policy_model(batch["state"]).reshape(len(batch["state"]), -1, self.num_quantiles)
        q_hat = (
            q_values.gather(1, batch["action"].unsqueeze(1).unsqueeze(1).expand(-1, -1, self.num_quantiles)).squeeze(1)
        )

        # 2) Calculate max Q-values for the next state
        next_q = torch.zeros(len(batch["next_state"]), dtype=torch.float)
        next_q[~batch["terminal"]] = (
            self.target_model(batch["next_state"][~batch["terminal"]])
            .reshape(len(batch["next_state"][~batch["terminal"]]), -1, self.num_quantiles)
            .mean(2).max(1)[0]
        )

        # 3) Calculate target Q-values
        target_q = (batch["reward"] + gamma * next_q).float().detach()

        # 4) Calculate loss
        loss: Tensor = self.huber_loss(q_hat, target_q)

        # 5) Update policy model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def get_q_values(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            q_values = self.target_model(x.to(self._device)).reshape(len(x), -1, self.num_quantiles).mean(2)
        return q_values
