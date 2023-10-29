from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from offline_rl.encoder import VectorEncoder
from offline_rl.models.base import BaseModel


class DoubleDQN(BaseModel):
    """Offline Double Deep Q-Network.

    See paper: https://arxiv.org/pdf/1509.06461.pdf
    """

    policy_model: VectorEncoder
    target_model: VectorEncoder

    def __init__(self, hidden_layers: Sequence[int] = (8, 8)) -> None:
        self.hidden_layers = hidden_layers

    def _initialize(self, state_size: int, num_actions: int) -> None:
        kwargs = {
            "n_input": state_size,
            "n_output": num_actions,
            "hidden_layers": self.hidden_layers,
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
            num_actions=dataloader.dataset.num_actions,
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
        q_hat = self.policy_model(batch["state"]).gather(1, batch["action"].unsqueeze(1))

        # 2) Calculate max Q-values for the next state
        next_q = torch.zeros(len(batch["next_state"]), dtype=torch.float)
        next_q[~batch["terminal"]] = self.target_model(batch["next_state"][~batch["terminal"]]).max(1)[0]

        # 3) Calculate target Q-values
        target_q = (batch["reward"] + gamma * next_q).float().unsqueeze(1).detach()

        # 4) Calculate loss
        loss = F.mse_loss(q_hat, target_q)

        # 5) Update policy model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _update_target_model(self, update_percentage: float) -> None:
        for policy_param, target_param in zip(
            self.policy_model.parameters(), self.target_model.parameters(), strict=True
        ):
            target_param.data.copy_(update_percentage * policy_param.data+ (1 - update_percentage) * target_param.data)

    def get_q_values(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            q_values = self.target_model(x.to(self._device))
        return q_values
