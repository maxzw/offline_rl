from collections.abc import Sequence

import torch
from torch import Tensor


def init_weights(m: torch.nn.Module) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class VectorEncoder(torch.nn.Module):
    """A simple MLP encoder for vector inputs."""

    def __init__(self, n_input: int, n_output: int, hidden_layers: Sequence[int] = ()) -> None:
        super().__init__()
        if len(hidden_layers) == 0:
            self.model = torch.nn.Sequential(torch.nn.Linear(n_input, n_output))
        else:
            layers = [torch.nn.Linear(n_input, hidden_layers[0]), torch.nn.ReLU()]
            for i in range(1, len(hidden_layers)):
                layers.append(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
                layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(hidden_layers[-1], n_output))
            self.model = torch.nn.Sequential(*layers)
        self.model.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
