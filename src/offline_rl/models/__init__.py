from offline_rl.models.base import BaseModel
from offline_rl.models.clipped_dqn import ClippedDQN
from offline_rl.models.double_dqn import DoubleDQN
from offline_rl.models.multi_dqn import MultiDQN
from offline_rl.models.qr_dqn import QRDQN

__all__ = ["get_model_type", "DoubleDQN", "ClippedDQN", "MultiDQN", "QRDQN"]

model_type_mapping = {
    "double_dqn": DoubleDQN,
    "clipped_dqn": ClippedDQN,
    "multi_dqn": MultiDQN,
    "qr_dqn": QRDQN,
}


def get_model_type(model_type: str) -> type[BaseModel]:
    model_class = model_type_mapping.get(model_type)
    if model_class is None:
        raise ValueError(f"Model type {model_type} not recognized.")
    return model_class
