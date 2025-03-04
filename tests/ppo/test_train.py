"""
Test for the train of the DQN.
"""

import os
from collections import deque
import torch

from src.agent_ppo.train import PPOTrainer
from src.agent_ppo.model import AgentPPO


TRAINER = PPOTrainer(AgentPPO())


def test_simulate_episode() -> None:
    """
    Test for the _simulate_one_episode function.
    """

    rewards, saved_log_probs = TRAINER._simulate_one_episode()

    assert isinstance(rewards, list) and isinstance(rewards[0], int)
    assert isinstance(saved_log_probs, list) and isinstance(
        saved_log_probs[0], torch.Tensor
    )


def test_standardize() -> None:
    """
    Test for the standardize function.
    """

    data = deque([1.0, 2.0, 3.0, 4.0, 5.0])
    result = TRAINER.standardize(data)

    assert (
        isinstance(result, torch.Tensor)
        and torch.isclose(result.mean(), torch.tensor(0.0))
        and torch.isclose(result.std(), torch.tensor(1.0))
    )


def test_fit() -> None:
    """
    Test for the fit function.
    """

    image_path = "images/tests/train_ppo_test.png"
    weights_path = "weights/tests/agent_ppo_test.pt"
    epochs = 2
    TRAINER.fit(
        epochs,
        image_path=image_path,
        weights_path=weights_path,
    )
    assert os.path.exists(image_path) and os.path.exists(weights_path)
