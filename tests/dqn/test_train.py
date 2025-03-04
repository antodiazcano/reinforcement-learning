"""
Test for the train of the DQN.
"""

import os
import torch

from src.agent_dqn.train import QTrainer
from src.agent_dqn.model import AgentDQN


TRAINER = QTrainer(AgentDQN())


def test_epsilon_policy() -> None:
    """
    Test for the _epsilon_policy function.
    """

    outputs = []
    n = 100

    for _ in range(n):
        random_episode = int(torch.randint(0, 1_000, (1,))[0])
        output = TRAINER._epsilon_policy(random_episode)
        if output not in outputs:
            outputs.append(output)
        if False in outputs and True in outputs:
            break

    assert False in outputs and True in outputs


def test_simulate_episode() -> None:
    """
    Test for the _simulate_one_episode function.
    """

    states, q_values, rewards = TRAINER._simulate_one_episode(
        int(torch.randint(0, 1_000, (1,))[0])
    )

    assert isinstance(states, list) and isinstance(states[0], torch.Tensor)
    assert isinstance(q_values, list) and isinstance(q_values[0], torch.Tensor)
    assert isinstance(rewards, list) and isinstance(rewards[0], int)


def test_fit() -> None:
    """
    Test for the fit function.
    """

    image_path = "images/tests/train_dqn_test.png"
    weights_path = "weights/tests/agent_dqn_test.pt"
    epochs = 2
    TRAINER.fit(
        epochs,
        image_path=image_path,
        weights_path=weights_path,
    )
    assert os.path.exists(image_path) and os.path.exists(weights_path)
