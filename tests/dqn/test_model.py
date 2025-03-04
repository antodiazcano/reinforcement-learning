"""
Script to test the DQN agent.
"""

import torch

from src.agent_dqn.model import AgentDQN
from src.snake import SnakeAI


MODEL = AgentDQN()
N = 100


def test_forward() -> None:
    """
    Test for the forward pass.
    """

    for _ in range(N):
        output = MODEL(torch.randn(11))
        assert output.shape == torch.Size([1, 3])
        assert torch.all(output > 0) and torch.all(output < 1)
        assert torch.isclose(output.sum(), torch.tensor(1.0))


def test_act() -> None:
    """
    Test for the act function.
    """

    for _ in range(N):
        action, log_prob = MODEL.act(torch.randn(11))
        assert isinstance(action, int)
        assert 0 <= action <= 2
        assert log_prob.shape == ()


def test_predict() -> None:
    """
    Test for the predict function.
    """

    game = SnakeAI(visualize=False)
    for _ in range(N):
        action, probs = MODEL.predict(game)
        assert isinstance(action, int)
        assert 0 <= action <= 2
        assert probs.shape == torch.Size([3])
        assert torch.all(probs > 0) and torch.all(probs < 1)
        assert torch.isclose(probs.sum(), torch.tensor(1.0))
