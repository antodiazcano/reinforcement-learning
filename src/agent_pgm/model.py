"""
This script defines the model used for the PGM agent.
"""

import torch
from torch import nn
from torch.distributions import Categorical

from src.snake import SnakeAI


class AgentPGM(nn.Module):
    """
    Class to define PGM the agent.
    """

    def __init__(self, hidden_1: int = 128, hidden_2: int = 64) -> None:
        """
        Constructor of the class.

        Parameters
        ----------
        hidden_1 : First hidden dimension.
        hidden_2 : Second hidden dimension.
        """

        super().__init__()

        self.fc1 = nn.Linear(11, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 3)
        self.relu = nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, game: SnakeAI) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        game : Instance representing the current game.

        Returns
        -------
        Output of the network.
        """

        game_tensor = game.game_to_array().unsqueeze(0).to(self.device)
        x = self.relu(self.fc1(game_tensor))
        x = self.relu(self.fc2(x))
        return nn.functional.softmax(self.fc3(x), dim=1)

    def act(self, game: SnakeAI) -> tuple[int, torch.Tensor]:
        """
        The agent actuates given the state. It is very important to return a tensor
        instead of a float in the log prob because if not the backward pass can not be
        done.

        Parameters
        ----------
        game : Current game.

        Returns
        -------
        Action chosen and log probability of the action.
        """

        probs = self.forward(game)[0]
        dist = Categorical(probs=probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action)

    def predict(self, game: SnakeAI) -> tuple[int, torch.Tensor]:
        """
        The agent actuates given the state. Note that the difference with the 'act'
        function is that here the maximum probability is always the action chosen and
        no gradients are computed because it is not used for training.

        Parameters
        ----------
        game : Current game.

        Returns
        -------
        Action chosen and probabilities of the actions.
        """

        self.eval()

        with torch.no_grad():
            probs = self.forward(game)[0]

        action = int(torch.argmax(probs))

        return action, nn.functional.softmax(probs, dim=0)
