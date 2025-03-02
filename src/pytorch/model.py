"""
This script defines the model used for the agent.
"""

import torch
from torch import nn
from torch.distributions import Categorical

from src.snake_ai import SnakeAI
from src.constants import Direction, BLOCK_SIZE


class Agent(nn.Module):
    """
    Class to define the agent.
    """

    def __init__(self, hidden_1: int = 128, hidden_2: int = 64) -> None:
        """
        Constructor of the class.
        """

        super().__init__()

        self.fc1 = nn.Linear(11, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 3)
        self.relu = nn.ReLU()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _game_to_tensor(self, game: SnakeAI) -> torch.Tensor:
        """
        Obtains a tensor that represents the current game.

        Parameters
        ----------
        game : Instance representing the current game.

        Returns
        -------
        Tensor representing the current game.
        """

        # Head and points around head
        head_x, head_y = game.snake[0]
        point_l = (head_x - BLOCK_SIZE, head_y)
        point_r = (head_x + BLOCK_SIZE, head_y)
        point_u = (head_x, head_y - BLOCK_SIZE)
        point_d = (head_x, head_y + BLOCK_SIZE)

        # Current direction as one hot
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Possible collisions
        collision_straight = (
            (dir_r and game.is_collision_point(point_r))
            or (dir_l and game.is_collision_point(point_l))
            or (dir_u and game.is_collision_point(point_u))
            or (dir_d and game.is_collision_point(point_d)),
        )[0]
        collision_right = (
            (dir_u and game.is_collision_point(point_r))
            or (dir_d and game.is_collision_point(point_l))
            or (dir_l and game.is_collision_point(point_u))
            or (dir_r and game.is_collision_point(point_d)),
        )[0]
        collision_left = (
            (dir_d and game.is_collision_point(point_r))
            or (dir_u and game.is_collision_point(point_l))
            or (dir_r and game.is_collision_point(point_u))
            or (dir_l and game.is_collision_point(point_d)),
        )[0]

        return torch.tensor(
            [
                collision_straight,
                collision_right,
                collision_left,
                dir_l,
                dir_r,
                dir_u,
                dir_d,
                # Food location
                game.food[0] < game.snake[0][0],  # food left
                game.food[0] > game.snake[0][0],  # food right
                game.food[1] < game.snake[0][1],  # food up
                game.food[1] > game.snake[0][1],  # food down
            ],
            dtype=torch.float32,
        )

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

        game_tensor = self._game_to_tensor(game).to(self.device)
        x = self.relu(self.fc1(game_tensor))
        x = self.relu(self.fc2(x))
        return nn.functional.softmax(self.fc3(x), dim=0)

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
        Action chosen and probability of the action.
        """

        probs = self.forward(game)
        dist = Categorical(probs=probs)
        action = dist.sample()

        return action.item(), probs[action]

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
        Action chosen and probabilities of the action.
        """

        with torch.no_grad():
            probs = self.forward(game)

        action = int(torch.argmax(probs))

        return action, nn.functional.softmax(probs, dim=0)
