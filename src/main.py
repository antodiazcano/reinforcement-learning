"""
Script to play the game with the trained agent.
"""

import torch

from src.snake import SnakeAI
from src.pytorch.model import Agent


def main() -> None:
    """
    Plays a game with the trained agent.
    """

    agent = Agent()
    weights = "weights/agent_pytorch.pt"
    agent.load_state_dict(torch.load(weights))
    game = SnakeAI()

    game_over = False
    while not game_over:
        action, _ = agent.act(game)
        game_over, _ = game.play_step(action)

    print(f"Score: {game.score}")


if __name__ == "__main__":
    main()
