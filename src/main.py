"""
Script to play the game with the trained agent.
"""

import pickle
import torch
import neat  # type:ignore

from src.snake import SnakeAI
from src.agent_torch.model import Agent


def play_torch() -> None:
    """
    Plays a game with the trained PyTorch agent.
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


def play_neat() -> None:
    """
    Plays a game with the trained Neat agent.
    """

    with open("weights/agent_neat.pkl", "rb") as file:
        agent = pickle.load(file)
    config_path = "src/agent_neat/config.txt"
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    net = neat.nn.FeedForwardNetwork.create(agent, config)
    game = SnakeAI()

    game_over = False
    while not game_over:
        game_input = game.game_to_array().numpy()
        action = net.activate(game_input)
        game_over, _ = game.play_step(action)


def main() -> None:
    """
    Plays a game with the trained agent.
    """

    use_torch = True
    if use_torch:
        play_torch()
    else:
        play_neat()


if __name__ == "__main__":
    main()
