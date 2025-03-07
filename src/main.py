"""
Script to play the game with the trained agent.
"""

from typing import Literal
import pickle
import torch
import neat  # type:ignore
import numpy as np

from src.snake import SnakeAI
from src.agent_pgm.model import AgentPGM
from src.agent_dqn.model import AgentDQN


def play_pgm() -> None:
    """
    Plays a game with the trained PGM agent.
    """

    agent = AgentPGM()
    weights = "weights/agent_pgm.pt"
    agent.load_state_dict(torch.load(weights))
    game = SnakeAI()

    game_over = False
    while not game_over:
        action, _ = agent.predict(game)
        game_over, _ = game.play_step(action)

    print(f"Score: {game.score}")


def play_dqn() -> None:
    """
    Plays a game with the trained DQN agent.
    """

    agent = AgentDQN()
    weights = "weights/agent_dqn.pt"
    agent.load_state_dict(torch.load(weights))
    game = SnakeAI()

    game_over = False
    while not game_over:
        action, _ = agent.predict(game)
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
        action = int(np.argmax(net.activate(game_input)))
        game_over, _ = game.play_step(action)


def main(agent: Literal["dqn", "pgm", "neat"]) -> None:
    """
    Plays a game with the trained agent.

    Parameters
    ----------
    agent : The trained agent we want to use.
    """

    if agent == "pgm":
        play_pgm()
    elif agent == "dqn":
        play_dqn()
    elif agent == "neat":
        play_neat()
    else:
        print("Choose a correct agent, i.e., 'pgm', 'dqn' or 'neat'.")


if __name__ == "__main__":
    AGENT: Literal["dqn", "pgm", "neat"] = "dqn"
    main(AGENT)
