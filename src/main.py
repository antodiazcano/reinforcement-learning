"""
Script to play the game with the trained agent.
"""

import pickle
import torch
import neat  # type:ignore

from src.snake import SnakeAI
from src.agent_ppo.model import AgentPPO
from src.dqn.model import AgentDQN


def play_ppo() -> None:
    """
    Plays a game with the trained PPO agent.
    """

    agent = AgentPPO()
    weights = "weights/agent_ppo.pt"
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
        action = net.activate(game_input)
        game_over, _ = game.play_step(action)


def main() -> None:
    """
    Plays a game with the trained agent.
    """

    agent = "dqn"

    if agent == "ppo":
        play_ppo()
    elif agent == "dqn":
        play_dqn()
    else:
        play_neat()


if __name__ == "__main__":
    main()
