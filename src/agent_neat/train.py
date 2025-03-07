"""
Script to train the NEAT model.
"""

import pickle
import neat  # type:ignore
import torch
import numpy as np

from src.snake import SnakeAI


def play_game(net: neat.nn.FeedForwardNetwork) -> float:
    """
    The agent plays a game.

    Parameters
    ----------
    net : Net that models the actions of the agent.

    Returns
    -------
    Mean reward obtained.
    """

    game = SnakeAI(visualize=False)
    total_reward = 0.0

    max_steps = 500
    for i in range(max_steps):
        game_input = game.game_to_array().numpy()
        network_output = torch.nn.functional.softmax(
            torch.tensor(net.activate(game_input)), dim=0
        ).numpy()
        action = np.random.choice(range(3), p=network_output)
        game_over, reward = game.play_step(action)
        total_reward += reward
        if game_over:
            break

    return total_reward / i


def evaluate_genomes(
    genomes: neat.genome.DefaultGenome, config: neat.config.Config
) -> None:
    """
    Updates the fitness of the population.

    Parameters
    ----------
    genomes : Population.
    config  : NEAT configuration.
    """

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0.0
        games = 1
        for _ in range(games):
            fitness += play_game(net)
        genome.fitness = fitness / games


def fit(
    epochs: int, config_file: str, weights_path: str = "weights/agent_neat.pkl"
) -> None:
    """
    Trains the model and saves the best individual.

    Parameters
    ----------
    epochs       : Epochs to train the model.
    config_file  : Path to the config file.
    weights_path : Path to save the weights of the agent.
    """

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(evaluate_genomes, epochs)
    with open(weights_path, "wb") as file:
        pickle.dump(winner, file)


if __name__ == "__main__":
    EPOCHS = 10
    CONFIG_FILE = "src/agent_neat/config.txt"
    fit(EPOCHS, CONFIG_FILE)
