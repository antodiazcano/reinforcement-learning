"""
Script to train the neat model.
"""

import pickle
import neat  # type:ignore

from src.snake import SnakeAI


def play_game(net: neat.nn.FeedForwardNetwork) -> float:
    """
    The agent plays a game vs minimax.

    Parameters
    ----------
    net : Net that models the actions of the agent.
    """

    game = SnakeAI(visualize=False)
    total_reward = 0.0

    max_steps = 100
    for i in range(max_steps):
        game_input = game.game_to_array().numpy()
        action = net.activate(game_input)
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
    """

    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness = 0.0
        games = 1
        for _ in range(games):
            fitness += play_game(net)
        genome.fitness = fitness / games


def fit(config_file: str) -> None:
    """
    Trains the model and saves the best individual.

    Parameters
    ----------
    config_file : Path to the config file.
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

    epochs = 10
    winner = population.run(evaluate_genomes, epochs)
    with open("weights/agent_neat.pkl", "wb") as file:
        pickle.dump(winner, file)


if __name__ == "__main__":
    fit("src/agent_neat/config.txt")
