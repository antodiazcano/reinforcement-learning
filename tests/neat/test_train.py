"""
Script to test the NEAT training.
"""

import os
import neat  # type: ignore

from src.agent_neat.train import play_game, evaluate_genomes, fit


CONFIG = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "src/agent_neat/config.txt",
)
GENOME = neat.DefaultGenome(0)
GENOMES = [(0, GENOME), (1, neat.DefaultGenome(1))]
NET = neat.nn.FeedForwardNetwork.create(GENOME, CONFIG)


def test_play_game() -> None:
    """
    Test for the play_game function.
    """

    result = play_game(NET)
    assert isinstance(result, float)


def test_evaluate_genomes() -> None:
    """
    Test for the evaluate_genomes function.
    """

    evaluate_genomes(GENOMES, CONFIG)
    assert GENOME.fitness is not None and isinstance(GENOME.fitness, float)
    assert GENOMES[1][1].fitness is not None and isinstance(
        GENOMES[1][1].fitness, float
    )


def test_fit() -> None:
    """
    Test for the fit function.
    """

    epochs = 1
    weights_path = "weights/tests/agent_neat_test.pkl"
    fit(epochs, "src/agent_neat/config.txt", weights_path=weights_path)
    assert os.path.exists(weights_path)
