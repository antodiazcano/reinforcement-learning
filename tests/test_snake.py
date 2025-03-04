"""
Script to test the Snake class.
"""

import pytest
import torch

from src.snake import SnakeAI
from src.constants import Direction, BLOCK_SIZE


@pytest.fixture
def game() -> SnakeAI:
    """
    Dummy fixture.
    """

    return SnakeAI(visualize=False)


def test_play_step(game: SnakeAI) -> None:
    """
    Test for the play_step function.
    """

    # Move is ok
    initial_position = game.snake[0]
    game.play_step(0)  # Move straight
    assert game.snake[0] != initial_position

    # Eating food is ok
    before_food = (game.snake[0][0] + BLOCK_SIZE, game.snake[0][1])
    game.food = before_food
    game.play_step(0)
    assert game.score == 1 and game.food != before_food


def test_move(game: SnakeAI) -> None:
    """
    Test for the _move function.
    """

    game = SnakeAI(visualize=False)
    initial_x, initial_y = game.snake[0]

    # Test move right
    new_x, new_y = game._move(Direction.RIGHT)
    assert new_x == initial_x + BLOCK_SIZE
    assert new_y == initial_y

    # Test move left
    new_x, new_y = game._move(Direction.LEFT)
    assert new_x == initial_x - BLOCK_SIZE
    assert new_y == initial_y

    # Test move up
    new_x, new_y = game._move(Direction.UP)
    assert new_x == initial_x
    assert new_y == initial_y - BLOCK_SIZE

    # Test move down
    new_x, new_y = game._move(Direction.DOWN)
    assert new_x == initial_x
    assert new_y == initial_y + BLOCK_SIZE


def test_is_collision(game: SnakeAI) -> None:
    """
    Tests for the is_collision function.
    """

    game.snake = [(0, 0), (BLOCK_SIZE, 0), (2 * BLOCK_SIZE, 0)]

    assert game.is_collision((-BLOCK_SIZE, 0))  # out of bounds
    assert game.is_collision((BLOCK_SIZE, 0))  # collision with itself
    assert not game.is_collision((0, BLOCK_SIZE))  # no collision


def test_distance_to_food(game: SnakeAI) -> None:
    """
    Test for the _distance_to_food function.
    """

    game.snake[0] = (100, 100)
    game.food = (150, 120)
    expected_distance = abs(100 - 150) + abs(100 - 120)
    assert game._distance_to_food() == expected_distance


@pytest.mark.parametrize(
    "current_direction, action, expected_direction",
    [
        (Direction.RIGHT, 0, Direction.RIGHT),
        (Direction.RIGHT, 1, Direction.DOWN),
        (Direction.RIGHT, 2, Direction.UP),
        (Direction.UP, 0, Direction.UP),
        (Direction.UP, 1, Direction.RIGHT),
        (Direction.UP, 2, Direction.LEFT),
        (Direction.LEFT, 0, Direction.LEFT),
        (Direction.LEFT, 1, Direction.UP),
        (Direction.LEFT, 2, Direction.DOWN),
        (Direction.DOWN, 0, Direction.DOWN),
        (Direction.DOWN, 1, Direction.LEFT),
        (Direction.DOWN, 2, Direction.RIGHT),
    ],
)
def test_get_direction(
    game: SnakeAI,
    current_direction: Direction,
    action: int,
    expected_direction: Direction,
) -> None:
    """
    Test for the _get_direction function.
    """

    game.direction = current_direction
    assert game._get_direction(action) == expected_direction


def test_game_to_array(game: SnakeAI) -> None:
    """
    Test for the game_to_array function.
    """

    state_tensor = game.game_to_array()
    assert isinstance(state_tensor, torch.Tensor)
    assert len(state_tensor.shape) == 1 and state_tensor.shape[0] == 11
