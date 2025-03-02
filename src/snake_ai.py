"""
Script to play a game as an AI.
"""

import pygame

from src.snake_base import SnakeGame
from src.constants import (
    Direction,
    INCREMENT_SPEED,
    POSITIVE_REWARD,
    NEGATIVE_REWARD,
    LOSS_REWARD,
    WIDTH,
    HEIGHT,
    BLOCK_SIZE,
)


class SnakeAI(SnakeGame):
    """
    Class to represent the Snake game for AI playing.
    """

    def is_collision_point(self, point: tuple[int, int]) -> bool:
        """
        Checks if a point collides with the boundary or with snake.

        Parameters
        ----------
        point : Point to check.

        Returns
        -------
        Boolean variable indicating if there is collision or not.
        """

        point_x, point_y = point[0], point[1]

        # Hits boundary
        if (
            point_x > WIDTH - BLOCK_SIZE
            or point_x < 0
            or point_y > HEIGHT - BLOCK_SIZE
            or point_y < 0
        ):
            return True

        # Hits itself
        if (point_x, point_y) in self.snake[1:]:
            return True

        return False

    def play_step(self, action: int) -> tuple[bool, int]:
        """
        Moves the snake depending on the action chosen by the agent.

        Parameters
        ----------
        action : Action predicted by the agent.

        Returns
        -------
        Boolean variable indicating if the game is over or not and reward obtained.
        """

        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, 0

        # Move
        direction = self._get_direction(action)
        new_head = self._move(direction)
        self.snake.insert(0, new_head)

        # Update UI and clock
        self._update_ui()
        self.clock.tick(self.speed)

        # Check collision
        if self._is_collision():
            return True, LOSS_REWARD

        # Check if we eat food
        if self.snake[0] == self.food:
            self.score += 1
            self.food = self._place_food()
            self.speed += INCREMENT_SPEED
            return False, POSITIVE_REWARD

        # Delete last element
        self.snake.pop()

        return False, NEGATIVE_REWARD

    def _get_direction(self, action: tuple[int, int, int]) -> Direction:
        """
        Updates the position of the head of the snake. 0 is straight, 1 is right and 2
        is left. Note that 'behind' does not make sense because we would lose. We could
        use 0, 1, 2 and 3 for up, left, right, down, and it would also be valid, but
        this is more efficient.

        Parameters
        ----------
        action : Number representing the change of direction.

        Returns
        -------
        New direction of the snake.
        """

        if action == 0:
            direction = self.direction

        elif action == 1:
            if self.direction == Direction.RIGHT:
                direction = Direction.DOWN
            elif self.direction == Direction.LEFT:
                direction = Direction.UP
            elif self.direction == Direction.UP:
                direction = Direction.RIGHT
            else:
                direction = Direction.LEFT

        else:
            if self.direction == Direction.RIGHT:
                direction = Direction.UP
            elif self.direction == Direction.LEFT:
                direction = Direction.DOWN
            elif self.direction == Direction.UP:
                direction = Direction.LEFT
            else:
                direction = Direction.RIGHT

        return direction
