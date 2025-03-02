"""
Script for the base class of the game.
"""

import random
from abc import ABC
import pygame

from src.constants import (
    Direction,
    WHITE,
    RED,
    LIGHT_BLUE,
    DARK_BLUE,
    BLACK,
    FONT,
    WIDTH,
    HEIGHT,
    BLOCK_SIZE,
    INITIAL_SPEED,
)


class SnakeGame(ABC):
    """
    Class to represent the Snake game.
    """

    def __init__(self) -> None:
        """
        Constructor of the class.
        """

        # Start pygame
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()

        # Snake attributes. The snake is represented as a list of pixels, initially 3,
        # and the head will always be the first element.
        head_x = WIDTH // 2
        head_y = HEIGHT // 2
        self.snake = [
            (head_x, head_y),
            (head_x - BLOCK_SIZE, head_y),
            (head_x - 2 * BLOCK_SIZE, head_y),
        ]

        # Other attributes
        self.direction = Direction.RIGHT
        self.speed = INITIAL_SPEED
        self.score = 0
        self.food = self._place_food()

    def _place_food(self) -> tuple[int, int]:
        """
        Places the food in a random cell.

        Returns
        -------
        New position of the food.
        """

        x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        return x, y

    def _move(self, direction: Direction) -> tuple[int, int]:
        """
        Updates the position of the head of the snake.

        Parameters
        ----------
        direction : Direction where the move has been made.

        Returns
        -------
        New coordinates of the head.
        """

        head_x = self.snake[0][0]
        head_y = self.snake[0][1]

        if direction == Direction.RIGHT:
            head_x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            head_x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            head_y += BLOCK_SIZE
        elif direction == Direction.UP:
            head_y -= BLOCK_SIZE

        return head_x, head_y

    def _is_collision(self) -> bool:
        """
        Checks if the snake has collided with the boundary or with itself.

        Returns
        -------
        Boolean variable indicating if there is collision or not.
        """

        head_x, head_y = self.snake[0]

        # Hits boundary
        if (
            head_x > WIDTH - BLOCK_SIZE
            or head_x < 0
            or head_y > HEIGHT - BLOCK_SIZE
            or head_y < 0
        ):
            return True

        # Hits itself
        if (head_x, head_y) in self.snake[1:]:
            return True

        return False

    def _update_ui(self) -> None:
        """
        Updates the user interface.
        """

        self.display.fill(BLACK)

        # Draw snake
        for point in self.snake:
            pygame.draw.rect(
                self.display,
                LIGHT_BLUE,
                pygame.Rect(point[0], point[1], BLOCK_SIZE, BLOCK_SIZE),
            )
            pygame.draw.rect(
                self.display,
                DARK_BLUE,
                pygame.Rect(
                    point[0] + BLOCK_SIZE // 5,
                    point[1] + BLOCK_SIZE // 5,
                    BLOCK_SIZE // 2,
                    BLOCK_SIZE // 2,
                ),
            )

        # Draw food
        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE),
        )

        # Draw score
        text = FONT.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])

        # Display
        pygame.display.flip()
