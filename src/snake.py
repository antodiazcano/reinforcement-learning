"""
Script to play a game as an AI.
"""

import random
import pygame

from src.constants import (
    Direction,
    INCREMENT_SPEED,
    POSITIVE_REWARD,
    NEGATIVE_REWARD,
    LOSS_REWARD,
    WIDTH,
    HEIGHT,
    BLOCK_SIZE,
    INITIAL_SPEED,
    RED,
    WHITE,
    BLACK,
    LIGHT_BLUE,
    DARK_BLUE,
    FONT,
)


class SnakeAI:
    """
    Class to represent the Snake game for AI playing.
    """

    def __init__(self, visualize: bool = True) -> None:
        """
        Constructor of the class.

        Parameters
        ----------
        visualize : True if we want to show the game on screen and False otherwise. When
            training is better to set it to False.
        """

        # Start pygame
        self.visualize = visualize
        if self.visualize:
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

    def is_collision(self, point: tuple[int, int]) -> bool:
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

        # Move
        direction = self._get_direction(action)
        new_head = self._move(direction)
        self.snake.insert(0, new_head)

        # Update UI and clock
        if self.visualize:
            self._update_ui()
        self.clock.tick(self.speed)

        # Check collision
        if self.is_collision(self.snake[0]):
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

    def _get_direction(self, action: int) -> Direction:
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
