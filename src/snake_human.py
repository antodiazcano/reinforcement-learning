"""
Script to play a game as a human.
"""

import pygame

from src.snake_base import SnakeGame
from src.constants import Direction, INCREMENT_SPEED


class SnakeHuman(SnakeGame):
    """
    Class to represent the Snake game for playing as a human.
    """

    def play_step(self) -> bool:
        """
        Moves the snake depending on the key pressed by the user.

        Returns
        -------
        Boolean variable indicating if the game is over or not.
        """

        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        # Move
        new_head = self._move(self.direction)
        self.snake.insert(0, new_head)

        # Update UI and clock
        self._update_ui()
        self.clock.tick(self.speed)

        # Check collision
        if self._is_collision():
            return True

        # Check if we eat food
        if self.snake[0] == self.food:
            self.score += 1
            self.food = self._place_food()
            self.speed += INCREMENT_SPEED
        else:
            self.snake.pop()  # delete last element

        return False

    def play(self) -> None:
        """
        Plays a game.
        """

        while True:
            game_over = self.play_step()
            if game_over:
                break

        print(f"Final score: {self.score}")


if __name__ == "__main__":
    pygame.init()
    game = SnakeHuman()
    game.play()
    pygame.quit()
