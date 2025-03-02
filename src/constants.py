"""
Script to define the constants used in the project.
"""

from enum import Enum
import pygame

pygame.init()


class Direction(Enum):
    """
    Class to define the possible directions we can choose.
    """

    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
LIGHT_BLUE = (0, 0, 255)
DARK_BLUE = (0, 100, 255)
BLACK = (0, 0, 0)

# Game
FONT = pygame.font.SysFont("arial", 25)
BLOCK_SIZE = 20
WIDTH = 640
HEIGHT = 480
INITIAL_SPEED = 10
INCREMENT_SPEED = 1

# Rewards
POSITIVE_REWARD = 5
NEGATIVE_REWARD = -1
LOSS_REWARD = -10
