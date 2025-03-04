"""
Script for the training of the DQN agent.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # type:ignore

from src.agent_dqn.model import AgentDQN
from src.snake import SnakeAI


class QTrainer:
    """
    Class to train the DQN agent.
    """

    def __init__(self, model: AgentDQN, lr: float = 1e-3, c: int = 1) -> None:
        """
        Constructor of the class.

        Parameters
        ----------
        model : Agent used.
        lr    : Learning rate.
        c     : Q hat is updated every c epochs.
        """

        self.q = model
        self.q_hat = model
        self.q_hat.load_state_dict(self.q.state_dict())
        self.c = c
        self.optim = torch.optim.Adam(self.q.parameters(), lr)
        self.criterion = torch.nn.MSELoss()
        self.rewards: list[float] = []

    def _epsilon_policy(
        self,
        episode: int,
        min_epsilon: float = 0.01,
        max_epsilon: float = 1.0,
        decay_rate: float = 6e-3,
    ) -> bool:
        """
        Returns True to choose a random action and False otherwise. It is mostly for
        exploration at first and exploitation in the end. An exponential function is
        used to model the epsilon value.

        Parameters
        ----------
        episode     : Current episode.
        min_epsilon : Min value allowed for epsilon.
        max_epsilon : Max value allowed for epsilon.
        decay_rate  : Decay rate of the exponential function.

        Returns
        -------
        True if we have to act random and False otherwise.
        """

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * episode
        )
        dice = np.random.uniform()

        if dice < epsilon:
            return True
        return False

    def _simulate_one_episode(
        self, episode: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
        """
        Simulates one episode and obtains some information.

        Parameters
        ----------
        episode : Current episode.

        Returns
        -------
        List with all states, list with all q values and list with all rewards.
        """

        game = SnakeAI(visualize=False)
        states = [game.game_to_array()]
        q_values = []
        rewards = []

        game_over = False
        while not game_over:
            # Get random action or not depending on epsilon policy
            act_random = self._epsilon_policy(episode)
            if act_random:
                q_vals = self.q(game.game_to_array())
                action = np.random.randint(0, 3)
                q_value = q_vals[0, action]
            else:
                action, q_value = self.q.act(game.game_to_array())

            # Act and save information
            game_over, reward = game.play_step(action)
            q_values.append(q_value)
            rewards.append(reward)
            states.append(game.game_to_array())

        return states[:-1], q_values, rewards

    def fit(
        self,
        epochs: int,
        gamma: float = 0.9,
        image_path: str = "images/train_dqn.png",
        weights_path: str = "weights/agent_dqn.pt",
    ) -> None:
        """
        Trains the DQN.

        Parameters
        ----------
        epochs       : Number of epochs to train.
        gamma        : Gamma factor.
        image_path   : Path so save the evolution of the training.
        weights_path : Path to save the weights of the agent.
        """

        print_every = max(10, epochs // 10)
        best_reward = -1e8

        for epoch in tqdm(range(1, epochs + 1)):
            # Simulate episode
            states, q_values, rewards = self._simulate_one_episode(epoch)
            self.rewards.append(float(np.mean(rewards)))

            # Generate target values
            n_states = len(states)
            y = torch.zeros(n_states)
            for i in range(n_states):
                if i == n_states - 1:
                    y[i] = rewards[i]
                else:
                    y[i] = rewards[i] + gamma * torch.max(self.q_hat(states[i]))

            # Gradient descent
            loss = self.criterion(torch.stack(q_values), y)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Update Q hat
            if epoch % self.c == 0:
                self.q_hat.load_state_dict(self.q.state_dict())

            # Update best agent
            best_reward = self._update_agent_and_verbose(
                best_reward, epochs, print_every, weights_path
            )

        self.plot_training(image_path)

    def _update_agent_and_verbose(
        self, best_reward: float, epochs: int, print_every: int, path: str
    ) -> float:
        """
        Prints on screen the state of the training and updates the agent.

        Parameters
        ----------
        best_reward : Best reward obtained until now.
        epochs      : Total number of epochs.
        print_every : Number of epochs to print on screen.
        path        : Path to save the weights of the agent.
        """

        epoch = len(self.rewards)

        # Verbose
        if epoch % print_every == 0 or epoch == 1 or epoch == epochs + 1:
            print(f"Episode {epoch} / {epochs}\tScore: {self.rewards[-1]:.4f}")

        # Update best agent
        if self.rewards[-1] > best_reward:
            best_reward = self.rewards[-1]
            torch.save(self.q.state_dict(), path)
            print(f"Epoch: {epoch}, new best mean reward: {best_reward:.2f}")

        return best_reward

    def plot_training(self, path: str) -> None:
        """
        Saves the plot of the rewards obtained during the training.

        Parameters
        ----------
        path : Path so save the evolution of the training.
        """

        fig, axs = plt.subplots(1, 1, figsize=(16, 6))

        axs.grid()
        axs.set_xlabel("Epoch")
        axs.set_ylabel("Mean Reward")
        axs.plot(range(1, len(self.rewards) + 1), self.rewards)

        fig.savefig(path)
        plt.close(fig)


def main() -> None:
    """
    Function to train the agent.
    """

    agent = AgentDQN()
    trainer = QTrainer(agent)

    epochs = 1_000
    gamma = 0.9

    trainer.fit(epochs=epochs, gamma=gamma)


if __name__ == "__main__":
    main()
