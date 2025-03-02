"""
Script for the training of the agent.
"""

from collections import deque
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm  # type: ignore

from src.snake import SnakeAI
from src.pytorch.model import Agent


class Trainer:
    """
    Class to train the agent.
    """

    def __init__(self, agent: Agent, lr: float = 1e-3) -> None:
        """
        Constructor of the class.

        Parameters
        ----------
        game   : Game.
        agent  : Model of the agent.
        lr     : Learning rate.
        """

        self.agent = agent
        self.optim = torch.optim.Adam(self.agent.parameters(), lr)
        self.rewards: list[float] = []

    def _simulate_one_episode(self) -> tuple[list[int], list[torch.Tensor]]:
        """
        Simulates one game.

        Returns
        -------
        Rewards and probs obtained in each step.
        """

        saved_probs: list[torch.Tensor] = []
        rewards: list[int] = []
        game = SnakeAI(visualize=False)

        game_over = False
        while not game_over:
            action, prob = self.agent.act(game)
            game_over, reward = game.play_step(action)
            rewards.append(reward)
            saved_probs.append(prob)

        return rewards, saved_probs

    @staticmethod
    def standardize(x: deque) -> torch.Tensor:
        """
        Function to standardize a list.

        Parameters
        ----------
        x : Deque to standardize.

        Returns
        -------
        Standardized deque.
        """

        eps = 1e-8
        x_tensor = torch.tensor(x)
        return (x_tensor - x_tensor.mean()) / (x_tensor.std() + eps)

    def fit(
        self, epochs: int, gamma: float = 0.1, print_every: int | None = None
    ) -> None:
        """
        Training of the agent.

        Parameters
        ----------
        epochs      : Number of epochs to train.
        gamma       : Gamma factor.
        print_every : Number of epochs to update the opponent.
        """

        if print_every is None:
            print_every = epochs // 10

        best_reward = -1e8

        for _ in tqdm(range(1, epochs + 1)):

            # Simulate one episode
            rewards, saved_probs = self._simulate_one_episode()
            self.rewards.append(sum(rewards) / len(rewards))

            # Calculate returns
            returns: deque = deque()
            for t in range(len(rewards))[::-1]:
                disc_return_t = returns[0] if len(returns) > 0 else 0
                returns.appendleft(gamma * disc_return_t + rewards[t])
            returns_tensor = self.standardize(returns).to(torch.float32)

            # Calculate loss function
            policy_loss = []
            for prob, disc_return in zip(saved_probs, returns_tensor):
                policy_loss.append(-prob * disc_return)
            policy_loss_tensor = torch.stack(policy_loss).sum()

            # Optimize policy
            self.optim.zero_grad()
            policy_loss_tensor.backward()
            self.optim.step()

            best_reward = self._update_agent_and_verbose(
                best_reward, epochs, print_every
            )

        self._plot_training()

    def _update_agent_and_verbose(
        self, best_reward: float, epochs: int, print_every: int
    ) -> float:
        """
        Prints on screen the state of the training and updates the agent.

        Parameters
        ----------
        epochs      : Total number of epochs.
        best_reward : Best reward obtained until now.
        print_every : Number of epochs to print on screen.
        """

        epoch = len(self.rewards)

        # Verbose
        if epoch % print_every == 0 or epoch == 1 or epoch == epochs + 1:
            print(f"Episode {epoch} / {epochs}\tScore: {self.rewards[-1]:.4f}")

        # Update best agent
        if self.rewards[-1] > best_reward:
            best_reward = self.rewards[-1]
            torch.save(self.agent.state_dict(), "weights/agent_pytorch.pt")
            print(f"Epoch: {epoch}, new best mean reward: {best_reward:.2f}")

        return best_reward

    def _plot_training(self) -> None:
        """
        Saves the plot of the rewards obtained during the training.
        """

        fig, axs = plt.subplots(1, 1, figsize=(16, 6))

        axs.grid()
        axs.set_xlabel("Epoch")
        axs.set_ylabel("Mean Reward")
        axs.plot(range(1, len(self.rewards) + 1), self.rewards)

        fig.savefig("images/train.png")
        plt.close(fig)


def main() -> None:
    """
    Function to train the agent.
    """

    agent = Agent()
    trainer = Trainer(agent)

    epochs = 10
    gamma = 0.1
    print_every = 10

    trainer.fit(epochs=epochs, gamma=gamma, print_every=print_every)


if __name__ == "__main__":
    main()
