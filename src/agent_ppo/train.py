"""
Script for the training of the PPO agent.
"""

from collections import deque
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm  # type: ignore

from src.snake import SnakeAI
from src.agent_ppo.model import AgentPPO


class PPOTrainer:
    """
    Class to train the PPO agent.
    """

    def __init__(self, agent: AgentPPO, lr: float = 1e-3) -> None:
        """
        Constructor of the class.

        Parameters
        ----------
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
        Rewards and log probs obtained in each step.
        """

        saved_log_probs: list[torch.Tensor] = []
        rewards: list[int] = []
        game = SnakeAI(visualize=False)

        game_over = False
        while not game_over:
            action, log_prob = self.agent.act(game)
            game_over, reward = game.play_step(action)
            rewards.append(reward)
            saved_log_probs.append(log_prob)

        return rewards, saved_log_probs

    @staticmethod
    def standardize(x: deque) -> torch.Tensor:
        """
        Function to standardize a deque.

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
        self,
        epochs: int,
        gamma: float = 0.9,
        image_path: str = "images/train_ppo.png",
        weights_path: str = "weights/agent_ppo.pt",
    ) -> None:
        """
        Training of the agent.

        Parameters
        ----------
        epochs       : Number of epochs to train.
        gamma        : Gamma factor.
        image_path   : Path so save the evolution of the training.
        weights_path : Path to save the weights of the agent.
        """

        print_every = max(10, epochs // 10)
        best_reward = -1e8

        for _ in tqdm(range(1, epochs + 1)):
            # Simulate one episode
            rewards, saved_log_probs = self._simulate_one_episode()
            self.rewards.append(sum(rewards) / len(rewards))

            # Calculate returns
            returns: deque = deque()
            for t in range(len(rewards))[::-1]:
                disc_return_t = returns[0] if len(returns) > 0 else 0
                returns.appendleft(gamma * disc_return_t + rewards[t])
            returns_tensor = self.standardize(returns).to(torch.float32)

            # Calculate loss function
            policy_loss = []
            for prob, disc_return in zip(saved_log_probs, returns_tensor):
                policy_loss.append(-prob * disc_return)
            policy_loss_tensor = torch.stack(policy_loss).sum()

            # Optimize policy
            self.optim.zero_grad()
            policy_loss_tensor.backward()
            self.optim.step()

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
            torch.save(self.agent.state_dict(), path)
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

    agent = AgentPPO()
    trainer = PPOTrainer(agent)

    epochs = 1_000
    gamma = 0.9

    trainer.fit(epochs=epochs, gamma=gamma)


if __name__ == "__main__":
    main()
