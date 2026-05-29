import torch
import gymnasium as gym

from dqn_agent import DQNAgent
from train import train


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        gamma=0.99,
        lr=3e-4,
        batch_size=128,
        buffer_size=10000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=1500,
        target_update=10,
    )

    scores, losses = train(agent, episodes=300)
    print("Training finished.")
    print(f"Best reward: {max(scores)}")
    print(f"Average reward: {sum(scores[-10:]) / min(10, len(scores)):.2f}")


if __name__ == "__main__":
    main()