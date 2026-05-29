import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt


def train(agent, env_name="CartPole-v1", episodes=300, max_steps=500, save_path="dqn_cartpole.pth"):
    env = gym.make(env_name)
    scores = []
    losses = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            loss = agent.optimize()
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

            if done:
                break

        scores.append(total_reward)

        if (ep + 1) % agent.target_update == 0:
            agent.update_target()

        if (ep + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print(f"Episode {ep + 1}: avg reward (last 10) = {avg_score:.1f}")

    agent.save(save_path)
    env.close()

    plt.figure(figsize=(10, 4))
    plt.plot(scores, label="episode reward")
    if len(scores) >= 10:
        rolling = np.convolve(scores, np.ones(10) / 10, mode="valid")
        plt.plot(range(9, len(scores)), rolling, label="rolling mean (10)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN on CartPole-v1")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.show()

    return scores, losses