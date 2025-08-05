import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import numpy as np
import os
import torch 

from agent import RLAgent



device = "cuda" if torch.cuda.is_available() else "cpu"
env = gym.make("LunarLander-v2", render_mode="rgb_array")

# Directory to save videos
video_dir = "lunarlander_videos"
os.makedirs(video_dir, exist_ok = True)

# Create environment with rgb_array for video support
agent = RLAgent(env, device = device)
agent.load("double_dqn_agent.pth")


# Attach video recorder wrapper to the environment
env = RecordVideo(
    env, 
    video_folder=video_dir,
    name_prefix="lunarlander",
    episode_trigger=lambda x: True  # Always record
)

num_episodes = 100
rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(state, is_train=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        total_reward += reward
        done = terminated or truncated

    rewards.append(total_reward)
    print(f"Episode {episode+1}: Reward = {total_reward}")

env.close()

avg_reward = np.mean(rewards)
print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward}")
print(f"Videos saved in: {video_dir}")