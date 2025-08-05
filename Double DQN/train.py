import gymnasium as gym 
import torch

from agent import RLAgent



device = "cuda" if torch.cuda.is_available() else "cpu"
env = gym.make("LunarLander-v2")


########### Train the agent ##################
agent = RLAgent(env, device = device) 
agent.train()