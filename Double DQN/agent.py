import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from net import QNetwork
from buffer import ExperienceReplayBuffer

# Only for Jupyter Notebooks (for output display)
from IPython.display import clear_output


class RLAgent:
    def __init__(self, 
                 env: gym.Env,   
                 hidden_dim: int = 256,
                 discount_factor: float = 0.99,
                 batch_size: int = 256,
                 learning_rate: float = 1e-4,
                 update_controller: float = 0.005,
                 buffer_size: int = 300000,
                 max_epsilon: float = 1.0,
                 min_epsilon: float = 0.02,
                 eps_decay: float = 0.9995,
                 decay_steps: int = 1500,
                 target_update_freq: int = 100,
                 device: str = "cpu"):
        
        self.__env = env
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self.__action_dim = action_dim
        self.__device = device
        
        self.__main_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.__target_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.__target_network.load_state_dict(self.__main_network.state_dict())
        self.__target_network.eval()

        self.__discount_factor = discount_factor
        self.__batch_size = batch_size
        self.__tau = update_controller
        self.__target_update_freq = target_update_freq

        self.__epsilon = max_epsilon
        self.__max_eps = max_epsilon
        self.__min_eps = min_epsilon
        self.__eps_decay = eps_decay
        self.__decay_steps = decay_steps

        self.__buffer = ExperienceReplayBuffer(capacity=buffer_size)
        self.__trainer = optim.AdamW(self.__main_network.parameters(), lr=learning_rate) 
        self.__step_count = 0

    def __decide_from_network(self, state: np.ndarray):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.__device)
        with torch.no_grad():
            return self.__main_network(state).argmax().item()

    def act(self, state: np.ndarray, is_train: bool = False):
        if is_train:
            if np.random.rand() < self.__epsilon:
                return np.random.randint(0, self.__action_dim)
            else:
                return self.__decide_from_network(state)
        else:
            return self.__decide_from_network(state)

    def store(self, transition: object):
        self.__buffer.add(transition)

    def soft_update(self):
        for target_param, local_param in zip(self.__target_network.parameters(), self.__main_network.parameters()):
            target_param.data.copy_(self.__tau * local_param.data + (1.0 - self.__tau) * target_param.data)
            
    def update(self):
        if len(self.__buffer) < self.__batch_size:
            return

        batch = self.__buffer.sample(batch_size=self.__batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.__device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.__device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.__device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.__device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.__device)

        q_values = self.__main_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_actions = self.__main_network(next_states).argmax(dim=1)
            target_q_values = self.__target_network(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.__discount_factor * target_q_values * (1 - dones)

        loss = F.smooth_l1_loss(q_values, targets, reduction='mean')

        self.__trainer.zero_grad()
        loss.backward()
        self.__trainer.step()

        self.soft_update()
        self.__step_count += 1

    def train(self, episodes: int = 8000):
        rewards = []
        avg_rewards = []
        epsilons = []

        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        for episode in range(1, episodes + 1):
            state, _ = self.__env.reset()
            total_episode_reward = 0
            done = False

            while not done:
                action = self.act(state, is_train=True)
                next_state, reward, terminated, truncated, _ = self.__env.step(action)
                done = terminated or truncated
                self.store((state, action, reward, next_state, done))
                self.update()
                state = next_state
                total_episode_reward += reward

            rewards.append(total_episode_reward)
            epsilons.append(self.__epsilon)
            avg_reward = np.mean(rewards[-50:])
            avg_rewards.append(avg_reward)

            clear_output(wait=True)
            ax1.clear()
            ax2.clear()

            ax1.plot(rewards, label='Episode Reward')
            ax1.plot(avg_rewards, label='Moving Average (Last 50)')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title(f'Episode {episode} with Reward: {total_episode_reward:.2f} and Avg of Last 50: {avg_reward:.2f}')
            ax1.legend()

            ax2.plot(epsilons, color='orange', label='Epsilon')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Epsilon')
            ax2.set_title(f'Epsilon Decay Over Episodes Last Value = {self.__epsilon : .2f}')
            ax2.legend()

            display(fig)
            plt.pause(0.01)

            self.__epsilon = max(self.__epsilon * self.__eps_decay, self.__min_eps) 

        plt.ioff()
        plt.show()
        
    
    def save(self, filepath: str):
        torch.save(self.__main_network.state_dict(), filepath)
        print(f"AGENT: Model saved to {filepath} successfully!")

    def load(self, filepath: str, load_target: bool = True):
        self.__main_network.load_state_dict(torch.load(filepath, map_location=self.__device))
        if load_target:
            self.__target_network.load_state_dict(self.__main_network.state_dict())
        print(f"AGENT: Model loading complete from {filepath}!")