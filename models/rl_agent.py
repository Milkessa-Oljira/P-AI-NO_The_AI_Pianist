# rl_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium import spaces

class ActorCritic(nn.Module):
    """
    A simple actor-critic network. In this implementation, we assume that the
    observation is flattened into a 1D vector and the action output is a flattened
    vector representing all controls.
    """
    def __init__(self, observation_space, action_space, hidden_dim=256):
        super(ActorCritic, self).__init__()
        # Assume observation_space is a Dict with a fixed flattened shape.
        obs_dim = np.prod(observation_space["left_hand"].shape) + \
                  np.prod(observation_space["right_hand"].shape) + \
                  np.prod(observation_space["prev_finger_press"].shape)
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # For simplicity, we assume the overall action dimension is 26 (see main.py)
        act_dim = 26
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch_size, obs_dim)
        x = torch.flatten(x, start_dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value

class RLAgent:
    """
    Implements a simple policy gradient (actor-critic) RL agent.
    """
    def __init__(self, observation_space, action_space, lr=1e-4, hidden_dim=256,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.policy = ActorCritic(observation_space, action_space, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def select_action(self, observation):
        """
        Given an observation (a dict), flatten it and sample an action.
        Returns a tuple (action, log_prob, value).
        """
        # Flatten observation into a single vector
        obs_vec = np.concatenate([
            observation["left_hand"],
            observation["right_hand"],
            observation["prev_finger_press"]
        ])
        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.policy.eval()
        with torch.no_grad():
            action_logits, value = self.policy(obs_tensor)
        # For demonstration, we assume a Gaussian policy with fixed std
        mean = action_logits
        std = torch.ones_like(mean) * 0.1
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy().flatten(), log_prob.cpu().item(), value.cpu().item()
    
    def update(self, trajectories):
        """
        Performs one update using a simple policy gradient update.
        'trajectories' is a dict with keys: observations, actions, rewards, log_probs, values.
        """
        # Compute discounted returns
        returns = []
        discounted_sum = 0
        gamma = 0.99
        for reward in trajectories['rewards'][::-1]:
            discounted_sum = reward + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        log_probs = torch.tensor(trajectories['log_probs'], dtype=torch.float32).to(self.device)
        values = torch.tensor(trajectories['values'], dtype=torch.float32).to(self.device)
        advantages = returns - values
        # Policy loss (using advantage)
        policy_loss = - (log_probs * advantages.detach()).mean()
        # Value loss (mean squared error)
        value_loss = advantages.pow(2).mean()
        loss = policy_loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)
    
    def load(self, filepath):
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
