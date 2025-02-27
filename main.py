# main.py
import gymnasium as gym
import numpy as np
import torch
import time
import pygame
import argparse
from gymnasium import spaces

# Import our models
from models.critic_model import CriticModel
from models.rl_agent import RLAgent

class PianoGymEnv(gym.Env):
    """
    Custom Gymnasium environment that simulates piano playing with two hands.
    The observation consists of left/right hand positions and the previous finger press state.
    The action (for demonstration) is a flattened vector representing:
      - left/right horizontal movements,
      - finger stretching for left/right hands,
      - binary decisions for finger presses,
      - pedal actions,
      - finished_playing flag.
    """
    def __init__(self):
        super(PianoGymEnv, self).__init__()
        # Define observation space.
        self.observation_space = spaces.Dict({
            "left_hand": spaces.Box(low=0, high=87, shape=(5,), dtype=np.int32),
            "right_hand": spaces.Box(low=0, high=87, shape=(5,), dtype=np.int32),
            "prev_finger_press": spaces.MultiBinary(10)
        })
        # Define a simplified action space.
        # For demonstration we use a continuous vector of length 26.
        self.action_space = spaces.Box(low=-10, high=10, shape=(26,), dtype=np.float32)
        
        self.max_steps = 300
        self.reset()
        
        # Initialize Pygame for rendering.
        pygame.init()
        self.screen = pygame.display.set_mode((800, 200))
        pygame.display.set_caption("Piano RL Environment")
    
    def reset(self):
        # Initial hand positions are set to mid-range positions.
        self.left_hand = np.array([40, 42, 44, 46, 48])
        self.right_hand = np.array([52, 54, 56, 58, 60])
        self.prev_finger_press = np.zeros(10, dtype=np.int32)
        self.current_step = 0
        self.episode_performance = []  # to record performance details for critic evaluation
        return self._get_obs()
    
    def _get_obs(self):
        return {
            "left_hand": self.left_hand,
            "right_hand": self.right_hand,
            "prev_finger_press": self.prev_finger_press
        }
    
    def step(self, action):
        """
        Interprets the action vector, updates the environment state, and computes reward.
        Action vector layout (26 values):
          0: left_hand_movement
          1: right_hand_movement
          2-6: left_finger_stretch (5 values)
          7-11: right_finger_stretch (5 values)
          12-21: finger_press decisions (10 values; thresholded to binary)
          22-24: pedal actions (sustain, sostenuto, una_corda)
          25: finished_playing flag (>0 means finish)
        """
        # Parse action
        left_hand_move = int(np.round(action[0]))
        right_hand_move = int(np.round(action[1]))
        left_stretch = np.round(action[2:7]).astype(int)
        right_stretch = np.round(action[7:12]).astype(int)
        finger_press = (action[12:22] > 0).astype(int)
        pedals = action[22:25]  # continuous values (to be scaled/interpreted)
        finished_flag = action[25] > 0
        
        # Update hand positions by applying horizontal movements
        self.left_hand = np.clip(self.left_hand + left_hand_move, 0, 87)
        self.right_hand = np.clip(self.right_hand + right_hand_move, 0, 87)
        
        # (For demonstration) we assume finger stretch simply offsets the base position of each hand.
        left_finger_positions = self.left_hand[0] + left_stretch
        right_finger_positions = self.right_hand[0] + right_stretch
        
        # Apply basic reward penalties:
        reward = 0
        # Punish if hands are too close (overlap) (threshold set to 13 units)
        if np.abs(np.mean(self.left_hand) - np.mean(self.right_hand)) < 13:
            reward -= 1
        # Punish if any hand goes out of piano bounds
        if (np.any(self.left_hand < 0) or np.any(self.left_hand > 87) or 
            np.any(self.right_hand < 0) or np.any(self.right_hand > 87)):
            reward -= 1
        
        # Update previous finger press state
        self.prev_finger_press = finger_press
        
        # Record the performance data for later critic evaluation.
        self.episode_performance.append({
            "left_hand": self.left_hand.copy(),
            "right_hand": self.right_hand.copy(),
            "finger_press": finger_press,
            "pedals": pedals
        })
        
        self.current_step += 1
        done = finished_flag or (self.current_step >= self.max_steps)
        
        # If episode is done, use the critic to compute the final reward.
        if done:
            performance_rep = self._convert_performance_to_representation(self.episode_performance)
            critic = CriticModel()
            try:
                # Attempt to load a pre-trained critic; if not found, use a default score.
                critic.load("critic_model.pth")
                critic_score = critic.score_performance(performance_rep)
            except Exception:
                critic_score = 0.5  # fallback neutral score
            # Scale the critic score by proximity to 300 steps (ideal length)
            step_scaling = max(0.0, 1.0 - abs(self.current_step - self.max_steps) / self.max_steps)
            reward += critic_score * step_scaling
        
        return self._get_obs(), reward, done, {}
    
    def _convert_performance_to_representation(self, performance):
        """
        Converts the stored performance (list of dicts) into a 2D tensor representation.
        For each timestep, we concatenate:
          left_hand (5), right_hand (5), finger_press (10), pedals (3)
        Total feature dimension = 23.
        For demonstration, we pad or project this 23-dim vector to a fixed 128-dim feature vector.
        """
        rep = []
        for entry in performance:
            features = np.concatenate([
                entry["left_hand"],
                entry["right_hand"],
                entry["finger_press"],
                entry["pedals"]
            ])
            # Pad to 128 dimensions
            if features.shape[0] < 128:
                pad_width = 128 - features.shape[0]
                features = np.pad(features, (0, pad_width), 'constant')
            rep.append(features)
        rep = np.array(rep, dtype=np.float32)
        return rep
    
    def render(self, mode='human'):
        """
        Render the piano keys and hand positions using Pygame.
        """
        self.screen.fill((255, 255, 255))
        key_width = self.screen.get_width() // 88
        # Draw piano keys
        for i in range(88):
            rect = pygame.Rect(i * key_width, 100, key_width, 100)
            pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
        # Draw simplified hand positions as circles (using first key of each hand)
        left_x = int(self.left_hand[0] * key_width + key_width // 2)
        right_x = int(self.right_hand[0] * key_width + key_width // 2)
        pygame.draw.circle(self.screen, (255, 0, 0), (left_x, 80), 10)
        pygame.draw.circle(self.screen, (0, 0, 255), (right_x, 80), 10)
        pygame.display.flip()
    
    def close(self):
        pygame.quit()

def train():
    env = PianoGymEnv()
    agent = RLAgent(env.observation_space, env.action_space)
    num_episodes = 1000
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        trajectories = {"observations": [], "actions": [], "rewards": [], "log_probs": [], "values": []}
        episode_reward = 0
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            # Select an action from the agent.
            flat_obs = np.concatenate([obs["left_hand"], obs["right_hand"], obs["prev_finger_press"]])
            action, log_prob, value = agent.select_action(obs)
            trajectories["observations"].append(flat_obs)
            trajectories["actions"].append(action)
            trajectories["log_probs"].append(log_prob)
            trajectories["values"].append(value)
            obs, reward, done, _ = env.step(action)
            trajectories["rewards"].append(reward)
            episode_reward += reward
            env.render()
            time.sleep(0.05)
        loss = agent.update(trajectories)
        print(f"Episode {episode+1}: Total Reward = {episode_reward:.2f}, Loss = {loss:.4f}")
        # Save models periodically
        if (episode+1) % 100 == 0:
            agent.save("rl_agent.pth")
            # Optionally save the critic as well if trained further.
            critic = CriticModel()
            critic.save("critic_model.pth")
    env.close()

def test():
    env = PianoGymEnv()
    agent = RLAgent(env.observation_space, env.action_space)
    agent.load("rl_agent.pth")
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                return
        action, _, _ = agent.select_action(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(0.05)
    print("Test episode finished with reward:", total_reward)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        test()
