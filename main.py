# main.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import torch
import time
import argparse

# Import the RL agent and critic evaluation from the models package
from models.rl_agent import PPOAgent
from models.critic_model import evaluate_piano_performance

class PianoEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, render_mode=False, critic_model_path="critic_model.pth"):
        super(PianoEnv, self).__init__()
        self.render_mode = render_mode
        self.critic_model_path = critic_model_path
        
        # Observation:
        # - left_hand: 5 discrete positions (0 to 87)
        # - right_hand: 5 discrete positions (0 to 87)
        # - prev_finger: 10 binary values (0 or 1) for previous step finger presses
        self.observation_space = spaces.Dict({
            "left_hand": spaces.MultiDiscrete([88] * 5),
            "right_hand": spaces.MultiDiscrete([88] * 5),
            "prev_finger": spaces.MultiBinary(10)
        })
        
        # Action space (enhanced) as a Dict:
        # 1. left/right hand horizontal movements: Discrete values in [0,174] which map to -87...+87.
        # 2. Left hand finger stretch: pinky (Discrete(2)), ring (Discrete(1)), middle (Discrete(1)), index (Discrete(1)), thumb (Discrete(3)).
        # 3. Right hand finger stretch: thumb (Discrete(3)), index (Discrete(1)), middle (Discrete(1)), ring (Discrete(1)), pinky (Discrete(2)).
        # 4. Finger press: MultiBinary(10)
        # 5. Continuous actions for finger velocity and duration (10 each).
        # 6. Pedal actions: continuous 3-values in [0,1]
        # 7. Finished playing flag: Discrete(2) (0: continue, 1: finish)
        self.action_space = spaces.Dict({
            "left_hand_horizontal_movement": spaces.Discrete(175),
            "right_hand_horizontal_movement": spaces.Discrete(175),
            "left_hand_finger_stretch": spaces.Dict({
                "pinky": spaces.Discrete(2),
                "ring": spaces.Discrete(1),
                "middle": spaces.Discrete(1),
                "index": spaces.Discrete(1),
                "thumb": spaces.Discrete(3)
            }),
            "right_hand_finger_stretch": spaces.Dict({
                "thumb": spaces.Discrete(3),
                "index": spaces.Discrete(1),
                "middle": spaces.Discrete(1),
                "ring": spaces.Discrete(1),
                "pinky": spaces.Discrete(2)
            }),
            "finger_press": spaces.MultiBinary(10),
            "finger_velocity": spaces.Box(low=0, high=127, shape=(10,), dtype=np.float32),
            "finger_duration": spaces.Box(low=0, high=10, shape=(10,), dtype=np.float32),
            "pedal_actions": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            "finished_playing": spaces.Discrete(2)
        })
        
        # Initial state: set default hand positions and previous finger press state.
        self.state = {
            "left_hand": np.array([20, 22, 24, 26, 28], dtype=np.int32),
            "right_hand": np.array([60, 62, 64, 66, 68], dtype=np.int32),
            "prev_finger": np.zeros(10, dtype=np.int32)
        }
        
        # Store the sequence of steps (notes) for performance MIDI generation.
        self.performance_sequence = []
        
        # Rendering initialization if required.
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 400))
            pygame.display.set_caption('Piano RL Environment')
            self.clock = pygame.time.Clock()
    
    def step(self, action):
        reward = 0
        done = False
        info = {}
        
        # Convert discrete horizontal movement to a signed integer: 0 maps to -87, 174 maps to +87.
        left_hor_move = action["left_hand_horizontal_movement"] - 87
        right_hor_move = action["right_hand_horizontal_movement"] - 87
        
        # Update hand positions by adding the horizontal movements and clip to piano bounds.
        self.state["left_hand"] = np.clip(self.state["left_hand"] + left_hor_move, 0, 87)
        self.state["right_hand"] = np.clip(self.state["right_hand"] + right_hor_move, 0, 87)
        
        # Process finger stretch:
        # Use the current hand's central position (median) as a base and apply offsets.
        left_center = int(np.median(self.state["left_hand"]))
        left_offsets = action["left_hand_finger_stretch"]
        # For left hand: pinky can move -1 step if value is 0; ring, middle, index remain 0; thumb: add value.
        left_fingers = np.array([
            left_center + (-1 if left_offsets["pinky"].item() == 0 else 0),
            left_center,
            left_center,
            left_center,
            left_center + left_offsets["thumb"].item()
        ])
        self.state["left_hand"] = left_fingers
        
        right_center = int(np.median(self.state["right_hand"]))
        right_offsets = action["right_hand_finger_stretch"]
        # For right hand: thumb: add value; index, middle, ring fixed; pinky: add 0 if value is 0, else +1.
        right_fingers = np.array([
            right_center + right_offsets["thumb"].item(),
            right_center,
            right_center,
            right_center,
            right_center + (0 if right_offsets["pinky"].item() == 0 else 1)
        ])
        self.state["right_hand"] = right_fingers
        
        # Update previous finger press state.
        self.state["prev_finger"] = np.array(action["finger_press"], dtype=np.int32)
        
        # Append current state to performance sequence.
        note = {
            "left": self.state["left_hand"].tolist(),
            "right": self.state["right_hand"].tolist(),
            "press": self.state["prev_finger"].tolist()
        }
        self.performance_sequence.append(note)
        
        # Out-of-bound penalty: if any hand position is at the boundaries, apply a penalty.
        if np.any(self.state["left_hand"] <= 0) or np.any(self.state["left_hand"] >= 87):
            reward -= 1
        if np.any(self.state["right_hand"] <= 0) or np.any(self.state["right_hand"] >= 87):
            reward -= 1
        
        # Hand overlap penalty: ensure distance between the medians of hands is at least 13.
        left_center = int(np.median(self.state["left_hand"]))
        right_center = int(np.median(self.state["right_hand"]))
        if abs(left_center - right_center) < 13:
            reward -= 1
        
        # Check if episode is finished.
        if action["finished_playing"] == 1:
            done = True
            # Generate a MIDI sequence from performance data and evaluate with the critic model.
            midi_sequence = self.generate_midi_sequence()
            critic_score = evaluate_piano_performance(midi_sequence, self.critic_model_path)
            # Scale the critic's score based on how close the step count is to 300.
            steps = len(self.performance_sequence)
            if abs(steps - 300) < 1:
                step_scale = 1.0
            else:
                step_scale = max(0.0, 1.0 - abs(steps - 300) / 300.0)
            reward += critic_score * step_scale
        
        return self.state, reward, done, info
    
    def reset(self):
        self.state = {
            "left_hand": np.array([20, 22, 24, 26, 28], dtype=np.int32),
            "right_hand": np.array([60, 62, 64, 66, 68], dtype=np.int32),
            "prev_finger": np.zeros(10, dtype=np.int32)
        }
        self.performance_sequence = []
        return self.state
    
    def render(self, mode='human'):
        if not self.render_mode:
            return
        self.screen.fill((255, 255, 255))
        # Draw 88 piano keys as rectangles.
        key_width = 1200 // 88
        for i in range(88):
            rect = pygame.Rect(i * key_width, 300, key_width, 100)
            pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
        # Draw left hand positions in red.
        for pos in self.state["left_hand"]:
            pygame.draw.circle(self.screen, (255, 0, 0), (int(pos * key_width + key_width / 2), 250), 10)
        # Draw right hand positions in blue.
        for pos in self.state["right_hand"]:
            pygame.draw.circle(self.screen, (0, 0, 255), (int(pos * key_width + key_width / 2), 250), 10)
        pygame.display.flip()
        self.clock.tick(30)
    
    def generate_midi_sequence(self):
        # For simplicity, generate a sequence of MIDI note pitches by combining left and right hand notes.
        midi_sequence = []
        for note in self.performance_sequence:
            midi_sequence.extend(note["left"])
            midi_sequence.extend(note["right"])
        return midi_sequence
    
    def close(self):
        if self.render_mode:
            pygame.quit()


def train_agent(env, agent, num_episodes=1000):
    # Rollout buffers to collect trajectories.
    rollouts = {
        'observations': [],
        'actions': {},
        'log_probs': [],
        'returns': [],
        'advantages': []
    }
    all_rewards = []
    gamma = agent.gamma
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = []
        while not done:
            # Flatten observation: left_hand (5) + right_hand (5) + prev_finger (10) = 20 dims.
            obs_vector = np.concatenate([obs['left_hand'], obs['right_hand'], obs['prev_finger']])
            action, log_prob, value = agent.select_action(obs_vector)
            
            # Save observation and action in rollout buffer.
            rollouts['observations'].append(obs_vector)
            if not rollouts['actions']:
                # Initialize keys for actions.
                for key, val in action.items():
                    rollouts['actions'][key] = [val]
            else:
                for key, val in action.items():
                    rollouts['actions'][key].append(val)
            rollouts['log_probs'].append(log_prob.item())
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps.append((obs, reward))
            
            if env.render_mode:
                env.render()
        
        # Compute discounted returns.
        returns = []
        discounted_sum = 0
        for (_, reward) in reversed(episode_steps):
            discounted_sum = reward + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = np.array(returns)
        advantages = returns.copy()  # For simplicity, use returns as advantages.
        rollouts['returns'].extend(returns.tolist())
        rollouts['advantages'].extend(advantages.tolist())
        
        all_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward {episode_reward}")
        
        # Update the agent after each episode.
        loss = agent.update(rollouts)
        rollouts = {
            'observations': [],
            'actions': {},
            'log_probs': [],
            'returns': [],
            'advantages': []
        }
    return all_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--render', action='store_true', help='Render the environment using pygame')
    parser.add_argument('--critic_model_path', type=str, default='critic_model.pth', help='Path to the pre-trained critic model')
    args = parser.parse_args()
    
    env = PianoEnv(render_mode=args.render, critic_model_path=args.critic_model_path)
    # The flattened observation is 5 + 5 + 10 = 20 dimensions.
    obs_dim = 20
    agent = PPOAgent(obs_dim)
    
    rewards = train_agent(env, agent, num_episodes=args.episodes)
    print("Training complete. Episode rewards:", rewards)
    
    # Save the trained agent.
    agent.save('ppo_agent.pth')
    env.close()


if __name__ == '__main__':
    main()
