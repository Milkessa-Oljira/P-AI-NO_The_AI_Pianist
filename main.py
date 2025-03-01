import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import argparse

from models.rl_agent import PPOAgent
from models.critic_model import evaluate_piano_performance

class PianoEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    # Define base offsets for natural finger spread
    base_offsets_left = {'pinky': -2, 'ring': -1, 'middle': 0, 'index': 1, 'thumb': 2}
    base_offsets_right = {'thumb': -2, 'index': -1, 'middle': 0, 'ring': 1, 'pinky': 2}
    
    def __init__(self, render_mode=False, critic_model_path="critic_model.pth"):
        super(PianoEnv, self).__init__()
        self.render_mode = render_mode
        self.critic_model_path = critic_model_path
        
        # Timing parameters for dynamic rendering
        self.time_per_step = 0.1  # Seconds per step (10 FPS base)
        self.current_step = 0.0   # Track steps as float
        self.active_notes = []    # List of (key, release_step) tuples
        self.epsilon = 1e-5       # Small tolerance for floating-point precision
        
        # Observation space
        self.observation_space = spaces.Dict({
            "left_hand": spaces.MultiDiscrete([88] * 5),
            "right_hand": spaces.MultiDiscrete([88] * 5),
            "prev_finger": spaces.MultiBinary(10)
        })
        
        # Action space
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
        
        # Initial state with centers
        self.state = {
            "left_center": 24,
            "right_center": 60,
            "left_hand": np.array([24 + self.base_offsets_left[f] for f in ['pinky', 'ring', 'middle', 'index', 'thumb']], dtype=np.int32),
            "right_hand": np.array([60 + self.base_offsets_right[f] for f in ['thumb', 'index', 'middle', 'ring', 'pinky']], dtype=np.int32),
            "prev_finger": np.zeros(10, dtype=np.int32)
        }
        self.performance_sequence = []
        
        # Pygame initialization
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 400))
            pygame.display.set_caption('Piano RL Environment')
            self.clock = pygame.time.Clock()
            self.relative_pos = {0: 0, 1: 0.5, 2: 1, 3: 1.5, 4: 2, 5: 3, 6: 3.5, 7: 4, 8: 4.5, 9: 5, 10: 5.5, 11: 6}
            self.key_rects = self._init_key_rects()
    
    def _init_key_rects(self):
        key_rects = []
        total_units = 7 * 7 + self.relative_pos[87 % 12]
        scale = 1200 / total_units
        white_key_width = 24.5  # Adjusted to ensure keys touch
        black_key_width = 14
        white_key_height = 120
        black_key_height = 90
        base_y = 280
        
        for n in range(88):
            note = n % 12
            octave = n // 12
            rel_pos = self.relative_pos[note]
            pos = (octave * 7 + rel_pos) * scale
            is_white = note in {0, 2, 4, 5, 7, 9, 11}
            width = white_key_width if is_white else black_key_width
            height = white_key_height if is_white else black_key_height
            y = base_y - height if is_white else base_y - height - 30
            left = pos - width / 2
            rect = pygame.Rect(left, y, width, height)
            color = (240, 240, 240) if is_white else (20, 20, 20)
            key_rects.append((rect, color, is_white))
        return key_rects
    
    def step(self, action):
        self.current_step += 1.0
        reward = 0
        done = False
        info = {}
        
        # Extract stretch actions for left hand
        left_pinky_stretch = action["left_hand_finger_stretch"]["pinky"].item()  # 0 or 1
        left_thumb_stretch = action["left_hand_finger_stretch"]["thumb"].item()  # 0, 1, or 2
        left_pinky_offset = -3 if left_pinky_stretch == 0 else -2
        left_thumb_offset = 2 + left_thumb_stretch
        min_center_left = -left_pinky_offset  # e.g., 3 or 2
        max_center_left = 87 - left_thumb_offset  # e.g., 85, 84, or 83
        
        # Extract stretch actions for right hand
        right_thumb_stretch = action["right_hand_finger_stretch"]["thumb"].item()  # 0, 1, or 2
        right_pinky_stretch = action["right_hand_finger_stretch"]["pinky"].item()  # 0 or 1
        right_thumb_offset = -2 - right_thumb_stretch  # -2, -3, or -4
        right_pinky_offset = 2 + (1 if right_pinky_stretch == 1 else 0)  # 2 or 3
        min_center_right = -right_thumb_offset  # e.g., 2, 3, or 4
        max_center_right = 87 - right_pinky_offset  # e.g., 85 or 84
        
        # Update hand centers with dynamic clipping
        left_hor_move = action["left_hand_horizontal_movement"] - 87
        new_left_center = self.state["left_center"] + left_hor_move
        self.state["left_center"] = np.clip(new_left_center, min_center_left, max_center_left)
        
        right_hor_move = action["right_hand_horizontal_movement"] - 87
        new_right_center = self.state["right_center"] + right_hor_move
        self.state["right_center"] = np.clip(new_right_center, min_center_right, max_center_right)
        
        # Calculate finger positions for left hand
        left_fingers = np.array([
            self.state["left_center"] + left_pinky_offset,
            self.state["left_center"] + self.base_offsets_left['ring'],
            self.state["left_center"] + self.base_offsets_left['middle'],
            self.state["left_center"] + self.base_offsets_left['index'],
            self.state["left_center"] + left_thumb_offset
        ], dtype=np.int32)
        self.state["left_hand"] = np.clip(left_fingers, 0, 87)
        
        # Calculate finger positions for right hand
        right_fingers = np.array([
            self.state["right_center"] + right_thumb_offset,
            self.state["right_center"] + self.base_offsets_right['index'],
            self.state["right_center"] + self.base_offsets_right['middle'],
            self.state["right_center"] + self.base_offsets_right['ring'],
            self.state["right_center"] + right_pinky_offset
        ], dtype=np.int32)
        self.state["right_hand"] = np.clip(right_fingers, 0, 87)
        
        # Update active notes with epsilon tolerance
        self.active_notes = [(key, release) for key, release in self.active_notes if release > self.current_step + self.epsilon]
        fingers = action["finger_press"]
        durations = action["finger_duration"]
        for i in range(10):
            if fingers[i] == 1:
                key = self.state["left_hand"][i] if i < 5 else self.state["right_hand"][i - 5]
                duration_steps = durations[i] / self.time_per_step
                release_step = self.current_step + duration_steps
                self.active_notes.append((key, release_step))
        
        self.state["prev_finger"] = np.array(fingers, dtype=np.int32)
        
        # Record performance
        note = {
            "left": self.state["left_hand"].tolist(),
            "right": self.state["right_hand"].tolist(),
            "press": self.state["prev_finger"].tolist()
        }
        self.performance_sequence.append(note)
        
        # Compute punishment
        punishment = 0
        if np.any(self.state["left_hand"] <= 0) or np.any(self.state["left_hand"] >= 87):
            punishment -= 1
        if np.any(self.state["right_hand"] <= 0) or np.any(self.state["right_hand"] >= 87):
            punishment -= 1
        if abs(self.state["left_center"] - self.state["right_center"]) < 13:
            punishment -= 1
        reward = punishment
        
        if action["finished_playing"] == 1:
            done = True
            midi_sequence = self.generate_midi_sequence()
            critic_score = evaluate_piano_performance(midi_sequence, self.critic_model_path)
            steps = len(self.performance_sequence)
            step_scale = max(0.0, 1.0 - abs(steps - 300) / 300.0) if abs(steps - 300) >= 1 else 1.0
            reward += critic_score * step_scale
            info['critic_score'] = critic_score
            info['step_scale'] = step_scale
        else:
            done = False
            info['critic_score'] = 0
            info['step_scale'] = 1.0
        
        return self.state, reward, done, info, punishment
    
    def reset(self):
        self.current_step = 0.0
        self.active_notes = []
        self.state = {
            "left_center": 24,
            "right_center": 60,
            "left_hand": np.array([24 + self.base_offsets_left[f] for f in ['pinky', 'ring', 'middle', 'index', 'thumb']], dtype=np.int32),
            "right_hand": np.array([60 + self.base_offsets_right[f] for f in ['thumb', 'index', 'middle', 'ring', 'pinky']], dtype=np.int32),
            "prev_finger": np.zeros(10, dtype=np.int32)
        }
        self.performance_sequence = []
        return self.state
    
    def render(self, mode='human'):
        if not self.render_mode:
            return
        
        self.screen.fill((200, 200, 200))
        
        # Determine pressed keys from active_notes with epsilon tolerance
        pressed_keys = set(key for key, release_step in self.active_notes if release_step > self.current_step + self.epsilon)
        
        # Draw piano keys
        for n, (rect, base_color, is_white) in enumerate(self.key_rects):
            if n in pressed_keys:
                color = (180, 180, 180) if is_white else (60, 60, 60)
                rect.y += 5
            else:
                color = base_color
            pygame.draw.rect(self.screen, color, rect)
            if is_white:
                pygame.draw.line(self.screen, (150, 150, 150), (rect.left, rect.top), (rect.left, rect.bottom), 2)
                pygame.draw.line(self.screen, (150, 150, 150), (rect.left, rect.bottom), (rect.right, rect.bottom), 2)
                pygame.draw.line(self.screen, (255, 255, 255), (rect.left, rect.top), (rect.right, rect.top), 1)
            else:
                pygame.draw.line(self.screen, (50, 50, 50), (rect.left, rect.top), (rect.right, rect.top), 2)
            if n in pressed_keys:
                rect.y -= 5
        
        # Hand rendering
        palm_y = 300
        finger_length = 20
        pressing_y = 280
        
        # Left hand
        left_finger_xs = [self.key_rects[key][0].centerx for key in self.state["left_hand"]]
        for i, x in enumerate(left_finger_xs):
            is_pressing = self.state["left_hand"][i] in pressed_keys
            y_end = pressing_y if is_pressing else palm_y - finger_length
            color = (255, 100, 100) if is_pressing else (200, 80, 80)
            width = 7 if is_pressing else 5
            pygame.draw.line(self.screen, color, (x, palm_y), (x, y_end), width)
        
        # Left palm
        if left_finger_xs:
            palm_left = min(left_finger_xs) - 10
            palm_right = max(left_finger_xs) + 10
            palm_height = 20
            pygame.draw.rect(self.screen, (180, 80, 80), (palm_left, palm_y - palm_height / 2, palm_right - palm_left, palm_height))
        
        # Right hand
        right_finger_xs = [self.key_rects[key][0].centerx for key in self.state["right_hand"]]
        for i, x in enumerate(right_finger_xs):
            is_pressing = self.state["right_hand"][i] in pressed_keys
            y_end = pressing_y if is_pressing else palm_y - finger_length
            color = (100, 100, 255) if is_pressing else (80, 80, 200)
            width = 7 if is_pressing else 5
            pygame.draw.line(self.screen, color, (x, palm_y), (x, y_end), width)
        
        # Right palm
        if right_finger_xs:
            palm_left = min(right_finger_xs) - 10
            palm_right = max(right_finger_xs) + 10
            palm_height = 20
            pygame.draw.rect(self.screen, (80, 80, 180), (palm_left, palm_y - palm_height / 2, palm_right - palm_left, palm_height))
        
        pygame.display.flip()
        self.clock.tick(1 / self.time_per_step)  # e.g., 10 FPS for 0.1s/step
    
    def generate_midi_sequence(self):
        midi_sequence = []
        for note in self.performance_sequence:
            midi_sequence.extend(note["left"])
            midi_sequence.extend(note["right"])
        return midi_sequence
    
    def close(self):
        if self.render_mode:
            pygame.quit()

def train_agent(env, agent, num_episodes=1000):
    """
    Train the PPO agent with a modified reward calculation.
    The total reward is normalized between 0 and 1, equal to the critic's reward if no punishments,
    and reduced based on the magnitude of punishments otherwise.
    """
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
        episode_punishment = 0
        episode_steps = []
        
        while not done:
            obs_vector = np.concatenate([obs['left_hand'], obs['right_hand'], obs['prev_finger']])
            action, log_prob, value = agent.select_action(obs_vector)
            
            # Initialize or append to rollouts
            if not rollouts['actions']:
                for key, val in action.items():
                    rollouts['actions'][key] = [val]
            else:
                for key, val in action.items():
                    rollouts['actions'][key].append(val)
            rollouts['observations'].append(obs_vector)
            rollouts['log_probs'].append(log_prob)
            
            # Step the environment
            obs, reward, done, info, punishment = env.step(action)
            episode_punishment += punishment  # Accumulate punishment (negative)
            episode_steps.append((obs, reward))
            
            if env.render_mode:
                env.render()
        
        # Compute normalized total reward when episode ends
        if done:
            critic_score = info['critic_score']
            step_scale = info['step_scale']
            total_punishment_magnitude = -episode_punishment  # Convert to positive
            threshold = 300  # Tuning parameter, adjust as needed
            normalized_reward = critic_score * step_scale * np.exp(-total_punishment_magnitude / threshold)
            print(f"Episode {episode+1}: Normalized Reward {normalized_reward:.4f}")
            all_rewards.append(normalized_reward)
        
        # Compute returns and advantages for PPO learning
        returns = []
        discounted_sum = 0
        for _, reward in reversed(episode_steps):
            discounted_sum = reward + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = np.array(returns)
        advantages = returns.copy()  # Simplified; typically normalized in PPO
        rollouts['returns'].extend(returns.tolist())
        rollouts['advantages'].extend(advantages.tolist())
        
        # Update agent and reset rollouts
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
    parser.add_argument('--visualize', type=int, default=1, help='Set to >0 to visualize one never-ending episode with random actions')
    parser.add_argument('--critic_model_path', type=str, default='model_chkpts/critic_model.pth', help='Path to the pre-trained critic model')
    args = parser.parse_args()
    
    env = PianoEnv(render_mode=args.render, critic_model_path=args.critic_model_path)
    obs_dim = 20
    agent = PPOAgent(obs_dim)
    
    if args.visualize > 0:
        print("Visualizing random actions in a never-ending episode... (Close window or press Ctrl+C to stop)")
        obs = env.reset()
        total_reward = 0
        step_count = 0
        running = True
        try:
            while running:
                if args.render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            break
                
                action = env.action_space.sample()
                action["finger_duration"] = np.clip(action["finger_duration"], 0, 1.0)
                obs, reward, done, info, _ = env.step(action)  # Ignore punishment
                total_reward += reward
                step_count += 1
                
                if args.render:
                    env.render()
                
                if step_count % 100 == 0:
                    print(f"Step {step_count}: Total Reward = {total_reward}")
        
        except KeyboardInterrupt:
            print(f"Visualization stopped manually. Total Steps = {step_count}, Total Reward = {total_reward}")
        
        if not running:
            print(f"Visualization stopped (window closed). Total Steps = {step_count}, Total Reward = {total_reward}")
    
    if args.episodes > 0:
        print("Starting training...")
        rewards = train_agent(env, agent, num_episodes=args.episodes)
        print("Training complete. Episode rewards:", rewards)
        agent.save('model_chkpts/ppo_agent.pth')
    
    env.close()

if __name__ == '__main__':
    main()