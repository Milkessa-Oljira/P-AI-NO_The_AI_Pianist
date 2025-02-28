import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import argparse

from models.rl_agent import PPOAgent
from models.critic_model import evaluate_piano_performance

class PianoEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, render_mode=False, critic_model_path="critic_model.pth"):
        super(PianoEnv, self).__init__()
        self.render_mode = render_mode
        self.critic_model_path = critic_model_path
        
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
        
        # Initial state
        self.state = {
            "left_hand": np.array([20, 22, 24, 26, 28], dtype=np.int32),
            "right_hand": np.array([60, 62, 64, 66, 68], dtype=np.int32),
            "prev_finger": np.zeros(10, dtype=np.int32)
        }
        self.performance_sequence = []
        
        # Pygame initialization for rendering
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 400))
            pygame.display.set_caption('Piano RL Environment')
            self.clock = pygame.time.Clock()
            self.relative_pos = {0: 0, 1: 0.5, 2: 1, 3: 1.5, 4: 2, 5: 3, 6: 3.5, 7: 4, 8: 4.5, 9: 5, 10: 5.5, 11: 6}
            self.key_rects = self._init_key_rects()
    
    def _init_key_rects(self):
        key_rects = []
        total_units = 7 * 7 + self.relative_pos[87 % 12]  # 88 keys span 50.5 units
        scale = 1200 / total_units
        white_key_width = 24
        black_key_width = 14
        white_key_height = 120
        black_key_height = 90
        base_y = 280  # Adjusted to fit hands below
        
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
        reward = 0
        done = False
        info = {}
        
        left_hor_move = action["left_hand_horizontal_movement"] - 87
        right_hor_move = action["right_hand_horizontal_movement"] - 87
        self.state["left_hand"] = np.clip(self.state["left_hand"] + left_hor_move, 0, 87)
        self.state["right_hand"] = np.clip(self.state["right_hand"] + right_hor_move, 0, 87)
        
        left_center = int(np.median(self.state["left_hand"]))
        left_offsets = action["left_hand_finger_stretch"]
        left_fingers = np.array([
            left_center + (-1 if left_offsets["pinky"].item() == 0 else 0),
            left_center,
            left_center,
            left_center,
            left_center + left_offsets["thumb"].item()
        ], dtype=np.int32)
        self.state["left_hand"] = np.clip(left_fingers, 0, 87)
        
        right_center = int(np.median(self.state["right_hand"]))
        right_offsets = action["right_hand_finger_stretch"]
        right_fingers = np.array([
            right_center + right_offsets["thumb"].item(),
            right_center,
            right_center,
            right_center,
            right_center + (0 if right_offsets["pinky"].item() == 0 else 1)
        ], dtype=np.int32)
        self.state["right_hand"] = np.clip(right_fingers, 0, 87)
        
        self.state["prev_finger"] = np.array(action["finger_press"], dtype=np.int32)
        
        note = {
            "left": self.state["left_hand"].tolist(),
            "right": self.state["right_hand"].tolist(),
            "press": self.state["prev_finger"].tolist()
        }
        self.performance_sequence.append(note)
        
        if np.any(self.state["left_hand"] <= 0) or np.any(self.state["left_hand"] >= 87):
            reward -= 1
        if np.any(self.state["right_hand"] <= 0) or np.any(self.state["right_hand"] >= 87):
            reward -= 1
        
        left_center = int(np.median(self.state["left_hand"]))
        right_center = int(np.median(self.state["right_hand"]))
        if abs(left_center - right_center) < 13:
            reward -= 1
        
        if action["finished_playing"] == 1:
            done = True
            midi_sequence = self.generate_midi_sequence()
            critic_score = 0
            # critic_score = evaluate_piano_performance(midi_sequence, self.critic_model_path)
            steps = len(self.performance_sequence)
            step_scale = max(0.0, 1.0 - abs(steps - 300) / 300.0) if abs(steps - 300) >= 1 else 1.0
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
        
        self.screen.fill((200, 200, 200))  # Light gray background
        
        # Determine pressed keys
        pressed_keys = set()
        for i in range(5):
            if self.state["prev_finger"][i] == 1:
                pressed_keys.add(self.state["left_hand"][i])
        for i in range(5):
            if self.state["prev_finger"][i + 5] == 1:
                pressed_keys.add(self.state["right_hand"][i])
        
        # Draw piano keys with 3D effect
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
        
        # Draw hands below keys
        finger_radius = 10
        hand_y_base = 300  # Positioned below keys (base_y=280)
        
        # Left hand (pinky to thumb)
        left_positions = []
        for i in range(5):
            key = self.state["left_hand"][i]
            if 0 <= key < 88:  # Index safety
                rect = self.key_rects[key][0]
                x = rect.centerx
                y_offset = i * 5
                y = hand_y_base + y_offset
                is_pressing = self.state["prev_finger"][i] == 1
                color = (255, 100, 100) if is_pressing else (200, 80, 80)
                if is_pressing:
                    y -= 10  # Move up when pressing
                pygame.draw.circle(self.screen, color, (int(x), int(y)), finger_radius)
                left_positions.append((int(x), int(y)))
        
        # Right hand (thumb to pinky)
        right_positions = []
        for i in range(5):
            key = self.state["right_hand"][i]
            if 0 <= key < 88:  # Index safety
                rect = self.key_rects[key][0]
                x = rect.centerx
                y_offset = (4 - i) * 5
                y = hand_y_base + y_offset
                is_pressing = self.state["prev_finger"][i + 5] == 1
                color = (100, 100, 255) if is_pressing else (80, 80, 200)
                if is_pressing:
                    y -= 10
                pygame.draw.circle(self.screen, color, (int(x), int(y)), finger_radius)
                right_positions.append((int(x), int(y)))
        
        # Draw palms
        if left_positions:
            palm_color = (180, 80, 80)
            palm_y = hand_y_base + 20
            palm_left = min(x for x, _ in left_positions)
            palm_right = max(x for x, _ in left_positions)
            palm_rect = pygame.Rect(palm_left, palm_y, palm_right - palm_left, 20)
            pygame.draw.rect(self.screen, palm_color, palm_rect, border_radius=5)
            pygame.draw.rect(self.screen, (150, 50, 50), palm_rect, 1)
        
        if right_positions:
            palm_color = (80, 80, 180)
            palm_y = hand_y_base + 20
            palm_left = min(x for x, _ in right_positions)
            palm_right = max(x for x, _ in right_positions)
            palm_rect = pygame.Rect(palm_left, palm_y, palm_right - palm_left, 20)
            pygame.draw.rect(self.screen, palm_color, palm_rect, border_radius=5)
            pygame.draw.rect(self.screen, (50, 50, 150), palm_rect, 1)
        
        pygame.display.flip()
        self.clock.tick(4)
    
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
            obs_vector = np.concatenate([obs['left_hand'], obs['right_hand'], obs['prev_finger']])
            action, log_prob, value = agent.select_action(obs_vector)
            
            if not rollouts['actions']:
                for key, val in action.items():
                    rollouts['actions'][key] = [val]
            else:
                for key, val in action.items():
                    rollouts['actions'][key].append(val)
            rollouts['observations'].append(obs_vector)
            rollouts['log_probs'].append(log_prob.item())
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps.append((obs, reward))
            
            if env.render_mode:
                env.render()
        
        returns = []
        discounted_sum = 0
        for (_, reward) in reversed(episode_steps):
            discounted_sum = reward + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = np.array(returns)
        advantages = returns.copy()
        rollouts['returns'].extend(returns.tolist())
        rollouts['advantages'].extend(advantages.tolist())
        
        all_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward {episode_reward}")
        
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
    parser.add_argument('--critic_model_path', type=str, default='critic_model.pth', help='Path to the pre-trained critic model')
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
                obs, reward, done, info = env.step(action)
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
        agent.save('ppo_agent.pth')
    
    env.close()

if __name__ == '__main__':
    main()