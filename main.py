import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pygame.midi
import argparse
import time

from models.rl_agent import PPOAgent
from models.critic_model import evaluate_piano_performance

class PianoEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, render_mode=False, critic_model_path="critic_model.pth"):
        super(PianoEnv, self).__init__()
        self.render_mode = render_mode
        self.critic_model_path = critic_model_path
        
        # Timing parameters
        self.time_per_step = 0.1  # Seconds per step
        self.current_step = 0.0
        self.active_notes = []  # List of tuples: (key, release_step, velocity)
        self.active_midi_notes = {}  # Dict: midi_note -> release_step
        self.epsilon = 1e-5
        
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
        
        # Initial state: hand centers and positions
        self.state = {
            "left_center": 24,
            "right_center": 60,
            "left_hand": np.array([24 - 2, 24 - 1, 24, 24 + 1, 24 + 2], dtype=np.int32),
            "right_hand": np.array([60 - 2, 60 - 1, 60, 60 + 1, 60 + 2], dtype=np.int32),
            "prev_finger": np.zeros(10, dtype=np.int32)
        }
        self.performance_sequence = []
        self.performance_velocities = []
        
        if self.render_mode:
            # Initialize Pygame and its MIDI module
            pygame.mixer.pre_init(44100, -16, 1, 512)
            pygame.midi.init()
            # Initialize MIDI output using the default output device
            default_id = pygame.midi.get_default_output_id()
            if default_id == -1:
                print("No default MIDI output device found. Disabling MIDI playback.")
                self.midi_out = None
            else:
                self.midi_out = pygame.midi.Output(default_id)
            pygame.init()
            self.screen = pygame.display.set_mode((1248, 400))
            pygame.display.set_caption('Piano RL Environment')
            self.clock = pygame.time.Clock()
            self.key_rects = self._init_key_rects()
            self.last_pressed_notes = set()
    
    def _init_key_rects(self):
        """Initialize key rectangles with realistic proportions and positions."""
        white_width = 24
        black_width = 14
        white_height = 120
        black_height = 80
        y_white = 200
        y_black = 180  # Black keys are slightly raised
        key_rects = []
        white_positions = {}
        white_count = 0
        
        for n in range(88):
            pitch_class = (21 + n) % 12  # MIDI notes 21 (A0) to 108 (C8)
            if pitch_class in [0, 2, 4, 5, 7, 9, 11]:  # White keys: C, D, E, F, G, A, B
                x = white_count * white_width
                white_positions[n] = x
                rect = pygame.Rect(x, y_white, white_width, white_height)
                key_rects.append((rect, (240, 240, 240), True))
                white_count += 1
            else:  # Black keys: C#, D#, F#, G#, A#
                left = max([k for k in white_positions if k < n], default=None)
                right = min([k for k in white_positions if k > n], default=None)
                if left is not None and right is not None:
                    x_left = white_positions[left]
                    x_right = white_positions[right]
                    x = (x_left + x_right) / 2 - black_width / 2
                elif left is not None:
                    x_left = white_positions[left]
                    x = x_left + white_width / 2 - black_width / 2
                elif right is not None:
                    x_right = white_positions[right]
                    x = x_right - white_width / 2 - black_width / 2
                else:
                    x = 0
                rect = pygame.Rect(x, y_black, black_width, black_height)
                key_rects.append((rect, (20, 20, 20), False))
        return key_rects
    
    def map_velocity(self, velocity):
        """Map continuous velocity (0-127) to a smooth perceived velocity."""
        low_threshold = 42
        mid_threshold = 85
        if velocity < low_threshold:
            return 21 + (velocity / low_threshold) * (64 - 21)
        elif velocity < mid_threshold:
            return 64 + ((velocity - low_threshold) / (mid_threshold - low_threshold)) * (106 - 64)
        else:
            return 106 + ((velocity - mid_threshold) / (127 - mid_threshold)) * (127 - 106)
    
    def map_duration(self, duration):
        """Map duration (0-10 sec) to seconds based on 120 BPM (0.5 sec per quarter note)."""
        quarter_notes = duration * 2  # Scale to 0-20 quarter notes
        return quarter_notes * 0.5
    
    def play_midi_note(self, key, velocity, duration):
        """
        Send a MIDI note_on event and schedule a note_off event.
        :param key: Piano key (0-87); will be converted to a MIDI note by adding 21.
        :param velocity: MIDI velocity (0-127)
        :param duration: Duration in seconds
        """
        midi_note = key + 21
        if self.midi_out is not None:
            self.midi_out.note_on(midi_note, int(velocity))
            off_time = self.current_step + (duration / self.time_per_step)
            self.active_midi_notes[midi_note] = off_time
        else:
            # Fallback: Use synthesized sound if no MIDI device available
            self.fallback_play_note(key, velocity, duration)   

    def fallback_play_note(self, key, velocity, duration):
        import numpy as np
        sample_rate = 44100
        midi_note = key + 21
        frequency = 440 * (2 ** ((midi_note - 69) / 12))
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Basic FM synthesis with ADSR envelope as a fallback
        attack_time = 0.05 * duration
        decay_time = 0.1 * duration
        sustain_level = 0.7
        release_time = 0.2 * duration
        envelope = np.ones_like(t)
        attack_samples = int(attack_time * sample_rate)
        decay_samples = int(decay_time * sample_rate)
        release_samples = int(release_time * sample_rate)
        sustain_samples = len(t) - attack_samples - decay_samples - release_samples
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if decay_samples > 0:
            envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain_level, decay_samples)
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)
        mod_index = 2.0
        mod_ratio = 2.0
        modulator = np.sin(2 * np.pi * frequency * mod_ratio * t)
        carrier = np.sin(2 * np.pi * frequency * t + mod_index * modulator)
        amplitude = velocity / 127.0
        waveform = carrier * envelope * amplitude
        waveform_int16 = np.int16(waveform * 32767)
        sound = pygame.sndarray.make_sound(waveform_int16)
        sound.play()    

    def step(self, action):
        self.current_step += 1.0
        reward = 0
        done = False
        info = {}
        
        # Update hand centers from horizontal movement actions
        left_hor_move = action["left_hand_horizontal_movement"] - 87
        self.state["left_center"] = np.clip(self.state["left_center"] + left_hor_move, 0, 87)
        right_hor_move = action["right_hand_horizontal_movement"] - 87
        self.state["right_center"] = np.clip(self.state["right_center"] + right_hor_move, 0, 87)
        
        # Update finger positions (simplified model)
        self.state["left_hand"] = np.clip(self.state["left_center"] + np.array([-2, -1, 0, 1, 2]), 0, 87)
        self.state["right_hand"] = np.clip(self.state["right_center"] + np.array([-2, -1, 0, 1, 2]), 0, 87)
        
        # Update active notes: remove expired ones
        self.active_notes = [(key, release, vel) for key, release, vel in self.active_notes if release > self.current_step + self.epsilon]
        # Process scheduled MIDI note offs
        if self.midi_out is not None:
            for midi_note, off_time in list(self.active_midi_notes.items()):
                if off_time <= self.current_step:
                    self.midi_out.note_off(midi_note, 0)
                    del self.active_midi_notes[midi_note]
        
        fingers = action["finger_press"]
        durations = action["finger_duration"]
        velocities = action["finger_velocity"]
        for i in range(10):
            if fingers[i] == 1:
                key = self.state["left_hand"][i] if i < 5 else self.state["right_hand"][i - 5]
                mapped_duration = self.map_duration(durations[i])
                release_step = self.current_step + mapped_duration / self.time_per_step
                mapped_velocity = self.map_velocity(velocities[i])
                self.active_notes.append((key, release_step, mapped_velocity))
                if self.render_mode:
                    self.play_midi_note(key, mapped_velocity, mapped_duration)
        
        self.state["prev_finger"] = np.array(fingers, dtype=np.int32)
        note = {
            "left": self.state["left_hand"].tolist(),
            "right": self.state["right_hand"].tolist(),
            "press": self.state["prev_finger"].tolist()
        }
        self.performance_sequence.append(note)
        self.performance_velocities.append([self.map_velocity(v) for v in velocities])
        
        # Compute punishment (simplified)
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
        self.active_midi_notes = {}
        self.state = {
            "left_center": 24,
            "right_center": 60,
            "left_hand": np.array([24 - 2, 24 - 1, 24, 24 + 1, 24 + 2], dtype=np.int32),
            "right_hand": np.array([60 - 2, 60 - 1, 60, 60 + 1, 60 + 2], dtype=np.int32),
            "prev_finger": np.zeros(10, dtype=np.int32)
        }
        self.performance_sequence = []
        self.performance_velocities = []
        return self.state
    
    def render(self, mode='human'):
        """Render a 3D-looking, highly animated realistic piano with MIDI-based sound."""
        if not self.render_mode:
            return
        
        self.screen.fill((220, 220, 220))
        
        # Determine pressed keys from the current state (immediate finger press)
        pressed_keys = set()
        for i, pressed in enumerate(self.state["prev_finger"]):
            if pressed == 1:
                if i < 5:
                    key = self.state["left_hand"][i]
                else:
                    key = self.state["right_hand"][i - 5]
                pressed_keys.add(key)

        # Draw keys with shading and press effects based on pressed_keys
        for n, (rect, base_color, is_white) in enumerate(self.key_rects):
            draw_rect = rect.copy()
            if n in pressed_keys:
                draw_rect.y += 5  # Simulate key depression
                color = (0, 0, 200)
            else:
                color = base_color
            pygame.draw.rect(self.screen, color, draw_rect)
            if is_white:
                pygame.draw.line(self.screen, (255, 255, 255), (draw_rect.left, draw_rect.top), (draw_rect.right, draw_rect.top), 1)
                pygame.draw.line(self.screen, (180, 180, 180), (draw_rect.left, draw_rect.bottom - 1), (draw_rect.right, draw_rect.bottom - 1), 1)
            else:
                pygame.draw.line(self.screen, (50, 50, 50), (draw_rect.left, draw_rect.top), (draw_rect.right, draw_rect.top), 1)
                pygame.draw.line(self.screen, (0, 0, 0), (draw_rect.left, draw_rect.bottom - 1), (draw_rect.right, draw_rect.bottom - 1), 1)
        
        # Draw piano body (frame)
        pygame.draw.rect(self.screen, (80, 40, 0), (0, 280, 1248, 120))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def generate_midi_sequence(self):
        note_events = []
        for t, note in enumerate(self.performance_sequence):
            for i in range(10):
                if note["press"][i] == 1:
                    key = note["left"][i] if i < 5 else note["right"][i - 5]
                    velocity = self.performance_velocities[t][i]
                    note_events.append((key + 21, velocity))
        midi_sequence = [pitch for pitch, _ in note_events]
        return midi_sequence
    
    def close(self):
        if self.render_mode:
            if self.midi_out is not None:
                self.midi_out.close()
            pygame.midi.quit()
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
        episode_punishment = 0
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
            rollouts['log_probs'].append(log_prob)
            
            obs, reward, done, info, punishment = env.step(action)
            episode_punishment += punishment
            episode_steps.append((obs, reward))
            
            if env.render_mode:
                env.render()
        
        if done:
            critic_score = info['critic_score']
            step_scale = info['step_scale']
            total_punishment_magnitude = -episode_punishment
            threshold = 300
            normalized_reward = critic_score * step_scale * np.exp(-total_punishment_magnitude / threshold)
            print(f"Episode {episode+1}: Normalized Reward {normalized_reward:.4f}")
            all_rewards.append(normalized_reward)
        
        returns = []
        discounted_sum = 0
        for _, reward in reversed(episode_steps):
            discounted_sum = reward + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = np.array(returns)
        advantages = returns.copy()
        rollouts['returns'].extend(returns.tolist())
        rollouts['advantages'].extend(advantages.tolist())
        
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
                action["finger_duration"] = np.clip(action["finger_duration"], 0, 10.0)
                obs, reward, done, info, _ = env.step(action)
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
