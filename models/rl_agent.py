# rl_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOPolicy(nn.Module):
    def __init__(self, obs_dim):
        """
        Policy network with a shared backbone and separate heads for each action branch:
          - Discrete actions: left/right hand horizontal movements, finger stretches,
            binary decisions for finger presses, finished_playing flag.
          - Continuous actions: finger velocity, finger duration, pedal controls.
          - Also outputs a state value for the critic.
        """
        super(PPOPolicy, self).__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Discrete heads for hand horizontal movements
        self.left_hand_hor = nn.Linear(256, 175)   # actions: -87 ... +87 mapped to 175 categories
        self.right_hand_hor = nn.Linear(256, 175)
        
        # Left hand finger stretch (only pinky and thumb are variable; others are fixed to 0)
        self.left_pinky = nn.Linear(256, 2)   # pinky: 2 options
        self.left_thumb = nn.Linear(256, 3)   # thumb: 3 options
        
        # Right hand finger stretch (only thumb and pinky are variable)
        self.right_thumb = nn.Linear(256, 3)  # thumb: 3 options
        self.right_pinky = nn.Linear(256, 2)  # pinky: 2 options
        
        # Finger press details for 10 fingers:
        # Binary press decision, continuous velocity, and continuous duration.
        self.finger_press_logits = nn.Linear(256, 10 * 2)  # reshape to (batch, 10, 2)
        self.finger_velocity_mean = nn.Linear(256, 10)     # mean velocity (0-127)
        self.finger_duration_mean = nn.Linear(256, 10)     # mean duration (0-10)
        self.finger_velocity_log_std = nn.Parameter(torch.zeros(10))  # log std per finger
        self.finger_duration_log_std = nn.Parameter(torch.zeros(10))
        
        # Pedal actions: continuous values for sustain, sostenuto, una_corda (range [0,1])
        self.pedal_mean = nn.Linear(256, 3)
        self.pedal_log_std = nn.Parameter(torch.zeros(3))
        
        # Finished playing flag (discrete: continue vs. stop)
        self.finished_playing = nn.Linear(256, 2)
        
        # Value head for the critic (state value)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        # x: (batch, obs_dim)
        feat = self.backbone(x)
        
        # Discrete actions
        left_hand_hor_logits = self.left_hand_hor(feat)    # (batch, 175)
        right_hand_hor_logits = self.right_hand_hor(feat)  # (batch, 175)
        
        # Left hand finger stretch
        left_pinky_logits = self.left_pinky(feat)          # (batch, 2)
        left_thumb_logits = self.left_thumb(feat)          # (batch, 3)
        left_ring = torch.zeros(feat.size(0), 1, device=feat.device, dtype=torch.long)
        left_middle = torch.zeros(feat.size(0), 1, device=feat.device, dtype=torch.long)
        left_index = torch.zeros(feat.size(0), 1, device=feat.device, dtype=torch.long)
        
        # Right hand finger stretch
        right_thumb_logits = self.right_thumb(feat)        # (batch, 3)
        right_pinky_logits = self.right_pinky(feat)        # (batch, 2)
        right_index = torch.zeros(feat.size(0), 1, device=feat.device, dtype=torch.long)
        right_middle = torch.zeros(feat.size(0), 1, device=feat.device, dtype=torch.long)
        right_ring = torch.zeros(feat.size(0), 1, device=feat.device, dtype=torch.long)
        
        # Finger press: binary decision for each of 10 fingers
        finger_press_logits = self.finger_press_logits(feat).view(-1, 10, 2)  # (batch, 10, 2)
        
        # Continuous actions for finger velocity and duration
        finger_velocity_mean = self.finger_velocity_mean(feat)    # (batch, 10)
        finger_duration_mean = self.finger_duration_mean(feat)    # (batch, 10)
        
        # Pedal actions (continuous)
        pedal_mean = self.pedal_mean(feat)                        # (batch, 3)
        
        # Finished playing flag
        finished_playing_logits = self.finished_playing(feat)     # (batch, 2)
        
        # State value
        value = self.value_head(feat).squeeze(-1)                 # (batch,)
        
        return {
            'left_hand_hor_logits': left_hand_hor_logits,
            'right_hand_hor_logits': right_hand_hor_logits,
            'left_pinky_logits': left_pinky_logits,
            'left_thumb_logits': left_thumb_logits,
            'left_ring': left_ring,
            'left_middle': left_middle,
            'left_index': left_index,
            'right_thumb_logits': right_thumb_logits,
            'right_pinky_logits': right_pinky_logits,
            'right_index': right_index,
            'right_middle': right_middle,
            'right_ring': right_ring,
            'finger_press_logits': finger_press_logits,
            'finger_velocity_mean': finger_velocity_mean,
            'finger_duration_mean': finger_duration_mean,
            'finger_velocity_log_std': self.finger_velocity_log_std.expand_as(finger_velocity_mean),
            'finger_duration_log_std': self.finger_duration_log_std.expand_as(finger_duration_mean),
            'pedal_mean': pedal_mean,
            'pedal_log_std': self.pedal_log_std.expand_as(pedal_mean),
            'finished_playing_logits': finished_playing_logits,
            'value': value
        }
    
    def get_action_distribution(self, x):
        out = self.forward(x)
        dist = {}
        # Discrete distributions
        dist['left_hand_hor'] = torch.distributions.Categorical(logits=out['left_hand_hor_logits'])
        dist['right_hand_hor'] = torch.distributions.Categorical(logits=out['right_hand_hor_logits'])
        dist['left_pinky'] = torch.distributions.Categorical(logits=out['left_pinky_logits'])
        dist['left_thumb'] = torch.distributions.Categorical(logits=out['left_thumb_logits'])
        dist['right_thumb'] = torch.distributions.Categorical(logits=out['right_thumb_logits'])
        dist['right_pinky'] = torch.distributions.Categorical(logits=out['right_pinky_logits'])
        dist['finished_playing'] = torch.distributions.Categorical(logits=out['finished_playing_logits'])
        dist['finger_press'] = torch.distributions.Categorical(logits=out['finger_press_logits'])
        
        # Continuous distributions
        dist['finger_velocity'] = torch.distributions.Normal(out['finger_velocity_mean'],
                                                             torch.exp(out['finger_velocity_log_std']))
        dist['finger_duration'] = torch.distributions.Normal(out['finger_duration_mean'],
                                                             torch.exp(out['finger_duration_log_std']))
        dist['pedal'] = torch.distributions.Normal(out['pedal_mean'],
                                                   torch.exp(out['pedal_log_std']))
        return dist, out['value']
    
    def sample_action(self, x):
        """
        Sample an action from the policy and return the action dictionary,
        total log probability, and value estimate.
        """
        dist, value = self.get_action_distribution(x)
        action = {}
        log_prob = 0.0
        
        # Hand horizontal movements
        action['left_hand_horizontal_movement'] = dist['left_hand_hor'].sample()
        log_prob += dist['left_hand_hor'].log_prob(action['left_hand_horizontal_movement'])
        action['right_hand_horizontal_movement'] = dist['right_hand_hor'].sample()
        log_prob += dist['right_hand_hor'].log_prob(action['right_hand_horizontal_movement'])
        
        # Left hand finger stretch
        action['left_hand_finger_stretch'] = {
            'pinky': dist['left_pinky'].sample(),
            'ring': torch.zeros_like(action['left_hand_horizontal_movement']),
            'middle': torch.zeros_like(action['left_hand_horizontal_movement']),
            'index': torch.zeros_like(action['left_hand_horizontal_movement']),
            'thumb': dist['left_thumb'].sample()
        }
        log_prob += dist['left_pinky'].log_prob(action['left_hand_finger_stretch']['pinky'])
        log_prob += dist['left_thumb'].log_prob(action['left_hand_finger_stretch']['thumb'])
        
        # Right hand finger stretch
        action['right_hand_finger_stretch'] = {
            'thumb': dist['right_thumb'].sample(),
            'pinky': dist['right_pinky'].sample(),
            'index': torch.zeros_like(action['right_hand_horizontal_movement']),
            'middle': torch.zeros_like(action['right_hand_horizontal_movement']),
            'ring': torch.zeros_like(action['right_hand_horizontal_movement'])
        }
        log_prob += dist['right_thumb'].log_prob(action['right_hand_finger_stretch']['thumb'])
        log_prob += dist['right_pinky'].log_prob(action['right_hand_finger_stretch']['pinky'])
        
        # Finger press for 10 fingers
        action['finger_press'] = dist['finger_press'].sample()  # (batch, 10)
        log_prob += dist['finger_press'].log_prob(action['finger_press']).sum(dim=-1)
        
        # Finger velocity and duration
        action['finger_velocity'] = dist['finger_velocity'].sample()  # (batch, 10)
        log_prob += dist['finger_velocity'].log_prob(action['finger_velocity']).sum(dim=-1)
        action['finger_duration'] = dist['finger_duration'].sample()  # (batch, 10)
        log_prob += dist['finger_duration'].log_prob(action['finger_duration']).sum(dim=-1)
        
        # Pedal controls
        action['pedal_actions'] = dist['pedal'].sample()  # (batch, 3)
        log_prob += dist['pedal'].log_prob(action['pedal_actions']).sum(dim=-1)
        
        # Finished playing flag
        action['finished_playing'] = dist['finished_playing'].sample()
        log_prob += dist['finished_playing'].log_prob(action['finished_playing'])
        
        return action, log_prob, value

    def evaluate_actions(self, x, actions):
        """
        Compute log probabilities, entropy, and value estimates for PPO updates.
        """
        dist, value = self.get_action_distribution(x)
        log_prob = 0.0
        entropy = 0.0
        
        log_prob += dist['left_hand_hor'].log_prob(actions['left_hand_horizontal_movement'])
        entropy += dist['left_hand_hor'].entropy()
        log_prob += dist['right_hand_hor'].log_prob(actions['right_hand_horizontal_movement'])
        entropy += dist['right_hand_hor'].entropy()
        
        log_prob += dist['left_pinky'].log_prob(actions['left_hand_finger_stretch']['pinky'])
        entropy += dist['left_pinky'].entropy()
        log_prob += dist['left_thumb'].log_prob(actions['left_hand_finger_stretch']['thumb'])
        entropy += dist['left_thumb'].entropy()
        
        log_prob += dist['right_thumb'].log_prob(actions['right_hand_finger_stretch']['thumb'])
        entropy += dist['right_thumb'].entropy()
        log_prob += dist['right_pinky'].log_prob(actions['right_hand_finger_stretch']['pinky'])
        entropy += dist['right_pinky'].entropy()
        
        # Finger press
        finger_press_logits = self.forward(x)['finger_press_logits'].view(-1, 10, 2)
        finger_press_dist = torch.distributions.Categorical(logits=finger_press_logits)
        log_prob += finger_press_dist.log_prob(actions['finger_press']).sum(dim=-1)
        entropy += finger_press_dist.entropy().sum(dim=-1)
        
        log_prob += dist['finger_velocity'].log_prob(actions['finger_velocity']).sum(dim=-1)
        entropy += dist['finger_velocity'].entropy().sum(dim=-1)
        log_prob += dist['finger_duration'].log_prob(actions['finger_duration']).sum(dim=-1)
        entropy += dist['finger_duration'].entropy().sum(dim=-1)
        
        log_prob += dist['pedal'].log_prob(actions['pedal_actions']).sum(dim=-1)
        entropy += dist['pedal'].entropy().sum(dim=-1)
        
        log_prob += dist['finished_playing'].log_prob(actions['finished_playing'])
        entropy += dist['finished_playing'].entropy()
        
        return log_prob, entropy, value

class PPOAgent:
    def __init__(self, obs_dim, lr=3e-4, gamma=0.99, clip_param=0.2, value_coef=0.5, entropy_coef=0.01):
        self.device = torch.device('cpu')
        self.policy = PPOPolicy(obs_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def select_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.sample_action(obs_tensor)
            action = {k: v[0] if isinstance(v, torch.Tensor) and v.dim() > 0 else v for k, v in action.items()}
            log_prob = log_prob.item() if log_prob.dim() > 0 else log_prob
            value = value.item() if value.dim() > 0 else value
            action_np = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v) for k, v in action.items()}
            # Clip actions to match environment expectations
            if "finger_velocity" in action_np:
                action_np["finger_velocity"] = np.clip(action_np["finger_velocity"], 0, 127)
            if "finger_duration" in action_np:
                action_np["finger_duration"] = np.clip(action_np["finger_duration"], 0, 10)
            if "pedal_actions" in action_np:
                action_np["pedal_actions"] = np.clip(action_np["pedal_actions"], 0, 1)
        return action_np, log_prob, value

    def update(self, rollouts):
        # Convert observations efficiently
        obs = torch.tensor(np.array(rollouts['observations']), dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(rollouts['log_probs'], dtype=torch.float32).to(self.device)
        returns = torch.tensor(rollouts['returns'], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(rollouts['advantages'], dtype=torch.float32).to(self.device)
        
        # Convert actions to tensors
        actions_tensors = {}
        for key, val in rollouts['actions'].items():
            if isinstance(val[0], dict):  # Handle nested dictionaries (e.g., finger_stretch)
                actions_tensors[key] = {
                    subkey: torch.tensor([v[subkey] for v in val], dtype=torch.long).to(self.device)
                    for subkey in val[0]
                }
            else:
                if np.isscalar(val[0]):  # Scalar actions (e.g., left_hand_horizontal_movement)
                    actions_tensors[key] = torch.tensor(val, dtype=torch.long).to(self.device)
                else:  # Array actions (e.g., finger_press)
                    actions_tensors[key] = torch.tensor(np.stack(val), dtype=torch.float32).to(self.device)
        
        # PPO update loop
        for _ in range(10):  # Number of optimization epochs
            new_log_probs, entropy, values = self.policy.evaluate_actions(obs, actions_tensors)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            action_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - values).pow(2).mean()
            entropy_loss = -entropy.mean()
            loss = action_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == '__main__':
    # Example usage
    obs_dim = 50  # Adjust as needed
    agent = PPOAgent(obs_dim)
    dummy_obs = np.random.randn(obs_dim)
    action, log_prob, value = agent.select_action(dummy_obs)
    print("Sampled action:", action)