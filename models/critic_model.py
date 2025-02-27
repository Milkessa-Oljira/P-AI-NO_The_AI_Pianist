# critic_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerCritic(nn.Module):
    """
    A transformer-based network that takes in a sequence representing a full performance
    and outputs a quality score between 0 and 1.
    """
    def __init__(self, input_dim=128, num_heads=8, num_layers=4, hidden_dim=512, dropout=0.1):
        super(TransformerCritic, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (seq_len, batch_size, input_dim)
        x = self.embedding(x)  # shape: (seq_len, batch_size, hidden_dim)
        x = self.transformer_encoder(x)
        # Use mean pooling over sequence length
        x = x.mean(dim=0)  # shape: (batch_size, hidden_dim)
        x = self.fc_out(x)
        x = torch.sigmoid(x)  # score in [0,1]
        return x

class CriticModel:
    """
    Wraps the TransformerCritic with training, saving, loading, and a callable
    interface for scoring an episode performance.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # input_dim should match the feature vector length produced by the environment’s conversion
        self.model = TransformerCritic(input_dim=128).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.BCELoss()  # since output is [0,1]
    
    def train_step(self, input_sequence, target_score):
        """
        Performs one training step.
        :param input_sequence: Tensor of shape (seq_len, batch_size, input_dim)
        :param target_score: Tensor of shape (batch_size, 1)
        """
        self.model.train()
        self.optimizer.zero_grad()
        input_sequence = input_sequence.to(self.device)
        target_score = target_score.to(self.device)
        output = self.model(input_sequence)
        loss = self.criterion(output, target_score)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def score_performance(self, episode_midi_representation):
        """
        Given an episode’s performance representation (e.g. a numpy array of shape
        (seq_len, input_dim)), returns a score between 0 and 1.
        """
        if not torch.is_tensor(episode_midi_representation):
            episode_midi_representation = torch.tensor(episode_midi_representation, dtype=torch.float32)
        # Add a batch dimension and permute to shape (seq_len, batch_size, input_dim)
        episode_midi_representation = episode_midi_representation.unsqueeze(1).to(self.device)
        self.model.eval()
        with torch.no_grad():
            score = self.model(episode_midi_representation)
        return score.item()
    
    def save(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    
    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
