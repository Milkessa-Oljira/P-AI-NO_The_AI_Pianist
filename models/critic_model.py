# critic_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from music21 import midi
import argparse
from tqdm import tqdm

class MAESTRODataset(Dataset):
    """
    Dataset class for the MAESTRO dataset containing MIDI files.
    Each item is processed into a sequence of note pitches.
    """
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.file_paths = [os.path.join(dataset_root, f) for f in os.listdir(dataset_root)
                           if f.endswith('.midi') or f.endswith('.mid')]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        sequence = self.process_midi(file_path)
        return sequence

    def process_midi(self, file_path):
        try:
            mf = midi.MidiFile()
            mf.open(file_path)
            mf.read()
            mf.close()
            notes = []
            for track in mf.tracks:
                for event in track.events:
                    if event.isNoteOn():
                        notes.append(event.pitch)
            # Return as tensor of note pitches
            return torch.tensor(notes, dtype=torch.long)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return torch.tensor([], dtype=torch.long)

def collate_fn(batch):
    """
    Collate function to pad sequences in a batch to the maximum sequence length.
    """
    # Filter out any empty sequences
    batch = [item for item in batch if item.size(0) > 0]
    if len(batch) == 0:
        return torch.tensor([], dtype=torch.long)
    max_len = max(x.size(0) for x in batch)
    padded_batch = [torch.cat([x, torch.zeros(max_len - x.size(0), dtype=torch.long)]) for x in batch]
    return torch.stack(padded_batch)

class CriticModel(nn.Module):
    """
    Transformer-based Critic Model for evaluating piano performance.
    Given a sequence of MIDI note pitches, it outputs a score between 0 and 1.
    """
    def __init__(self, vocab_size=128, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super(CriticModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)          # (batch, seq_len, d_model)
        encoded = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        pooled = encoded.mean(dim=1)   # (batch, d_model) pooling over sequence length
        out = self.fc(pooled)          # (batch, 1)
        score = self.sigmoid(out)      # (batch, 1) score in [0,1]
        return score.squeeze()

def train_critic(dataset_root, save_path, epochs=10, batch_size=32, lr=1e-4, device='cpu'):
    """
    Train the CriticModel on the MAESTRO dataset.
    All performances in the dataset are high quality, so they are labeled as 1.
    """
    dataset = MAESTRODataset(dataset_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    model = CriticModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}"):
            if batch.numel() == 0:
                continue
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)  # Model outputs score for each sequence
            # Since all training samples are good performances, label them as 1
            labels = torch.ones_like(outputs, device=device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Critic model saved to {save_path}")

def evaluate_piano_performance(midi_sequence, model_path, device='cpu'):
    """
    Evaluate an RL-generated MIDI sequence using the pre-trained critic model.
    
    Args:
        midi_sequence (list[int] or torch.Tensor): A sequence of MIDI note pitches.
        model_path (str): Path to the saved critic model weights.
        device (str): Device to run the model on ('cpu' or 'cuda').
    
    Returns:
        float: A score between 0 and 1 indicating performance quality.
    """
    model = CriticModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        if not isinstance(midi_sequence, torch.Tensor):
            midi_tensor = torch.tensor(midi_sequence, dtype=torch.long, device=device).unsqueeze(0)
        else:
            midi_tensor = midi_sequence.unsqueeze(0).to(device)
        score = model(midi_tensor).item()
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, nargs='?', default='datasets/', help='Path to the MAESTRO dataset folder')
    parser.add_argument('--save_path', type=str, default='model_chkpts/critic_model.pth', help='Path to save the trained critic model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cpu or cuda')
    args = parser.parse_args()
    train_critic(args.dataset_root, args.save_path, args.epochs, args.batch_size, args.lr, args.device)
