import torch
import torch.nn as nn
import numpy as np

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

# Prettify printing tensors when debugging
#import lovely_tensors as lt
#lt.monkey_patch()


################################################################################
# Model

from spectral_ssm import SpectralSSM

class AudioRNN(nn.Module):
    def __init__(self, dim, d_hidden):
        super(AudioRNN, self).__init__()

        # batch_first=True: The input/output will be batched
        self.rnn = nn.RNN(dim, d_hidden, batch_first=True)

        # Project to output dim
        self.fc = nn.Linear(d_hidden, dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out


class SimpleMLP(nn.Module):
    def __init__(self, dim, d_hidden):
        super(SimpleMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, dim)
        )

    def forward(self, x):
        return self.net(x)

import torch.nn.functional as F

class SimpleRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = SimpleRMSNorm(dim)

    def forward(self, x):
        return self.norm(self.fn(x) + x)

class AudioMLP(nn.Module):
    def __init__(self, dim, d_hidden):
        super(AudioMLP, self).__init__()

        self.encode = PreNormResidual(dim, SimpleMLP(dim, d_hidden))

    def forward(self, x):
        x = self.encode(x)
        return x


################################################################################
# Dataset

from torch.utils.data import Dataset, DataLoader, random_split

import torchaudio
from torchaudio.transforms import MFCC
from sklearn.preprocessing import StandardScaler

def preprocess_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    # Adjust n_mels here if necessary, and ensure n_fft and hop_length are appropriate for your sample rate
    mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=12, melkwargs={'n_mels': 40, 'n_fft': 400, 'hop_length': 160})
    mfcc = mfcc_transform(waveform)

    # Average across channels if stereo audio
    if mfcc.dim() > 2:
        mfcc = mfcc.mean(dim=0)

    # Standardize features
    scaler = StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc.numpy())

    # Swap dimensions: We want mfcc features in dim=-1
    result = np.transpose(mfcc_scaled)

    return result

def segment_audio(mfcc, segment_length):
    num_segments = mfcc.shape[0] // segment_length
    segments = mfcc[:num_segments*segment_length].reshape(num_segments, segment_length, -1)
    return segments

import os
import multiprocessing
from multiprocessing import Pool

def dataset_process_file(file_path):
    processed_audio = preprocess_audio(file_path)
    # +1 because we remove one element in __getitem__ below
    parts = segment_audio(processed_audio, args.segment_length + 1)
    return parts

class AudioSegmentDataset(Dataset):
    def __init__(self, args):

        audio_files = []
        for filename in os.listdir(args.dir):
            # torchaudio does not have an API to check if a file extension is supported
            if filename.lower().endswith(('.wav', '.mp3', '.flac')):
                audio_files.append(os.path.join(args.dir, filename))

        print(f"Loading {len(audio_files)} audio files..")

        # Use multiprocessing Pool to process files in parallel
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            all_parts = pool.map(dataset_process_file, audio_files)

        # Combine parts into one dataset
        self.segments = np.concatenate(all_parts, axis=0)

        print(f"Loaded dataset shape: {self.segments.shape}")

    def get_feature_dim(self):
        return self.segments.shape[2]

    def __len__(self):
        return self.segments.shape[0]

    def __getitem__(self, idx):
        # Segment shape: (sequence_length, num_features)
        segment = self.segments[idx]

        # Input features: All time steps except the last
        input_features = torch.tensor(segment[:-1], dtype=torch.float32)
        
        # Target features: All time steps except the first
        target_features = torch.tensor(segment[1:], dtype=torch.float32)

        return input_features, target_features

def generate_audio_datasets(args):
    # Convert list of segments into a dataset
    dataset = AudioSegmentDataset(args)
    mfcc_feature_dim = dataset.get_feature_dim()

    # Split the dataset into training and validation
    validation_size = int(len(dataset) * 0.25)
    training_size = len(dataset) - validation_size
    train_dataset, validation_dataset = random_split(dataset, [training_size, validation_size])

    # Create DataLoaders
    # Note: num_workers=4 and/or pin_memory=True do not improve training throughput
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    return dataset, train_loader, val_loader, mfcc_feature_dim


################################################################################
# Training Loop

import torch.optim as optim
from tqdm.auto import tqdm

def train(dataset, model, train_loader, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    if args.mgpu:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    if args.compile:
        model = torch.compile(model)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Wrap the epoch range with tqdm
    epochs_tqdm = tqdm(range(args.epochs), desc='Overall Progress', leave=True)

    for epoch in epochs_tqdm:
        model.train()
        optimizer.zero_grad()

        sum_train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # Reset gradients for each batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            sum_train_loss += loss.item()

        avg_train_loss = sum_train_loss / len(train_loader)

        model.eval()

        sum_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                sum_val_loss += loss.item()

        avg_val_loss = sum_val_loss / len(val_loader)

        # Update tqdm description for epochs with average loss
        epochs_tqdm.set_description(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        epochs_tqdm.refresh()  # to show immediately the update

        if epoch % 10 == 9:
            print(f"\n")

    print(f"\nFinal Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}\n")


################################################################################
# Entrypoint

import argparse
import random

def seed_random(seed):
    if seed == 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    seed_random(args.seed)

    dataset, train_loader, val_loader, input_dim = generate_audio_datasets(args)

    # RNN performs much better
    #model = AudioRNN(dim=input_dim, d_hidden=input_dim*4)
    #model = AudioMLP(dim=input_dim, d_hidden=input_dim*4)
    model = SpectralSSM(d_in=input_dim, d_hidden=input_dim, d_out=input_dim, L=args.segment_length, num_layers=2)

    print(f"Model parameters: {count_parameters(model)}")

    train(dataset, model, train_loader, val_loader, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an RNN on audio data for next-sequence prediction.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='Size of RNN hidden state.')
    parser.add_argument('--segment_length', type=int, default=1024, help='Input segment size.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for randomization of data loader')
    parser.add_argument('--dir', type=str, default="./data", help='Directory to scan for audio files')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile')
    parser.add_argument('--mgpu', action='store_true', help='Enable multi-GPU training')
    args = parser.parse_args()

    main(args)
