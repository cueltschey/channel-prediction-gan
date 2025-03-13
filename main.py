import torch
import os
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import ast
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Load CSV Data
def parse_array(column):
    return column.apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)

def load_multiple_csvs(directory, features):
    dataframes = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, file), usecols=features)
            dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

# Define feature columns
features = ['center_freq', 'dist', 'h_dist', 'v_dist', 'avgPower', 'avgSnr',
            'freq_offset', 'avg_pl', 'aod_theta', 'aoa_theta', 'aoa_phi',
            'pitch', 'yaw', 'roll', 'vel_x', 'vel_y', 'vel_z', 'speed', 'avg_pl_rolling', 'avg_pl_ewma']

# Load all CSVs in the dataset directory
df = load_multiple_csvs("dataset/", features)

# Handle NaNs
df.fillna(0, inplace=True)

# Normalize the data
def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min()).replace(0, 1)  # Avoid division by zero
df = normalize_data(df)

labels = np.ones(len(df))

# Convert to PyTorch tensors
X = torch.tensor(df.values, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define GAN Generator and Discriminator
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# GAN Training
def train_gan(generator, discriminator, dataloader, epochs=100):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for real_data, _ in dataloader:
            batch_size = real_data.size(0)

            real_labels = torch.ones(batch_size, 1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            fake_labels = torch.zeros(batch_size, 1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # Train discriminator against real data
            optimizer_D.zero_grad()
            outputs = discriminator(real_data.to(DEVICE))
            loss_real = criterion(outputs, real_labels)

            # Random noise
            z = torch.randn(batch_size, len(features)).to(DEVICE)
            # Generate fake data
            fake_data = generator(z)

            # Discriminate fake data
            outputs = discriminator(fake_data.detach())
            loss_fake = criterion(outputs, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            outputs = discriminator(fake_data)
            loss_G = criterion(outputs, real_labels)
            loss_G.backward()
            optimizer_G.step()
        
        print(f"Epoch {epoch+1}, D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}")

# Initialize and train GAN
generator = Generator(input_size=len(features), output_size=len(features)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
discriminator = Discriminator(input_size=len(features)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
train_gan(generator, discriminator, dataloader)
