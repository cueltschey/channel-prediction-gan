import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import ast
from torch.utils.data import DataLoader, TensorDataset

# Load CSV Data
def parse_array(column):
    return column.apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)

features = ['center_freq', 'dist', 'h_dist', 'v_dist', 'avgPower', 'avgSnr',
            'freq_offset', 'avg_pl', 'aod_theta', 'aoa_theta', 'aoa_phi',
            'pitch', 'yaw', 'roll', 'vel_x', 'vel_y', 'vel_z', 'speed', 'avg_pl_rolling', 'avg_pl_ewma']

df = pd.read_csv("dataset/2023-12-15_15_41-results.csv", usecols=features)

# Handle NaNs
df.fillna(0, inplace=True)

# Normalize the data
def normalize_data(df):
    return (df - df.min()) / (df.max() - df.min()).replace(0, 1)  # Avoid division by zero
df = normalize_data(df)

# Assume binary labels (valid=1, invalid=0) - Replace with actual labels
labels = np.random.randint(0, 2, len(df))  # Placeholder labels

# Convert to PyTorch tensors
X = torch.tensor(df.values, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define Classifier Model
class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Training Function
def train_classifier(model, dataloader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    model.train()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.clamp(outputs, min=1e-7, max=1-1e-7)  # Ensure outputs are within (0,1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Train the Classifier
classifier = Classifier(input_size=len(features)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
train_classifier(classifier, dataloader)

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
            
            # Train Discriminator
            real_labels = torch.ones(batch_size, 1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            fake_labels = torch.zeros(batch_size, 1).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            optimizer_D.zero_grad()
            outputs = discriminator(real_data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
            loss_real = criterion(outputs, real_labels)
            
            z = torch.randn(batch_size, len(features)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            fake_data = generator(z)
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
