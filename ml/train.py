# ml/train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import chess
import chess.pgn
import random
import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Import our model and features
from ml.model import ChessPositionCNN
from ml.features import position_to_tensor, add_auxiliary_planes
from position import Position
from evaluate import Evaluate

class ChessDataset(Dataset):
    """Dataset for chess positions"""
    def __init__(self, positions, labels):
        self.positions = positions
        self.labels = labels
        
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return self.positions[idx], self.labels[idx]

def create_training_data(num_positions=10000, pgn_file=None):
    """
    Create training data from chess games.
    
    Args:
        num_positions: Number of positions to extract
        pgn_file: Path to PGN file with chess games
        
    Returns:
        tuple: (positions, labels) arrays
    """
    print(f"Creating {num_positions} training positions...")
    positions = []
    labels = []
    
    # Create traditional evaluator for labeling
    evaluator = Evaluate()
    
    # If no PGN file, generate random positions
    if pgn_file is None or not os.path.exists(pgn_file):
        # Generate random positions by playing random moves from the starting position
        for _ in tqdm.tqdm(range(num_positions)):
            board = chess.Board()
            
            # Make 10-40 random moves
            num_moves = random.randint(10, 40)
            
            try:
                for _ in range(num_moves):
                    if board.is_game_over():
                        break
                    moves = list(board.legal_moves)
                    if moves:
                        move = random.choice(moves)
                        board.push(move)
                
                # Convert to our Position class
                fen = board.fen()
                position = Position(fen)
                
                # Get evaluation using traditional evaluator
                score = evaluator.evaluate(position)
                
                # Normalize score to [-1, 1] range
                normalized_score = np.tanh(score / 100.0)
                
                # Convert position to tensor
                tensor = position_to_tensor(position)
                tensor = add_auxiliary_planes(tensor, position)
                
                # Convert from HWC to CHW format
                tensor = np.transpose(tensor, (2, 0, 1))
                
                positions.append(tensor)
                labels.append(normalized_score)
            except Exception as e:
                print(f"Error processing position: {e}")
                continue
    else:
        # Load positions from PGN file
        with open(pgn_file) as pgn:
            position_count = 0
            
            while position_count < num_positions:
                # Load the next game
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                
                try:
                    # Play through the game and sample positions
                    board = game.board()
                    for move in game.mainline_moves():
                        board.push(move)
                        
                        # Only sample some positions to avoid correlation
                        if random.random() < 0.1:  # 10% sampling rate
                            fen = board.fen()
                            position = Position(fen)
                            
                            # Get evaluation using traditional evaluator
                            score = evaluator.evaluate(position)
                            
                            # Normalize score to [-1, 1] range
                            normalized_score = np.tanh(score / 100.0)
                            
                            # Convert position to tensor
                            tensor = position_to_tensor(position)
                            tensor = add_auxiliary_planes(tensor, position)
                            
                            # Convert from HWC to CHW format
                            tensor = np.transpose(tensor, (2, 0, 1))
                            
                            positions.append(tensor)
                            labels.append(normalized_score)
                            
                            position_count += 1
                            if position_count >= num_positions:
                                break
                except Exception as e:
                    print(f"Error processing game: {e}")
                    continue
    
    # Convert to numpy arrays
    positions = np.array(positions, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32).reshape(-1, 1)
    
    print(f"Created dataset with {len(positions)} positions")
    return positions, labels

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """
    Train the CNN model.
    
    Args:
        model: The CNN model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        tuple: (trained_model, train_losses, val_losses)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track training and validation loss
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Training phase
        for inputs, targets in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            _, value = model(inputs)
            loss = criterion(value, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                _, value = model(inputs)
                loss = criterion(value, targets)
                val_loss += loss.item() * inputs.size(0)
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return model, train_losses, val_losses

def plot_training_history(train_losses, val_losses, save_path):
    """Plot and save training history"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Create directories if they don't exist
    os.makedirs("ml/models", exist_ok=True)
    os.makedirs("ml/plots", exist_ok=True)
    
    # Parameters
    num_positions = 10000  # Number of positions for training
    batch_size = 64
    num_epochs = 20
    validation_split = 0.2
    
    # Create or load training data
    pgn_file = None  # Set to your PGN file path if available
    positions, labels = create_training_data(num_positions, pgn_file)
    
    # Split into training and validation sets
    split_idx = int(len(positions) * (1 - validation_split))
    train_positions, val_positions = positions[:split_idx], positions[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # Create datasets and data loaders
    train_dataset = ChessDataset(train_positions, train_labels)
    val_dataset = ChessDataset(val_positions, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create and train the model
    model = ChessPositionCNN()
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs)
    
    # Generate timestamp for saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the trained model
    model_path = f"ml/models/chess_model.pt"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Plot and save training history
    plot_path = f"ml/plots/training_history_{timestamp}.png"
    plot_training_history(train_losses, val_losses, plot_path)
    print(f"Training plot saved to {plot_path}")

if __name__ == "__main__":
    main()