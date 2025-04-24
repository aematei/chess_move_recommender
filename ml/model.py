# ml/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessConvBlock(nn.Module):
    """A convolutional block with batch normalization and residual connection"""
    def __init__(self, channels):
        super(ChessConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class ChessPositionCNN(nn.Module):
    """CNN for evaluating chess positions"""
    def __init__(self, input_channels=16, num_filters=128, num_blocks=10):
        super(ChessPositionCNN, self).__init__()
        
        # Initial convolution layer
        self.conv_input = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # Residual blocks
        self.blocks = nn.ModuleList([ChessConvBlock(num_filters) for _ in range(num_blocks)])
        
        # Policy head (not used for evaluation, but could be useful for move ordering)
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 1968)  # Max possible moves < 1968
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        # Initial layers
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def get_evaluation(self, x):
        """Return only the evaluation score (for use in search)"""
        _, value = self.forward(x)
        return value
        
    def save(self, path):
        """Save model weights to a file"""
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """Load model weights from a file"""
        self.load_state_dict(torch.load(path))