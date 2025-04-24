# ml/features.py
import numpy as np
from consts import (W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
                   B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
                   WHITE, BLACK)

def position_to_tensor(position):
    """
    Convert a Position object to a tensor representation suitable for CNN input.
    
    Args:
        position: A Position object from the chess engine
        
    Returns:
        numpy.ndarray: An 8×8×12 tensor representing the board position
    """
    # Initialize empty tensor
    tensor = np.zeros((8, 8, 12), dtype=np.float32)
    
    # Map each piece type to its plane index
    piece_to_plane = {
        W_PAWN: 0, W_KNIGHT: 1, W_BISHOP: 2, W_ROOK: 3, W_QUEEN: 4, W_KING: 5,
        B_PAWN: 6, B_KNIGHT: 7, B_BISHOP: 8, B_ROOK: 9, B_QUEEN: 10, B_KING: 11
    }
    
    # Fill tensor based on position.squares
    for sq in range(64):
        piece = position.squares[sq]
        if piece:
            # Convert square index to 2D coordinates
            rank = sq // 8
            file = sq % 8
            plane = piece_to_plane[piece]
            tensor[rank, file, plane] = 1.0
            
    return tensor

def add_auxiliary_planes(tensor, position):
    """
    Add auxiliary planes for special state information (castling, en passant, etc.)
    
    Args:
        tensor: Base tensor with piece positions
        position: Position object containing game state
        
    Returns:
        numpy.ndarray: Tensor with additional planes
    """
    # Create new tensor with additional planes
    full_tensor = np.zeros((8, 8, 16), dtype=np.float32)
    
    # Copy the original piece planes
    full_tensor[:, :, :12] = tensor
    
    # Add castling rights plane
    castling_plane = np.zeros((8, 8), dtype=np.float32)
    if position.castling_rights:
        # Set values in appropriate squares for castling rights
        if position.castling_rights & (1 << 0):  # W_KINGSIDE
            castling_plane[0, 7] = 1.0
        if position.castling_rights & (1 << 1):  # W_QUEENSIDE
            castling_plane[0, 0] = 1.0
        if position.castling_rights & (1 << 2):  # B_KINGSIDE
            castling_plane[7, 7] = 1.0
        if position.castling_rights & (1 << 3):  # B_QUEENSIDE
            castling_plane[7, 0] = 1.0
    full_tensor[:, :, 12] = castling_plane
    
    # Add en passant plane
    en_passant_plane = np.zeros((8, 8), dtype=np.float32)
    if position.ep_square is not None:
        rank = position.ep_square // 8
        file = position.ep_square % 8
        en_passant_plane[rank, file] = 1.0
    full_tensor[:, :, 13] = en_passant_plane
    
    # Add side to move plane (all 1s if it's white to move, all 0s if black)
    side_to_move_plane = np.ones((8, 8), dtype=np.float32) if position.colour == WHITE else np.zeros((8, 8), dtype=np.float32)
    full_tensor[:, :, 14] = side_to_move_plane
    
    # Add move counter plane (normalized)
    move_count_plane = np.ones((8, 8), dtype=np.float32) * (position.fullmove_number / 100.0)
    full_tensor[:, :, 15] = move_count_plane
    
    return full_tensor