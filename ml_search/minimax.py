# search/minimax.py
import time
import torch
import numpy as np
from ml.features import position_to_tensor, add_auxiliary_planes

class MinimaxSearch:
    """Minimax search with alpha-beta pruning and iterative deepening"""
    
    def __init__(self, position, evaluator, max_depth=4):
        """
        Initialize the search algorithm.
        
        Args:
            position: The chess position to search from
            evaluator: The position evaluator (ML model)
            max_depth: Maximum search depth
        """
        self.position = position
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.best_move = None
        self.nodes_searched = 0
        self.tt = {}  # Transposition table
        self.pv_table = {}  # Principal variation table
        self.start_time = 0
        
    def iterative_deepening(self, max_time=5.0):
        """
        Perform iterative deepening search up to max_depth.
        
        Args:
            max_time: Maximum time to search in seconds
            
        Returns:
            int: The best move found
        """
        self.start_time = time.time()
        self.best_move = None
        self.nodes_searched = 0
        self.tt = {}
        self.pv_table = {}
        
        # Get a list of all legal moves
        legal_moves = []
        for move in self.position.get_pseudo_legal_moves():
            if self.position.is_legal(move):
                legal_moves.append(move)
                
        # If only one legal move, return it immediately
        if len(legal_moves) == 1:
            return legal_moves[0]
            
        # Try each depth until max_depth or time limit
        for depth in range(1, self.max_depth + 1):
            # Run alpha-beta search
            score = self.alpha_beta(depth, float('-inf'), float('inf'), True)
            
            elapsed = time.time() - self.start_time
            
            # If out of time, break
            if elapsed >= max_time and depth > 1:
                break
                
        return self.best_move
        
    def alpha_beta(self, depth, alpha, beta, maximizing_player):
        """
        Alpha-beta pruning implementation.
        
        Args:
            depth: Current search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing_player: Whether the current player is maximizing
            
        Returns:
            float: Evaluation score of the position
        """
        self.nodes_searched += 1
        
        # Check time limit
        if self.nodes_searched % 1000 == 0:
            if time.time() - self.start_time > 5.0:
                return 0  # Return early if time limit exceeded
        
        # Check for terminal node
        if depth == 0 or self.position.is_game_over():
            # Use CNN evaluator for leaf nodes
            return self.evaluator.evaluate(self.position)
            
        # Generate legal moves
        legal_moves = []
        for move in self.position.get_pseudo_legal_moves():
            if self.position.is_legal(move):
                legal_moves.append(move)
                
        # If no legal moves, it's checkmate or stalemate
        if not legal_moves:
            if self.position.is_in_check():
                return float('-inf')  # Checkmate
            else:
                return 0  # Stalemate
                
        # Sort moves based on simple heuristics (captures first)
        # This will later be replaced with more sophisticated move ordering
        def move_score(move):
            score = 0
            if self.position.squares[move & 0x3F]:  # Capturing move
                score += 10
            if move == self.pv_table.get(self.position.zobrist, None):  # PV move
                score += 1000
            return score
            
        legal_moves.sort(key=move_score, reverse=True)
                
        if maximizing_player:
            max_score = float('-inf')
            for move in legal_moves:
                self.position.make_move(move)
                score = self.alpha_beta(depth - 1, alpha, beta, False)
                self.position.undo_move()
                
                if score > max_score:
                    max_score = score
                    if depth == self.max_depth:
                        self.best_move = move
                        
                # Store in PV table
                self.pv_table[self.position.zobrist] = move
                        
                alpha = max(alpha, max_score)
                if beta <= alpha:
                    break  # Beta cutoff
                    
            return max_score
        else:
            min_score = float('inf')
            for move in legal_moves:
                self.position.make_move(move)
                score = self.alpha_beta(depth - 1, alpha, beta, True)
                self.position.undo_move()
                
                min_score = min(min_score, score)
                beta = min(beta, min_score)
                if beta <= alpha:
                    break  # Alpha cutoff
                    
            return min_score