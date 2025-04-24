# ml/evaluator.py
import torch
import numpy as np
from ml.features import position_to_tensor, add_auxiliary_planes
from ml.model import ChessPositionCNN

class MLEvaluator:
    """ML-based position evaluator using a CNN model"""
    
    def __init__(self, model_path=None, use_traditional_fallback=True):
        """
        Initialize the ML evaluator.
        
        Args:
            model_path: Path to the trained model weights file (if None, uses traditional eval)
            use_traditional_fallback: Whether to use traditional evaluation as a fallback
        """
        self.model = ChessPositionCNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.traditional_evaluator = None
        if use_traditional_fallback:
            # Import here to avoid circular imports
            from evaluate import Evaluate
            self.traditional_evaluator = Evaluate()
        
        self.use_ml = False
        if model_path:
            try:
                self.model.load(model_path)
                self.model.to(self.device)
                self.model.eval()  # Set to evaluation mode
                self.use_ml = True
            except Exception as e:
                print(f"Could not load model from {model_path}: {e}. Using traditional evaluation.")
                self.use_ml = False
            
        # Cache for position evaluations to avoid re-computing
        self.eval_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
            
    def clear_cache(self):
        """Clear the evaluation cache"""
        self.eval_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def evaluate(self, position):
        """
        Evaluate the position using the ML model or fallback to traditional.
        
        Args:
            position: Position object representing the current board state
            
        Returns:
            float: Evaluation score from current player's perspective
        """
        # Check cache first
        if position.zobrist in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[position.zobrist]
            
        self.cache_misses += 1
        
        if self.use_ml:
            # Convert position to tensor format for the CNN
            tensor = position_to_tensor(position)
            tensor = add_auxiliary_planes(tensor, position)
            tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Run through model
            with torch.no_grad():
                _, value = self.model(tensor)
                # Convert from [-1, 1] to centipawns range
                score = value.item() * 100
                
            # Add tempo bonus (similar to traditional eval)
            score += 28  # Same tempo bonus as in traditional evaluator
        elif self.traditional_evaluator:
            # Fallback to traditional evaluation
            score = self.traditional_evaluator.evaluate(position)
        else:
            # Simple material-only evaluation if no other options
            score = self._simple_material_eval(position)
            
        # Cache the result
        self.eval_cache[position.zobrist] = score
            
        return score
        
    def _simple_material_eval(self, position):
        """Simple material-only evaluation, used as last resort"""
        # Piece values (in centipawns): pawn=100, knight=320, bishop=330, rook=500, queen=900
        piece_values = [0, 100, 320, 330, 500, 900, 0]  # Index by piece type
        
        score = 0
        for sq in range(64):
            piece = position.squares[sq]
            if piece:
                piece_type = piece & 7
                piece_color = piece >> 3
                
                if piece_color == 0:  # White
                    score += piece_values[piece_type]
                else:  # Black
                    score -= piece_values[piece_type]
                    
        # Return from current player's perspective
        return score if position.colour == 0 else -score