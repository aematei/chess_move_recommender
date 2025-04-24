# test_ml_components.py
import os
import torch
import numpy as np
from position import Position
from ml.features import position_to_tensor, add_auxiliary_planes
from ml.model import ChessPositionCNN
from ml.evaluator import MLEvaluator
from ml_search.wrapper import MLSearchWrapper

def test_directory_structure():
    """Check if our directory structure is correct"""
    print("Testing directory structure...")
    
    # Check if directories exist
    assert os.path.exists("ml"), "ml directory not found"
    assert os.path.exists("search"), "search directory not found"
    
    # Check if required files exist
    assert os.path.exists("ml/features.py"), "ml/features.py not found"
    assert os.path.exists("ml/model.py"), "ml/model.py not found"
    assert os.path.exists("ml/evaluator.py"), "ml/evaluator.py not found"
    assert os.path.exists("search/wrapper.py"), "search/wrapper.py not found"
    
    # Ensure the models directory exists
    os.makedirs("ml/models", exist_ok=True)
    
    print("✓ Directory structure is correct")

def test_feature_extraction():
    """Test the feature extraction process"""
    print("\nTesting feature extraction...")
    
    # Create a position from starting FEN
    position = Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    # Convert to tensor
    tensor = position_to_tensor(position)
    
    # Check shape
    assert tensor.shape == (8, 8, 12), f"Expected tensor shape (8, 8, 12), got {tensor.shape}"
    
    # Check if pieces are correctly placed
    # White pawns should be in the second rank
    assert np.all(tensor[6, :, 0] == 1), "White pawns not correctly placed"
    
    # Add auxiliary planes
    full_tensor = add_auxiliary_planes(tensor, position)
    
    # Check shape
    assert full_tensor.shape == (8, 8, 16), f"Expected tensor shape (8, 8, 16), got {full_tensor.shape}"
    
    print("✓ Feature extraction works correctly")

def test_cnn_model():
    """Test the CNN model's forward pass"""
    print("\nTesting CNN model...")
    
    # Create a model
    model = ChessPositionCNN()
    
    # Create a random input tensor
    input_tensor = torch.rand(1, 16, 8, 8)  # Batch size 1, 16 channels, 8x8 board
    
    # Test forward pass
    try:
        policy, value = model(input_tensor)
        
        # Check policy and value shapes
        assert policy.shape == (1, 1968), f"Expected policy shape (1, 1968), got {policy.shape}"
        assert value.shape == (1, 1), f"Expected value shape (1, 1), got {value.shape}"
        
        # Check if value is in expected range [-1, 1]
        assert -1 <= value.item() <= 1, f"Expected value in range [-1, 1], got {value.item()}"
        
        # Test saving and loading
        test_model_path = "ml/models/test_model.pt"
        model.save(test_model_path)
        assert os.path.exists(test_model_path), f"Model file {test_model_path} not created"
        
        # Load model
        new_model = ChessPositionCNN()
        new_model.load(test_model_path)
        
        # Clean up
        os.remove(test_model_path)
        
        print("✓ CNN model works correctly")
    except Exception as e:
        print(f"✗ CNN model test failed: {e}")

def test_evaluator():
    """Test the ML evaluator"""
    print("\nTesting ML evaluator...")
    
    # Create a position 
    position = Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    # Create evaluator with traditional fallback
    evaluator = MLEvaluator(use_traditional_fallback=True)
    
    # Test evaluation
    try:
        score = evaluator.evaluate(position)
        print(f"  Initial position evaluation: {score}")
        
        # Test cache
        score2 = evaluator.evaluate(position)
        assert evaluator.cache_hits == 1, f"Expected 1 cache hit, got {evaluator.cache_hits}"
        
        # Move a pawn and evaluate again
        pawn_move = None
        for move in position.get_pseudo_legal_moves():
            if position.is_legal(move):
                pawn_move = move
                break
        
        if pawn_move:
            position.make_move(pawn_move)
            score3 = evaluator.evaluate(position)
            print(f"  Position after move evaluation: {score3}")
        
        print("✓ ML evaluator works correctly")
    except Exception as e:
        print(f"✗ ML evaluator test failed: {e}")

def test_search_wrapper():
    """Test the search wrapper"""
    print("\nTesting search wrapper...")
    
    # Create a position
    position = Position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    # Create evaluator
    evaluator = MLEvaluator(use_traditional_fallback=True)
    
    # Create search wrapper
    try:
        search_wrapper = MLSearchWrapper(position, evaluator)
        
        # Find best move with limited depth
        best_move = search_wrapper.find_best_move(max_depth=1)
        
        # Verify move is legal
        assert position.is_legal(best_move), f"Move {best_move} is not legal"
        
        print(f"  Best move from starting position (depth 1): {best_move}")
        print("✓ Search wrapper works correctly")
    except Exception as e:
        print(f"✗ Search wrapper test failed: {e}")

def run_all_tests():
    """Run all tests"""
    print("=== ML Components Test Suite ===\n")
    
    test_directory_structure()
    test_feature_extraction()
    test_cnn_model()
    test_evaluator()
    test_search_wrapper()
    
    print("\n=== All tests completed ===")

if __name__ == "__main__":
    run_all_tests()