# search/wrapper.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search import Search  # This will correctly import from search.py now

class MLSearchWrapper:
    """Wrapper around the existing search algorithm to use ML evaluation"""
    
    def __init__(self, position, ml_evaluator, max_depth=4):
        """
        Initialize the search wrapper.
        
        Args:
            position: The chess position to search from
            ml_evaluator: Our ML-based evaluator
            max_depth: Maximum search depth
        """
        self.position = position
        self.ml_evaluator = ml_evaluator
        
        # Create the original search object but replace its evaluator
        self.search = Search(position)
        self.search.eval = ml_evaluator
        
    def find_best_move(self, max_depth=None, time_limit=None):
        """Find the best move using the existing search with our ML evaluator"""
        # Check the implementation of iter_search in search.py to ensure parameters are passed correctly
        if max_depth is not None and time_limit is None:
            # If only max_depth is provided, use a very large time limit
            time_limit = 9999  # A very large number to effectively have no time limit
        elif time_limit is not None and max_depth is None:
            # If only time_limit is provided, use a large max_depth
            max_depth = 100  # A large depth that won't practically be reached
        elif max_depth is None and time_limit is None:
            # Default values if nothing is provided
            max_depth = 4
            time_limit = 5.0
            
        return self.search.iter_search(max_depth=max_depth, time_limit=time_limit)