"""
Implementation of the IDS (Iterative Deepening Search) agent for the KekeAI game.
Uses the Iterative Deepening Search algorithm to find the optimal solution.
"""

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, interpret_rules
from typing import List, Set, Tuple, Optional
from tqdm import trange


class IDSAgent(BaseAgent):
    """
    Iterative Deepening Search implementation.
    """
    
    def __init__(self):
        self.visited_states: Set[str] = set()
        self.current_depth_limit: int = 0
    
    def get_state_hash(self, state: GameState) -> str:
        """
        Creates a unique hash for the game state including object positions and rules.
        """
        try:
            # Serialize the object map
            obj_map_str = ""
            for row in state.obj_map:
                for cell in row:
                    if cell is None:
                        obj_map_str += "."
                    elif hasattr(cell, 'name'):
                        obj_map_str += cell.name
                    else:
                        obj_map_str += str(cell)
            
            # Serialize the background map
            back_map_str = ""
            for row in state.back_map:
                for cell in row:
                    if cell is None:
                        back_map_str += "."
                    elif hasattr(cell, 'name'):
                        back_map_str += cell.name
                    else:
                        back_map_str += str(cell)
            
            # Add the rules (sorted for consistency)
            rules_str = ''.join(sorted(state.rules)) if state.rules else ""
            
            return f"{obj_map_str}|{back_map_str}|{rules_str}"
        except Exception as e:
            print(f"Error creating state hash: {e}")
            # Fallback to a simple hash
            return str(hash(str(state)))
    
    
    def depth_limited_search(self, state: GameState, depth_limit: int, current_depth: int = 0, 
                           actions: List[Direction] = None, path_states: Set[str] = None) -> Optional[List[Direction]]:
        """
        Performs depth-limited search up to the specified depth limit.
        
        :param state: Current game state
        :param depth_limit: Maximum depth to search
        :param current_depth: Current depth in the search tree
        :param actions: Actions taken so far
        :param path_states: States visited in current path (for cycle detection)
        :return: List of actions if solution found, None otherwise
        """
        if actions is None:
            actions = []
        if path_states is None:
            path_states = set()
        
        # Ensure rules are interpreted for current state
        try:
            interpret_rules(state)
        except Exception as e:
            # If rule interpretation fails, this state is invalid
            return None
        
        # Get current state hash
        state_hash = self.get_state_hash(state)
        
        # Check for cycles in current path
        if state_hash in path_states:
            return None
        
        # Check if we've won
        if check_win(state):
            return actions
        
        # Check depth limit
        if current_depth >= depth_limit:
            return None
        
        # Add current state to path
        path_states.add(state_hash)
        
        # Explore all possible actions
        for action in [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]:
            try:
                # Create the new state
                next_state = advance_game_state(action, state.copy())
                
                # Skip if state generation failed
                if next_state is None:
                    continue
                
                # Recursive call with increased depth
                result = self.depth_limited_search(
                    next_state, 
                    depth_limit, 
                    current_depth + 1, 
                    actions + [action],
                    path_states.copy()
                )
                
                if result is not None:
                    return result
                    
            except Exception as e:
                # Handle errors in state generation
                print(f"Error in depth_limited_search at depth {current_depth}: {e}")
                continue
        
        return None
    
    def search(self, initial_state: GameState, iterations: int = 1000) -> List[Direction]:
        """
        Implements the Iterative Deepening Search algorithm.
        
        :param initial_state: The initial game state
        :param iterations: Maximum number of iterations (converted to depth limit)
        :return: List of actions leading to the solution
        """
        # Convert iterations to a reasonable depth limit
        # Since IDS explores exponentially, we use a smaller depth limit
        max_depth = min(50, max(10, iterations // 100))
        
        # Interpret the initial rules
        try:
            interpret_rules(initial_state)
        except Exception as e:
            print(f"Error interpreting initial rules: {e}")
            return []
        
        # Check if initial state is already winning
        if check_win(initial_state):
            print("Initial state is already winning!")
            return []
        
        # Debug: Print initial state info
        print(f"Initial state - Players: {len(initial_state.players)}, Winnables: {len(initial_state.winnables)}")
        print(f"Initial rules: {initial_state.rules}")
        
        nodes_expanded = 0
        
        # Iteratively increase depth limit
        for depth_limit in trange(1, max_depth + 1, desc="IDS Search"):
            print(f"Searching at depth limit: {depth_limit}")
            
            # Reset visited states for each depth iteration
            self.visited_states = set()
            self.current_depth_limit = depth_limit
            
            # Perform depth-limited search
            result = self.depth_limited_search(initial_state, depth_limit)
            
            if result is not None:
                print(f"Solution found at depth {len(result)} with depth limit {depth_limit}")
                print(f"Solution actions: {[action.name for action in result]}")
                return result
        
        print(f"No solution found within depth limit of {max_depth}")
        return []
    
    def search_with_max_depth(self, initial_state: GameState, max_depth: int = 50) -> List[Direction]:
        """
        Alternative search method that directly uses max_depth parameter.
        
        :param initial_state: The initial game state
        :param max_depth: Maximum depth to search
        :return: List of actions leading to the solution
        """
        return self.search(initial_state, max_depth * 100)