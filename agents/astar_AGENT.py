"""
Implementation of the A* agent for the KekeAI game.
Uses the A* search algorithm with a heuristic function to find the optimal solution.
"""

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, interpret_rules
from typing import List, Set, Tuple, Dict
import heapq
from tqdm import trange


class ASTARAgent(BaseAgent):
    """
    A* Search implementation with heuristic function.
    """
    
    def __init__(self):
        self.visited_states: Set[str] = set()
        self.g_costs: Dict[str, int] = {}
    
    def get_state_hash(self, state: GameState) -> str:
        """
        Creates a unique hash for the game state including object positions and rules.
        """
        # Serialize the object map
        obj_map_str = ""
        for row in state.obj_map:
            for cell in row:
                if hasattr(cell, 'name'):
                    obj_map_str += cell.name
                else:
                    obj_map_str += str(cell)
        
        # Serialize the background map
        back_map_str = ""
        for row in state.back_map:
            for cell in row:
                if hasattr(cell, 'name'):
                    back_map_str += cell.name
                else:
                    back_map_str += str(cell)
        
        # Add the rules
        rules_str = ''.join(sorted(state.rules))
        
        return f"{obj_map_str}|{back_map_str}|{rules_str}"
    
    def heuristic(self, state: GameState) -> float:
        """
        Improved heuristic function that considers multiple strategies.
        """
        if check_win(state):
            return 0.0
        
        # If there are no players or winnable objects, state is impossible
        if not state.players or not state.winnables:
            return 1000.0
        
        # Calculate the minimum distance from any player to any winnable object
        min_distance = float('inf')
        
        for player in state.players:
            player_pos = (player.x, player.y)
            for winnable in state.winnables:
                winnable_pos = (winnable.x, winnable.y)
                # Manhattan distance
                distance = abs(player_pos[0] - winnable_pos[0]) + abs(player_pos[1] - winnable_pos[1])
                min_distance = min(min_distance, distance)
        
        # If distance is infinite, heavily penalize
        if min_distance == float('inf'):
            return 1000.0
        
        # Bonus for having more players and winnable objects
        bonus = 0
        if len(state.players) > 1:
            bonus -= 1
        if len(state.winnables) > 1:
            bonus -= 1
            
        return max(0, min_distance + bonus)

    def search(self, initial_state: GameState, iterations: int = 1000) -> List[Direction]:
        """
        Implements the A* algorithm to find the optimal path.
        
        :param initial_state: The initial game state
        :param iterations: Maximum number of iterations
        :return: List of actions leading to the solution
        """
        # Interpret the initial rules
        interpret_rules(initial_state)
        
        # Priority queue: (f_cost, g_cost, state_hash, state, action_history)
        priority_queue = []
        
        # Calculate initial heuristic cost
        initial_state_hash = self.get_state_hash(initial_state)
        initial_h_cost = self.heuristic(initial_state)
        initial_f_cost = 0 + initial_h_cost
        
        # Add the initial state
        heapq.heappush(priority_queue, (initial_f_cost, 0, initial_state_hash, initial_state, []))
        
        # Initialize data structures
        self.visited_states = set()
        self.g_costs = {initial_state_hash: 0}
        
        # Counter to avoid infinite loops
        nodes_expanded = 0
        
        for i in trange(iterations, desc="A* Search"):
            if not priority_queue:
                print("Priority queue empty - no solution found")
                break
            
            # Pop the node with the lowest f cost
            f_cost, g_cost, state_hash, current_state, actions = heapq.heappop(priority_queue)
            
            # Skip if already visited with a better cost
            if state_hash in self.visited_states:
                continue
                
            # Mark as visited
            self.visited_states.add(state_hash)
            nodes_expanded += 1
            
            # Check if we've won
            if check_win(current_state):
                print(f"Solution found in {len(actions)} moves after expanding {nodes_expanded} nodes")
                return actions
            
            # Explore all possible actions
            for action in [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]:
                try:
                    # Create the new state
                    next_state = advance_game_state(action, current_state.copy())
                    interpret_rules(next_state)  # Ensure rules are updated
                    next_state_hash = self.get_state_hash(next_state)
                    
                    # Skip if already visited
                    if next_state_hash in self.visited_states:
                        continue
                    
                    # Compute costs
                    tentative_g_cost = g_cost + 1
                    
                    # If we already have a better path, skip
                    if next_state_hash in self.g_costs and tentative_g_cost >= self.g_costs[next_state_hash]:
                        continue
                    
                    # This is the best path so far
                    self.g_costs[next_state_hash] = tentative_g_cost
                    next_h_cost = self.heuristic(next_state)
                    next_f_cost = tentative_g_cost + next_h_cost
                    
                    # Add to priority queue
                    new_actions = actions + [action]
                    heapq.heappush(priority_queue, (next_f_cost, tentative_g_cost, next_state_hash, next_state, new_actions))
                    
                except Exception as e:
                    # Handle errors in state generation
                    continue
        
        print(f"No solution found after expanding {nodes_expanded} nodes")
        return []
