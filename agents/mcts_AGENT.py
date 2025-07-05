"""
MCTS (Monte Carlo Tree Search) agent implementation for the KekeAI game.
Uses the MCTS algorithm with Monte Carlo simulations to find the optimal solution.
"""

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, interpret_rules
from typing import List, Set, Tuple, Optional
import random
import math
from tqdm import trange
import copy


class MCTSNode:
    """
    Node for the MCTS tree.
    """
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None, action: Optional[Direction] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions = [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]
        self.state_hash = self._compute_state_hash()
    
    def _compute_state_hash(self) -> str:
        """Compute a unique hash for the state."""
        # Serialize the object map
        obj_map_str = ""
        for row in self.state.obj_map:
            for cell in row:
                if hasattr(cell, 'name'):
                    obj_map_str += cell.name
                else:
                    obj_map_str += str(cell)
        
        # Serialize the background map
        back_map_str = ""
        for row in self.state.back_map:
            for cell in row:
                if hasattr(cell, 'name'):
                    back_map_str += cell.name
                else:
                    back_map_str += str(cell)
        
        # Add the rules
        rules_str = ''.join(sorted(self.state.rules))
        
        return f"{obj_map_str}|{back_map_str}|{rules_str}"
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""
        return check_win(self.state)
    
    def uct_value(self, exploration_constant: float = 1.4) -> float:
        """Calculate the UCT (Upper Confidence Bound for Trees) value."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, exploration_constant: float = 1.4) -> 'MCTSNode':
        """Select the best child using UCT."""
        return max(self.children, key=lambda child: child.uct_value(exploration_constant))
    
    def select_untried_action(self) -> Direction:
        """Select an untried action."""
        return random.choice(self.untried_actions)


class MCTSAgent(BaseAgent):
    """
    MCTS (Monte Carlo Tree Search) implementation.
    """
    
    def __init__(self, exploration_constant: float = 1.4):
        self.exploration_constant = exploration_constant
        self.visited_states: Set[str] = set()
    
    def get_distance_to_goal(self, state: GameState) -> float:
        """Calculate the minimum distance from the player to the winning object."""
        if not state.players or not state.winnables:
            return float('inf')
        
        min_distance = float('inf')
        for player in state.players:
            player_pos = (player.x, player.y)
            for winnable in state.winnables:
                winnable_pos = (winnable.x, winnable.y)
                distance = abs(player_pos[0] - winnable_pos[0]) + abs(player_pos[1] - winnable_pos[1])
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: navigate the tree using UCT."""
        current = node
        
        while not current.is_terminal() and current.is_fully_expanded():
            if not current.children:
                break
            current = current.best_child(self.exploration_constant)
        
        return current
    
    def expand(self, node: MCTSNode) -> MCTSNode:
        """Expansion phase: add a new child."""
        if node.is_terminal():
            return node
        
        if not node.untried_actions:
            return node
        
        # Select an untried action
        action = node.select_untried_action()
        node.untried_actions.remove(action)
        
        try:
            # Create the new state
            new_state = advance_game_state(action, node.state.copy())
            interpret_rules(new_state)
            
            # Create the new child node
            child_node = MCTSNode(new_state, parent=node, action=action)
            node.children.append(child_node)
            
            return child_node
        except Exception as e:
            # If there's an error, return the current node
            return node
    
    def simulate(self, state: GameState, max_depth: int = 50) -> float:
        """
        Simulation phase: execute a random game with heuristics.
        """
        current_state = state.copy()
        visited_in_simulation: Set[str] = set()
        initial_distance = self.get_distance_to_goal(current_state)
        
        for step in range(max_depth):
            # Check if we've won
            if check_win(current_state):
                return 1.0
            
            # Avoid cycles in simulation
            state_hash = self._get_state_hash(current_state)
            if state_hash in visited_in_simulation:
                break
            visited_in_simulation.add(state_hash)
            
            # Choose an action with heuristics
            action = self._choose_action_with_heuristic(current_state)
            
            try:
                current_state = advance_game_state(action, current_state.copy())
                interpret_rules(current_state)
            except Exception as e:
                break
        
        # Calculate reward based on how close we got to the goal
        final_distance = self.get_distance_to_goal(current_state)
        
        if final_distance == float('inf'):
            return 0.0
        
        # Reward based on improvement
        if initial_distance == float('inf'):
            return 0.1
        
        improvement = max(0, initial_distance - final_distance)
        return min(1.0, improvement / max(1, initial_distance))
    
    def _get_state_hash(self, state: GameState) -> str:
        """Create a hash of the state."""
        obj_map_str = ""
        for row in state.obj_map:
            for cell in row:
                if hasattr(cell, 'name'):
                    obj_map_str += cell.name
                else:
                    obj_map_str += str(cell)
        return obj_map_str
    
    def _choose_action_with_heuristic(self, state: GameState) -> Direction:
        """Choose an action using a simple heuristic."""
        if not state.players or not state.winnables:
            return random.choice(list(Direction))
        
        player = state.players[0]
        target = state.winnables[0]
        
        # Calculate direction towards the goal
        dx = target.x - player.x
        dy = target.y - player.y
        
        # Choose action with bias towards the goal
        actions_weights = []
        
        for action in Direction:
            weight = 1.0
            
            if action == Direction.Right and dx > 0:
                weight = 3.0
            elif action == Direction.Left and dx < 0:
                weight = 3.0
            elif action == Direction.Down and dy > 0:
                weight = 3.0
            elif action == Direction.Up and dy < 0:
                weight = 3.0
            
            actions_weights.append((action, weight))
        
        # Weighted choice
        total_weight = sum(weight for _, weight in actions_weights)
        rand_val = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for action, weight in actions_weights:
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                return action
        
        return random.choice(list(Direction))
    
    def backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagation phase: update node values."""
        current = node
        while current is not None:
            current.visits += 1
            current.wins += reward
            current = current.parent
    
    def get_best_path(self, root: MCTSNode) -> List[Direction]:
        """Get the best path from the MCTS tree."""
        path = []
        current = root
        
        while current.children:
            # Choose the child with the most visits (most explored)
            best_child = max(current.children, key=lambda child: child.visits)
            
            if best_child.action:
                path.append(best_child.action)
            
            current = best_child
            
            # If we find a solution, stop
            if check_win(current.state):
                break
        
        return path
    
    def search(self, initial_state: GameState, iterations: int = 1000) -> List[Direction]:
        """
        Implement the MCTS algorithm to find the solution.
        
        :param initial_state: The initial game state
        :param iterations: Number of MCTS iterations
        :return: List of actions that lead to the solution
        """
        # Interpret initial rules
        interpret_rules(initial_state)
        
        # Create root node
        root = MCTSNode(initial_state)
        
        # Execute MCTS iterations
        for i in trange(iterations, desc="MCTS Search"):
            # 1. Selection
            selected_node = self.select(root)
            
            # 2. Expansion
            expanded_node = self.expand(selected_node)
            
            # 3. Simulation
            reward = self.simulate(expanded_node.state)
            
            # 4. Backpropagation
            self.backpropagate(expanded_node, reward)
            
            # Check if we found a solution
            if expanded_node.is_terminal():
                # Reconstruct the path to the solution
                path = []
                current = expanded_node
                while current.parent is not None:
                    if current.action:
                        path.append(current.action)
                    current = current.parent
                
                if path:
                    path.reverse()
                    print(f"Solution found in {len(path)} moves after {i+1} iterations")
                    return path
        
        # If we didn't find a complete solution, return the best path
        best_path = self.get_best_path(root)
        print(f"No complete solution found. Best path: {len(best_path)} moves")
        return best_path