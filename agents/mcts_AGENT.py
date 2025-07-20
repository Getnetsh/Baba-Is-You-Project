from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, evaluate_move_heuristic, move_creates_player_rule, move_creates_win_rule, move_removes_obstacle_rule, state_brings_player_closer_to_win, move_creates_positive_rule, check_player_deadlocked, something_was_pushed, destructive_rule_removed, something_sank, move_closer_to_destructive_rule, move_creates_win_condition
from typing import List, Dict
import math
import random
from tqdm import trange
from collections import defaultdict


class MCTSNode:
    def __init__(self, state: GameState, parent=None, action=None, isInvalid=False):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[Direction, 'MCTSNode'] = {}
        self.visits = 0
        self.wins = 0
        self.isInvalid = isInvalid

    def is_fully_expanded(self):
        return len(self.children) == len(Direction)-1  # Excludes Direction.Undefined

    def best_child(self, c_param=1.4):
        # Filter out no-op children first
        valid_children = [c for c in self.children.values() if not getattr(c, "isInvalid", False)]

        # If no valid children, fallback to all children (to avoid empty list)
        if not valid_children:
            valid_children = list(self.children.values())

        # Adaptive exploration: reduce c_param for well-visited nodes
        if self.visits > 100:
            c_param *= 0.8

        # UCT formula with win bonus for nodes that lead to wins
        return max(
            valid_children,
            key=lambda child: (child.wins / (child.visits + 1e-6)) +
            c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6)) +
            (0.1 if any(check_win(grandchild.state) for grandchild in child.children.values()) else 0)
        )


class MCTSAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        # Simple state tracking to avoid immediate loops
        self.recent_states = defaultdict(int)
        
    def search(self, initial_state: GameState, iterations: int = 250) -> List[Direction]:
        self.recent_states.clear()  # Reset for new search
        
        root = MCTSNode(initial_state)

        # Early termination if we find wins quickly
        early_win_found = False
        
        for i in trange(iterations):
            # Select a node to explore
            node = self.select(root)

            # If the node is a winning state, we can return immediately
            if check_win(node.state):
                return self.extract_solution(node)
            
            child = self.expand(node)
            if child is None:
                continue

            result = self.simulate(child)
            self.backpropagate(child, result)

            # Check for wins in tree periodically
            if i % 50 == 0 and i > 0:
                win_node = self.find_win_in_tree(root)
                if win_node is not None:
                    early_win_found = True
                    break

        # Try to find a path to a win in the tree
        if not early_win_found:
            win_node = self.find_win_in_tree(root)
        if win_node is not None:
            return self.extract_solution(win_node)

        # If no win found, return the most promising path
        best_path_node = self.deepest_best_node(root)
        if best_path_node is not None:
            return self.extract_solution(best_path_node)
        return []

    def find_win_in_tree(self, node):
        # Recursively search for a node in the tree that is a win
        if check_win(node.state):
            return node
        for child in node.children.values():
            if not child.isInvalid:  # Skip invalid children
                result = self.find_win_in_tree(child)
                if result is not None:
                    return result
        return None

    def deepest_best_node(self, root):
        # Find the deepest node with the highest win rate, but require minimum visits
        stack = [(root, 0)]
        best = (None, -1, -1.0)  # (node, depth, winrate)
        while stack:
            node, depth = stack.pop()
            if node.visits > 3:  # Only consider nodes with some visits
                winrate = node.wins / node.visits
                # Prefer deeper nodes with good win rates
                if depth > best[1] or (depth == best[1] and winrate > best[2]):
                    best = (node, depth, winrate)
            for child in node.children.values():
                if not child.isInvalid:
                    stack.append((child, depth + 1))
        return best[0]

    def select(self, node): 
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node

    def expand(self, node):
        tried_actions = set(node.children.keys())
        
        # Prioritize actions that haven't been tried yet
        available_actions = [d for d in Direction if d != Direction.Undefined and d not in tried_actions]
        
        if not available_actions:
            return node
            
        # Smart action prioritization based on domain knowledge
        action_scores = []
        for action in available_actions:
            score = 0
            
            # Quick simulation to score actions
            next_state = advance_game_state(action, node.state.copy())
            
            if str(next_state) != str(node.state):  # Valid move
                # Immediate win gets highest priority
                if check_win(next_state):
                    score += 1000
                # Progress toward win
                elif state_brings_player_closer_to_win(node.state, next_state):
                    score += 100
                # Avoid deadlocks
                elif check_player_deadlocked(next_state):
                    score -= 500
                # General heuristic
                score += evaluate_move_heuristic(node.state, next_state) * 0.1
            
            # Add small random factor to break ties
            score += random.uniform(-5, 5)
            action_scores.append((action, score))
        
        # Sort by score (highest first) and try the best action
        action_scores.sort(key=lambda x: x[1], reverse=True)
        action = action_scores[0][0]
        
        next_state = advance_game_state(action, node.state.copy())

        if str(next_state) == str(node.state):  # skip invalid/no-op moves
            invalid_child = MCTSNode(state=next_state, parent=node, action=action, isInvalid=True)
            node.children[action] = invalid_child
            return invalid_child
        else:
            valid_child = MCTSNode(state=next_state, parent=node, action=action)
            node.children[action] = valid_child
            return valid_child

    def simulate(self, node, rollout_limit=50):
        """
        Enhanced heuristic simulation with better scoring
        """
        state = node.state
        parent = node.parent.state if node.parent is not None else None
        grandparent = node.parent.parent.state if node.parent is not None and node.parent.parent is not None else None

        # Win/loss checks (keep these the same)
        if check_win(state):
            return 10000
        if check_player_deadlocked(state):
            return -500

        # Your working heuristic logic with small enhancements
        score = 0
        if parent is not None:
            # Your proven weights
            if state_brings_player_closer_to_win(parent, state):
                score += 1300
            
            # Add some additional heuristics with smaller weights
            if move_creates_positive_rule(parent, state):
                score += 300
            if something_was_pushed(parent, state):
                score += 150
            if destructive_rule_removed(parent, state):
                score += 400
            if something_sank(parent, state):
                score -= 200
            
            score += evaluate_move_heuristic(parent, state)

        # Keep your back-and-forth penalty but make it stronger
        if grandparent is not None:
            if str(state) == str(grandparent):
                score -= 1200

        # Enhanced state repetition penalty
        state_str = str(state)
        self.recent_states[state_str] += 1
        if self.recent_states[state_str] > 2:
            score -= 75 * (self.recent_states[state_str] - 1)

        return score

    def legal_moves(self, state):
        # Keep your existing logic
        moves = []
        for d in Direction:
            if d is Direction.Undefined:
                continue
            next_state = advance_game_state(d, state)
            if str(next_state) != str(state):
                moves.append(d)
        return moves 

    def backpropagate(self, node, result):
        # Keep your existing logic
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    def extract_solution(self, node) -> List[Direction]:
        # Keep your existing logic
        actions = []
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        return actions[::-1]