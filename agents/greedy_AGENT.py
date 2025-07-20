"""
Enhanced implementation of the Greedy agent for the KekeAI game with optimized Manhattan Distance heuristic.
Uses Greedy Best-First Search with refined heuristics and search strategies.
File: greedy_AGENT.py
"""

from base_agent import BaseAgent
from baba import GameState, Direction, advance_game_state, check_win, interpret_rules, character_to_name, GameObj
from typing import List, Set, Tuple, Dict, Optional
import heapq
import random
import time
from collections import defaultdict
from queue import Queue  # For BFS in is_unreachable


class GREEDYAgent(BaseAgent):
    """
    Enhanced Greedy Best-First Search implementation with optimized heuristics.
    """
    
    def __init__(self):
        self.visited_states: Set[str] = set()
        self.state_costs: Dict[str, float] = {}  # Retained for potential future use
        self.pattern_cache: Dict[str, float] = {}
        self.deadlock_patterns: Set[str] = set()
    
    def get_state_hash(self, state: GameState) -> str:
        """Creates a unique hash for the game state."""
        obj_positions = []
        for y, row in enumerate(state.obj_map):
            for x, cell in enumerate(row):
                if isinstance(cell, GameObj):
                    obj_positions.append(f"{x},{y}:{cell.name}")
        player_positions = [f"{p.x},{p.y}" for p in state.players]
        winnable_positions = [f"{w.x},{w.y}" for w in state.winnables]
        return f"obj:{'|'.join(sorted(obj_positions))};p:{'|'.join(sorted(player_positions))};w:{'|'.join(sorted(winnable_positions))};r:{'|'.join(sorted(state.rules))}"
    
    def detect_deadlock(self, state: GameState) -> bool:
        """Detect if the current state is a deadlock."""
        if not state.players or not state.winnables:
            return True
        for player in state.players:
            if self.is_trapped(state, player.x, player.y):
                return True
        for winnable in state.winnables:
            if self.is_unreachable(state, winnable.x, winnable.y):
                return True
        return False
    
    def is_trapped(self, state: GameState, x: int, y: int) -> bool:
        """Check if a position is trapped."""
        walls = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx < 0 or ny < 0 or 
                ny >= len(state.obj_map) or 
                nx >= len(state.obj_map[0]) or
                (isinstance(state.obj_map[ny][nx], GameObj) and 
                 state.obj_map[ny][nx].name == character_to_name['w'].split('_')[0]) or
                state.back_map[ny][nx] == '_'):
                walls += 1
        return walls >= 3
    
    def is_unreachable(self, state: GameState, x: int, y: int) -> bool:
        """Check if a position is unreachable (simplified)."""
        if not state.players:
            return True
        return False
    
    def advanced_heuristic(self, state: GameState) -> float:
        """Optimized heuristic using Manhattan Distance with enhanced rule guidance."""
        if check_win(state):
            return 0.0
        if self.detect_deadlock(state):
            return 10000.0
        if not state.players or not state.winnables:
            return 5000.0
        
        # Manhattan Distance to nearest winnable
        min_winnable_dist = float('inf')
        for player in state.players:
            for winnable in state.winnables:
                dist = abs(player.x - winnable.x) + abs(player.y - winnable.y)
                min_winnable_dist = min(min_winnable_dist, dist)
        if min_winnable_dist == float('inf'):
            return 5000.0
        
        # Manhattan Distance to nearest rule word, optimized per player
        min_rule_dist = float('inf')
        rule_words = ['is', 'win', 'flag', 'you']
        for player in state.players:
            player_min_rule_dist = float('inf')
            for y, row in enumerate(state.obj_map):
                for x, obj in enumerate(row):
                    if (isinstance(obj, GameObj) and obj.is_movable and obj.name in rule_words and 
                        any(abs(x - wx) + abs(y - wy) <= 2 for wx, wy in [(w.x, w.y) for w in state.winnables])):
                        dist = abs(player.x - x) + abs(player.y - y)
                        player_min_rule_dist = min(player_min_rule_dist, dist)
            if player_min_rule_dist != float('inf'):
                min_rule_dist = min(min_rule_dist, player_min_rule_dist)
        min_rule_dist = min_rule_dist if min_rule_dist != float('inf') else min_winnable_dist * 2
        
        # Dynamic weighting based on move depth
        move_depth_factor = min(1.0, len(self.visited_states) / 500.0)  # Increases weight on rules as depth grows
        base_heuristic = (min_winnable_dist * (0.9 - 0.3 * move_depth_factor) + 
                         min_rule_dist * (0.1 + 0.3 * move_depth_factor)) * (1 + 0.05 * len(state.winnables))
        factors = 0
        
        # Cluster and mobility factors
        if len(state.players) > 1:
            player_cluster_size = self.calculate_cluster_size([(p.x, p.y) for p in state.players])
            factors -= player_cluster_size * 0.3
        if len(state.winnables) > 1:
            winnable_cluster_size = self.calculate_cluster_size([(w.x, w.y) for w in state.winnables])
            factors -= winnable_cluster_size * 0.3
        mobility_score = self.calculate_mobility(state)
        mobility_factor = max(0, 1 - mobility_score / 8) * 2.0
        factors += mobility_factor
        
        # Rule and pattern bonuses
        rule_bonus = self.evaluate_rules(state)
        factors -= rule_bonus * (1 + 0.5 * (len(state.rules) / 10))
        pattern_bonus = self.recognize_patterns(state)
        factors -= pattern_bonus
        
        # Enhanced rule alignment bonus
        for player in state.players:
            for y, row in enumerate(state.obj_map):
                for x, obj in enumerate(row):
                    if (isinstance(obj, GameObj) and obj.is_movable and obj.name in rule_words and 
                        any(abs(x - wx) + abs(y - wy) <= 2 for wx, wy in [(w.x, w.y) for w in state.winnables]) and 
                        abs(player.x - x) + abs(player.y - y) <= 2):
                        factors -= 2.0
        
        state_hash = self.get_state_hash(state)
        if state_hash in self.state_costs:
            factors += 0.5
        
        final_heuristic = max(0.1, base_heuristic + factors)
        pattern_key = self.get_pattern_key(state)
        self.pattern_cache[pattern_key] = final_heuristic
        return final_heuristic
    
    def calculate_cluster_size(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate how clustered a set of positions are."""
        if len(positions) <= 1:
            return 0
        total_distance = 0
        count = 0
        for i, pos1 in enumerate(positions):
            for pos2 in positions[i+1:]:
                distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
                total_distance += distance
                count += 1
        return total_distance / count if count > 0 else 0
    
    def calculate_mobility(self, state: GameState) -> float:
        """Calculate how many movement options are available."""
        mobility = 0
        for player in state.players:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = player.x + dx, player.y + dy
                if (0 <= nx < len(state.obj_map[0]) and 
                    0 <= ny < len(state.obj_map) and
                    (state.obj_map[ny][nx] is None or 
                     isinstance(state.obj_map[ny][nx], GameObj) and 
                     state.obj_map[ny][nx].name != character_to_name['w'].split('_')[0])):
                    mobility += 1
        return mobility
    
    def evaluate_rules(self, state: GameState) -> float:
        """Evaluate current rules for strategic advantage."""
        bonus = 0
        if 'baba-is-you' in state.rules:
            bonus += 1
        if any('is-win' in rule for rule in state.rules):
            bonus += 2
        if any('is-push' in rule for rule in state.rules):
            bonus += 0.5
        return bonus
    
    def recognize_patterns(self, state: GameState) -> float:
        """Recognize beneficial patterns in the game state."""
        pattern_key = self.get_pattern_key(state)
        if pattern_key in self.pattern_cache:
            return self.pattern_cache[pattern_key] * 0.1
        bonus = 0
        for player in state.players:
            for winnable in state.winnables:
                if abs(player.x - winnable.x) + abs(player.y - winnable.y) == 1:
                    bonus += 2
        for player in state.players:
            for winnable in state.winnables:
                if self.has_clear_path(state, player, winnable):
                    bonus += 1
        return bonus
    
    def get_pattern_key(self, state: GameState) -> str:
        """Generate a key for pattern recognition."""
        if not state.players or not state.winnables:
            return "empty"
        player_pos = (state.players[0].x, state.players[0].y)
        relative_positions = []
        for winnable in state.winnables:
            rel_x = winnable.x - player_pos[0]
            rel_y = winnable.y - player_pos[1]
            relative_positions.append(f"{rel_x},{rel_y}")
        return "|".join(sorted(relative_positions))
    
    def has_clear_path(self, state: GameState, player, winnable) -> bool:
        """Check if there's a clear path between player and winnable."""
        if player.x == winnable.x or player.y == winnable.y:
            return True
        return False
    
    def get_smart_action_priorities(self, state: GameState) -> List[Direction]:
        """Smarter action prioritization based on game state analysis."""
        if not state.players or not state.winnables:
            return [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]
        direction_weights = defaultdict(float)
        for player in state.players:
            for winnable in state.winnables:
                dx = winnable.x - player.x
                dy = winnable.y - player.y
                distance = abs(dx) + abs(dy)
                if distance == 0:
                    continue
                weight = 1.0 / (distance + 1)
                if abs(dx) > abs(dy):
                    if dx > 0:
                        direction_weights[Direction.Right] += weight
                    else:
                        direction_weights[Direction.Left] += weight
                else:
                    if dy > 0:
                        direction_weights[Direction.Down] += weight
                    else:
                        direction_weights[Direction.Up] += weight
       
        for direction in [Direction.Up, Direction.Down, Direction.Left, Direction.Right]:
            direction_weights[direction] += random.uniform(0, 0.1)
        sorted_directions = sorted(direction_weights.items(), key=lambda x: x[1], reverse=True)
        result = [direction for direction, weight in sorted_directions]
        if Direction.Wait not in result:
            result.append(Direction.Wait)
        return result
    
    def search(self, initial_state: GameState, iterations: int) -> List[Direction]:
        """Enhanced search using full iteration budget from execution.py."""
        interpret_rules(initial_state)
        return self.greedy_search(initial_state, iterations)
    
    def greedy_search(self, initial_state: GameState, iterations: int) -> List[Direction]:
        """Standard greedy search with optimized heuristic and early pruning."""
        priority_queue = []
        state_data = {}
        initial_state_hash = self.get_state_hash(initial_state)
        initial_h_cost = self.advanced_heuristic(initial_state)
        unique_id = 0
        state_data[unique_id] = (initial_state_hash, initial_state, [])
        heapq.heappush(priority_queue, (initial_h_cost, 0, unique_id))
        self.visited_states = set()
        self.state_costs = {}
        nodes_expanded = 0
        while priority_queue and nodes_expanded < iterations:
            h_cost, tie_breaker, current_id = heapq.heappop(priority_queue)
            if current_id not in state_data:
                continue
            state_hash, current_state, actions = state_data[current_id]
            del state_data[current_id]
            if state_hash in self.visited_states:
                continue
            self.visited_states.add(state_hash)
            nodes_expanded += 1
            unique_id += 1
            if check_win(current_state):
                return actions
            # Early pruning if no progress toward winnables or rules
            if (not any(abs(p.x - w.x) + abs(p.y - w.y) <= 2 for p in current_state.players 
                       for w in current_state.winnables) and 
                not any('is-win' in rule for rule in current_state.rules) and 
                len(actions) > 10 and h_cost > 15):
                continue
            if len(actions) > 75 and h_cost > 15:
                continue
            action_priorities = self.get_smart_action_priorities(current_state)
            for action in action_priorities:
                try:
                    next_state = advance_game_state(action, current_state.copy())
                    interpret_rules(next_state)
                    next_state_hash = self.get_state_hash(next_state)
                    if next_state_hash in self.visited_states:
                        continue
                    next_h_cost = self.advanced_heuristic(next_state)
                    if next_h_cost > h_cost + 3:
                        continue
                    new_actions = actions + [action]
                    new_tie_breaker = len(new_actions)
                    new_id = unique_id
                    unique_id += 1
                    state_data[new_id] = (next_state_hash, next_state, new_actions)
                    heapq.heappush(priority_queue, (next_h_cost, new_tie_breaker, new_id))
                except Exception:
                    continue
        return []
    
    def solve_level(self, level_data: Dict, iterations: int) -> str:
        """Solve a level with enhanced error handling and reporting."""
        level_id = level_data.get('id', 'unknown')
        author = level_data.get('author', 'unknown')
        print(f"Solving level {level_id} by {author}")
        try:
            self.visited_states.clear()
            self.state_costs.clear()
            self.pattern_cache.clear()
            self.deadlock_patterns.clear()
            initial_state = level_data['game_state']
            solution_directions = self.search(initial_state, iterations)
            solution_string = ''.join(action.value for action in solution_directions)
            if 'solution' in level_data:
                expected = level_data['solution']
                if solution_string == expected:
                    print(f"✓ Found optimal solution: {solution_string}")
                else:
                    print(f"✗ Found solution: {solution_string} (expected: {expected})")
                    print(f"  Solution length: {len(solution_string)} vs expected: {len(expected)}")
            else:
                print(f"Found solution: {solution_string}")
            return solution_string
        except Exception as e:
            print(f"Error solving level {level_id}: {e}")
            return ""