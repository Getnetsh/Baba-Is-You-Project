"""
Implementazione dell'agente MCTS (Monte Carlo Tree Search) per il gioco KekeAI.
Utilizza l'algoritmo MCTS con simulazioni Monte Carlo per trovare la soluzione ottimale.
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
    Nodo per l'albero MCTS.
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
        # Serializza la mappa degli oggetti
        obj_map_str = ""
        for row in self.state.obj_map:
            for cell in row:
                if hasattr(cell, 'name'):
                    obj_map_str += cell.name
                else:
                    obj_map_str += str(cell)
        
        # Serializza la mappa di sfondo
        back_map_str = ""
        for row in self.state.back_map:
            for cell in row:
                if hasattr(cell, 'name'):
                    back_map_str += cell.name
                else:
                    back_map_str += str(cell)
        
        # Aggiungi le regole
        rules_str = ''.join(sorted(self.state.rules))
        
        return f"{obj_map_str}|{back_map_str}|{rules_str}"
    
    def is_fully_expanded(self) -> bool:
        """Controlla se tutte le azioni sono state provate."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Controlla se questo è un nodo terminale."""
        return check_win(self.state)
    
    def uct_value(self, exploration_constant: float = 1.4) -> float:
        """Calcola il valore UCT (Upper Confidence Bound for Trees)."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def best_child(self, exploration_constant: float = 1.4) -> 'MCTSNode':
        """Seleziona il figlio migliore usando UCT."""
        return max(self.children, key=lambda child: child.uct_value(exploration_constant))
    
    def select_untried_action(self) -> Direction:
        """Seleziona un'azione non ancora provata."""
        return random.choice(self.untried_actions)


class MCTSAgent(BaseAgent):
    """
    MCTS (Monte Carlo Tree Search) implementation.
    """
    
    def __init__(self, exploration_constant: float = 1.4):
        self.exploration_constant = exploration_constant
        self.visited_states: Set[str] = set()
    
    def get_distance_to_goal(self, state: GameState) -> float:
        """Calcola la distanza minima dal giocatore all'oggetto vincente."""
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
        """Fase di selezione: naviga l'albero usando UCT."""
        current = node
        
        while not current.is_terminal() and current.is_fully_expanded():
            if not current.children:
                break
            current = current.best_child(self.exploration_constant)
        
        return current
    
    def expand(self, node: MCTSNode) -> MCTSNode:
        """Fase di espansione: aggiungi un nuovo figlio."""
        if node.is_terminal():
            return node
        
        if not node.untried_actions:
            return node
        
        # Seleziona un'azione non provata
        action = node.select_untried_action()
        node.untried_actions.remove(action)
        
        try:
            # Crea il nuovo stato
            new_state = advance_game_state(action, node.state.copy())
            interpret_rules(new_state)
            
            # Crea il nuovo nodo figlio
            child_node = MCTSNode(new_state, parent=node, action=action)
            node.children.append(child_node)
            
            return child_node
        except Exception as e:
            # Se c'è un errore, ritorna il nodo corrente
            return node
    
    def simulate(self, state: GameState, max_depth: int = 50) -> float:
        """
        Fase di simulazione: esegui una partita casuale con euristica.
        """
        current_state = state.copy()
        visited_in_simulation: Set[str] = set()
        initial_distance = self.get_distance_to_goal(current_state)
        
        for step in range(max_depth):
            # Controlla se abbiamo vinto
            if check_win(current_state):
                return 1.0
            
            # Evita cicli nella simulazione
            state_hash = self._get_state_hash(current_state)
            if state_hash in visited_in_simulation:
                break
            visited_in_simulation.add(state_hash)
            
            # Scegli un'azione con euristica
            action = self._choose_action_with_heuristic(current_state)
            
            try:
                current_state = advance_game_state(action, current_state.copy())
                interpret_rules(current_state)
            except Exception as e:
                break
        
        # Calcola la ricompensa basata su quanto ci siamo avvicinati all'obiettivo
        final_distance = self.get_distance_to_goal(current_state)
        
        if final_distance == float('inf'):
            return 0.0
        
        # Ricompensa basata sul miglioramento
        if initial_distance == float('inf'):
            return 0.1
        
        improvement = max(0, initial_distance - final_distance)
        return min(1.0, improvement / max(1, initial_distance))
    
    def _get_state_hash(self, state: GameState) -> str:
        """Crea un hash dello stato."""
        obj_map_str = ""
        for row in state.obj_map:
            for cell in row:
                if hasattr(cell, 'name'):
                    obj_map_str += cell.name
                else:
                    obj_map_str += str(cell)
        return obj_map_str
    
    def _choose_action_with_heuristic(self, state: GameState) -> Direction:
        """Scegli un'azione usando un'euristica semplice."""
        if not state.players or not state.winnables:
            return random.choice(list(Direction))
        
        player = state.players[0]
        target = state.winnables[0]
        
        # Calcola la direzione verso l'obiettivo
        dx = target.x - player.x
        dy = target.y - player.y
        
        # Scegli l'azione con bias verso l'obiettivo
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
        
        # Scelta pesata
        total_weight = sum(weight for _, weight in actions_weights)
        rand_val = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for action, weight in actions_weights:
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                return action
        
        return random.choice(list(Direction))
    
    def backpropagate(self, node: MCTSNode, reward: float):
        """Fase di backpropagazione: aggiorna i valori dei nodi."""
        current = node
        while current is not None:
            current.visits += 1
            current.wins += reward
            current = current.parent
    
    def get_best_path(self, root: MCTSNode) -> List[Direction]:
        """Ottieni il miglior percorso dall'albero MCTS."""
        path = []
        current = root
        
        while current.children:
            # Scegli il figlio con il maggior numero di visite (più esplorato)
            best_child = max(current.children, key=lambda child: child.visits)
            
            if best_child.action:
                path.append(best_child.action)
            
            current = best_child
            
            # Se troviamo una soluzione, fermati
            if check_win(current.state):
                break
        
        return path
    
    def search(self, initial_state: GameState, iterations: int = 1000) -> List[Direction]:
        """
        Implementa l'algoritmo MCTS per trovare la soluzione.
        
        :param initial_state: Lo stato iniziale del gioco
        :param iterations: Numero di iterazioni MCTS
        :return: Lista delle azioni che portano alla soluzione
        """
        # Interpreta le regole iniziali
        interpret_rules(initial_state)
        
        # Crea il nodo radice
        root = MCTSNode(initial_state)
        
        # Esegui le iterazioni MCTS
        for i in trange(iterations, desc="MCTS Search"):
            # 1. Selezione
            selected_node = self.select(root)
            
            # 2. Espansione
            expanded_node = self.expand(selected_node)
            
            # 3. Simulazione
            reward = self.simulate(expanded_node.state)
            
            # 4. Backpropagazione
            self.backpropagate(expanded_node, reward)
            
            # Controlla se abbiamo trovato una soluzione
            if expanded_node.is_terminal():
                # Ricostruisci il percorso verso la soluzione
                path = []
                current = expanded_node
                while current.parent is not None:
                    if current.action:
                        path.append(current.action)
                    current = current.parent
                
                if path:
                    path.reverse()
                    print(f"Soluzione trovata in {len(path)} mosse dopo {i+1} iterazioni")
                    return path
        
        # Se non abbiamo trovato una soluzione, ritorna il miglior percorso
        best_path = self.get_best_path(root)
        print(f"Nessuna soluzione completa trovata. Miglior percorso: {len(best_path)} mosse")
        return best_path