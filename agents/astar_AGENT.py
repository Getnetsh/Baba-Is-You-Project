"""
Implementazione dell'agente A* per il gioco KekeAI.
Utilizza l'algoritmo di ricerca A* con funzione euristica per trovare la soluzione ottimale.
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
        Crea un hash unico per lo stato del gioco includendo posizioni e regole.
        """
        # Serializza la mappa degli oggetti
        obj_map_str = ""
        for row in state.obj_map:
            for cell in row:
                if hasattr(cell, 'name'):
                    obj_map_str += cell.name
                else:
                    obj_map_str += str(cell)
        
        # Serializza la mappa di sfondo
        back_map_str = ""
        for row in state.back_map:
            for cell in row:
                if hasattr(cell, 'name'):
                    back_map_str += cell.name
                else:
                    back_map_str += str(cell)
        
        # Aggiungi le regole
        rules_str = ''.join(sorted(state.rules))
        
        return f"{obj_map_str}|{back_map_str}|{rules_str}"
    
    def heuristic(self, state: GameState) -> float:
        """
        Funzione euristica migliorata che considera multiple strategie.
        """
        if check_win(state):
            return 0.0
        
        # Se non ci sono giocatori o oggetti vincenti, stato impossibile
        if not state.players or not state.winnables:
            return 1000.0
        
        # Calcola la distanza minima da qualsiasi giocatore a qualsiasi oggetto vincente
        min_distance = float('inf')
        
        for player in state.players:
            player_pos = (player.x, player.y)
            for winnable in state.winnables:
                winnable_pos = (winnable.x, winnable.y)
                # Distanza di Manhattan
                distance = abs(player_pos[0] - winnable_pos[0]) + abs(player_pos[1] - winnable_pos[1])
                min_distance = min(min_distance, distance)
        
        # Se la distanza è infinita, penalizza pesantemente
        if min_distance == float('inf'):
            return 1000.0
        
        # Bonus per avere più giocatori e oggetti vincenti
        bonus = 0
        if len(state.players) > 1:
            bonus -= 1
        if len(state.winnables) > 1:
            bonus -= 1
            
        return max(0, min_distance + bonus)

    def search(self, initial_state: GameState, iterations: int = 1000) -> List[Direction]:
        """
        Implementa l'algoritmo A* per trovare il percorso ottimale.
        
        :param initial_state: Lo stato iniziale del gioco
        :param iterations: Numero massimo di iterazioni
        :return: Lista delle azioni che portano alla soluzione
        """
        # Interpreta le regole iniziali
        interpret_rules(initial_state)
        
        # Priority queue: (f_cost, g_cost, state_hash, state, action_history)
        priority_queue = []
        
        # Calcola il costo euristico iniziale
        initial_state_hash = self.get_state_hash(initial_state)
        initial_h_cost = self.heuristic(initial_state)
        initial_f_cost = 0 + initial_h_cost
        
        # Aggiungi lo stato iniziale
        heapq.heappush(priority_queue, (initial_f_cost, 0, initial_state_hash, initial_state, []))
        
        # Inizializza le strutture dati
        self.visited_states = set()
        self.g_costs = {initial_state_hash: 0}
        
        # Contatore per evitare cicli infiniti
        nodes_expanded = 0
        
        for i in trange(iterations, desc="A* Search"):
            if not priority_queue:
                print("Priority queue vuota - nessuna soluzione trovata")
                break
            
            # Estrai il nodo con il costo f più basso
            f_cost, g_cost, state_hash, current_state, actions = heapq.heappop(priority_queue)
            
            # Se già visitato con costo migliore, salta
            if state_hash in self.visited_states:
                continue
                
            # Marca come visitato
            self.visited_states.add(state_hash)
            nodes_expanded += 1
            
            # Controlla se abbiamo vinto
            if check_win(current_state):
                print(f"Soluzione trovata in {len(actions)} mosse dopo {nodes_expanded} nodi espansi")
                return actions
            
            # Esplora tutte le azioni possibili
            for action in [Direction.Up, Direction.Down, Direction.Left, Direction.Right, Direction.Wait]:
                # Crea il nuovo stato
                try:
                    next_state = advance_game_state(action, current_state.copy())
                    interpret_rules(next_state)  # Assicurati che le regole siano aggiornate
                    next_state_hash = self.get_state_hash(next_state)
                    
                    # Salta se già visitato
                    if next_state_hash in self.visited_states:
                        continue
                    
                    # Calcola i costi
                    tentative_g_cost = g_cost + 1
                    
                    # Se abbiamo già un percorso migliore, salta
                    if next_state_hash in self.g_costs and tentative_g_cost >= self.g_costs[next_state_hash]:
                        continue
                    
                    # Questo è il miglior percorso finora
                    self.g_costs[next_state_hash] = tentative_g_cost
                    next_h_cost = self.heuristic(next_state)
                    next_f_cost = tentative_g_cost + next_h_cost
                    
                    # Aggiungi alla coda di priorità
                    new_actions = actions + [action]
                    heapq.heappush(priority_queue, (next_f_cost, tentative_g_cost, next_state_hash, next_state, new_actions))
                    
                except Exception as e:
                    # Gestisci errori nella generazione dello stato
                    continue
        
        print(f"Nessuna soluzione trovata dopo {nodes_expanded} nodi espansi")
        return []