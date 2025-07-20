
# Baba is Y'all mechanic/rule base
# Version: 2.0py
# Original Code by Milk
# Translated to Python by Descar
import copy
from dataclasses import dataclass
from typing import List, Dict, Union
from enum import Enum
import pygame
from copy import deepcopy

# Maps of Baba is You are stored in ascii
# Each character is assigned to a tile used in the game.
# Tiles ending with the term "obj" are elements depicted by images
# Tiles ending with the term "word" represent "obj" and can be used to form rules
character_to_name = {
    '_': "border",
    ' ': "empty",
    # '.': "empty",
    'b': "baba_obj",
    'B': "baba_word",
    '1': "is_word",
    '2': "you_word",
    '3': "win_word",
    's': "skull_obj",
    'S': "skull_word",
    'f': "flag_obj",
    'F': "flag_word",
    'o': "floor_obj",
    'O': "floor_word",
    'a': "grass_obj",
    'A': "grass_word",
    '4': "kill_word",
    'l': "lava_obj",
    'L': "lava_word",
    '5': "push_word",
    'r': "rock_obj",
    'R': "rock_word",
    '6': "stop_word",
    'w': "wall_obj",
    'W': "wall_word",
    '7': "move_word",
    '8': "hot_word",
    '9': "melt_word",
    'k': "keke_obj",
    'K': "keke_word",
    'g': "goop_obj",
    'G': "goop_word",
    '0': "sink_word",
    'v': "love_obj",
    'V': "love_word",
}

# reverse mapping from game object to character in the map
name_to_character = {y: x for x, y in character_to_name.items()}


# direct mapping from ascii characters to images for faster drawing of the map
def make_img_hash():
    """
    Create a dictionary that maps ASCII characters to their respective images.

    :return: Dictionary where keys are characters and values are Pygame image surfaces.
    """
    img_hash = {}
    all_chars = character_to_name.keys()

    for c in all_chars:
        # Check if the character already has an assigned image,
        # e.g. in case the character_to_name dictionary contains one element multiple times
        if c not in img_hash:
            img = pygame.image.load("img/" + character_to_name[c] + ".png")
            img_hash[c] = img
    return img_hash


imgHash = make_img_hash()

# features and their pairs that have special interaction rules
features = ["hot", "melt", "open", "shut", "move"]
featPairs = [["hot", "melt"], ["open", "shut"]]


class Direction(Enum):
    """
    Enumeration for directional values with an Undefined state.
    """
    Left = 'l'
    Right = 'r'
    Up = 'u'
    Down = 'd'
    Wait = 's'
    Undefined = None

    @classmethod
    def _missing_(cls, value):
        """
        Returns the Undefined direction for invalid or unrecognized values.
        """
        return cls.Undefined

    @staticmethod
    def opposite(value):
        """
        Returns the opposite direction for a given Direction value.

        Args:
            value (Direction): The direction to find the opposite of.

        Returns:
            Direction: The opposite direction, or Undefined if given Wait or Undefined.
        """
        if value == Direction.Left:
            return Direction.Right
        if value == Direction.Right:
            return Direction.Left
        if value == Direction.Up:
            return Direction.Down
        if value == Direction.Down:
            return Direction.Up
        else:
            return Direction.Undefined


class GameObjectType(Enum):
    """
    Enumeration for differentiating types of game objects.
    """
    Physical = 1            # physical objects may be changed due to rules such as "baba is rock"
    Word = 2                # words cannot be changed due to rules
    Keyword = 3             # keywords cannot be changed due to rules, but signify a rule to be interpreted
    Undefined = None

    @classmethod
    def _missing_(cls, value):
        return cls.Undefined


@dataclass
class GameObj:
    name: str                       # name of the object given character_to_name
    x: int                          # current x-position
    y: int                          # current y-position
    img: str                        # the name of the image to be drawn in render-mode
    object_type: GameObjectType     # the type of gam object used for rule identification
    obj: str                        # only used for word objects
    is_movable: bool                # can the objects be pushed around?
                                    # word objects and keyword objects are always pushable
                                    # physical objects only if a move rules allows to
    is_stopped: bool                #
    feature: str                    # only used for physical objects
    dir: Direction                  # only used for physical object

    def __init__(self, name: GameObjectType,
                 img: str,
                 x: int,
                 y: int,
                 object_type: GameObjectType = GameObjectType.Undefined,
                 obj: str = "",
                 is_movable: bool = False,
                 is_stopped: bool = False):
        self.name = name
        self.x = x
        self.y = y
        self.img = img
        self.object_type = object_type

        self.is_movable = is_movable
        self.is_stopped = is_stopped

        # physical object features
        self.feature = ""
        self.dir = Direction.Undefined

        # word object features
        self.obj = obj

    @classmethod
    def create_physical_object(cls, name, img_character, x, y):
        return cls(name, img_character, x, y, object_type=GameObjectType.Physical)

    @classmethod
    def create_word_object(cls, name, img_character, obj, x, y):
        return cls(name, img_character, x, y, obj=obj, object_type=GameObjectType.Word, is_movable=True)

    @classmethod
    def create_keyword_object(cls, name, img_character, obj, x, y):
        return cls(name, img_character, x, y, obj=obj, object_type=GameObjectType.Keyword, is_movable=True)


@dataclass
class GameState:
    orig_map: List
    obj_map: List[List[Union[str, GameObj]]]        # foreground objects
    back_map: List[List[Union[str, GameObj]]]       # background objects
    words: List                                     # all word objects
    keywords: List                                  # all keyword objects
    phys: List                                      # all physical objects
    is_connectors: List                             # all is keywords (todo: could be merged with keywords)
    sort_phys: Dict                                 # physical objects sorted by type
    rules: List                                     # all rules that have been found
    rule_objs: List                                 # todo: description missing
    players: List                                   # all player objects
                                                    # (used to check win condition, created by "x-is-you" rules)
    auto_movers: List                               # all characters that move automatically ("x-is-move")
    winnables: List                                 # all objects that let the player win, when touched ("x-is-win")
    pushables: List                                 # all objects that can be pushed ("x-is-push")
    killers: List                                   # all objects that kill player objects when touched ("x-is-kill")
    sinkers: List                                   # all objects that remove other objects when pushed into ("x-is-sink")
    featured: Dict                                  # todo: description missing
    overlaps: List                                  # todo: description missing
    unoverlaps: List                                # todo: description missing
    lazy_evaluation_properties: Dict                # characteristics that may be read multiple times but are
                                                    # inefficient to evaluate, e.g. is this a winning game state

    def __init__(self):
        self.reset()

    def reset(self):
        self.orig_map = []
        self.obj_map = []
        self.back_map = []
        self.words = []
        self.keywords = []
        self.phys = []
        self.is_connectors = []
        self.sort_phys = {}
        self.rules = []
        self.rule_objs = []
        self.players = []
        self.auto_movers = []
        self.winnables = []
        self.pushables = []
        self.killers = []
        self.sinkers = []
        self.featured = {}
        self.overlaps = []
        self.unoverlaps = []
        self.lazy_evaluation_properties = dict()

    def copy(self):
        new_game_state = copy.deepcopy(self)
        new_game_state.lazy_evaluation_properties = dict()
        return new_game_state

    def __str__(self):
        return double_map_to_string(self.obj_map, self.back_map)


def advance_game_state(action: Direction, state: GameState):
    """
    Advances the game state based on a given action and updates game rules and win conditions.

    Args:
        action (Direction): The player's action, represented as a `Direction` enum value.
                            If the action is not "space", it initiates player movement.
        state (GameState): The current game state, containing all game elements, rules,
                           and properties that may be modified during the state update.

    Returns:
        GameState: The updated game state after processing player movements, automatic movers,
                   rule interpretation, and win condition checks.

    Process:
        - If the action is not "Wait", it moves the player in the specified direction.
        - Move other objects for which a rule "x-is-move" exists.
        - Reinterpret game rules if any moved objects represent words or keywords.
        - Resets lazy evaluation properties in the game state.
    """
    moved_objects = []

    if action != Direction.Wait:
        move_players(action, moved_objects, state)

    move_auto_movers(moved_objects, state)

    for moved_object in moved_objects:
        if is_word(moved_object) or is_key_word(moved_object):
            interpret_rules(state)
            break

    state.lazy_evaluation_properties = dict()
    return state


def can_move(game_obj: GameObj, action: Direction,
             om: List[List[Union[str, GameObj]]], bm: List[List[Union[str, GameObj]]],
             moved_objs: List, players, pushables, phys, sort_phys):
    # move objects only once, check if object has already been moved
    if game_obj in moved_objs:
        return False
    if game_obj == " ":
        return False
    if not game_obj.is_movable:
        return False

    object_at_target = ' '
    # check for the given movement direction if anything (border or out of bounce) forbids movement
    if action == Direction.Up:
        if game_obj.y - 1 < 0:
            return False
        if bm[game_obj.y - 1][game_obj.x] == '_':
            return False
        object_at_target = om[game_obj.y - 1][game_obj.x]
    elif action == Direction.Down:
        if game_obj.y + 1 >= len(bm):
            return False
        if bm[game_obj.y + 1][game_obj.x] == '_':
            return False
        object_at_target = om[game_obj.y + 1][game_obj.x]
    elif action == Direction.Left:
        if game_obj.x - 1 < 0:
            return False
        if bm[game_obj.y][game_obj.x - 1] == '_':
            return False
        object_at_target = om[game_obj.y][game_obj.x - 1]
    elif action == Direction.Right:
        if game_obj.x + 1 >= len(bm[0]):
            return False
        if bm[game_obj.y][game_obj.x + 1] == '_':
            return False
        object_at_target = om[game_obj.y][game_obj.x + 1]

    # check the target position
    if object_at_target == ' ':  # empty tile allows for movement
        return True
    if object_at_target.is_stopped:  # if target position is blocked and cannot be moved itself, stop movement
        return False
    if object_at_target.is_movable:  # if target position is blocked and movable, check if movement is possible
        # check recursively if object at target position can be pushed
        if object_at_target in pushables:
            return move_obj(object_at_target, action, om, bm, moved_objs, players, pushables, phys, sort_phys)
        # required to allow killables to move to player positions
        elif object_at_target in players and game_obj not in players:
            return True
        # move two players at the same time (was missing in the original framework implementation
        # todo: not sure if this rule cannot be implemented somehow else or should receive a higher precedence
        elif object_at_target in players and game_obj in players:
            if game_obj.name == object_at_target.name:
                return move_obj_merge(object_at_target, action, om, bm, moved_objs, players, pushables, phys, sort_phys)
            else:
                return move_obj(object_at_target, action, om, bm, moved_objs, players, pushables, phys, sort_phys)
        # if object at target position is a physical object it can only be pushed if it is also a pushable
        elif object_at_target.object_type == GameObjectType.Physical and object_at_target not in pushables:
            return False
        # if both objects are movable and physical, move them together
        elif ((game_obj.is_movable or object_at_target.is_movable) and
              game_obj.object_type == GameObjectType.Physical and
              object_at_target.object_type == GameObjectType.Physical):
            return True
        # if both objects are of the same type and player objects and physical objects
        # -> potentially merge them or move them both
        elif (game_obj.name == object_at_target.name and game_obj in players and
              is_phys(object_at_target) and is_phys(game_obj)):
            return move_obj_merge(object_at_target, action, om, bm, moved_objs, players, pushables, phys, sort_phys)

        else:
            return move_obj(object_at_target, action, om, bm, moved_objs, players, pushables, phys, sort_phys)

    if not object_at_target.is_stopped and not object_at_target.is_movable:
        return True

    return True


def _execute_move(game_obj: GameObj, direction: Direction, om: List[List[Union[str, GameObj]]], moved_objs: List[GameObj]):
    """
    Executes a move for a game object in a specified direction, updating its position
    and marking it as moved.

    Args:
        game_obj (GameObj): The game object to be moved.
        direction (Direction): The direction in which to move the object.
        om (List[List[Union[str, GameObj]]]): The object map, representing the game grid with
                                              current positions of objects and empty spaces.
        moved_objs (List[GameObj]): A list to track objects that have been moved in the current step.

    Returns:
        bool: True after successfully executing the move.
    """
    # free up the previous position
    om[game_obj.y][game_obj.x] = ' '

    # move object towards chosen direction
    if direction == Direction.Up:
        game_obj.y -= 1
    elif direction == Direction.Down:
        game_obj.y += 1
    elif direction == Direction.Left:
        game_obj.x -= 1
    elif direction == Direction.Right:
        game_obj.x += 1
    om[game_obj.y][game_obj.x] = game_obj
    game_obj.dir = direction

    # add object to moved objects
    moved_objs.append(game_obj)
    return True


def move_obj(game_obj: GameObj, direction: Direction,
             om: List[List[Union[str, GameObj]]], bm: List[List[Union[str, GameObj]]],
             moved_objs: List, players, pushables, phys, sort_phys: Dict):
    """
    Attempts to move a game object in a specified direction if possible, and executes
    the move if successful.

    Args:
        game_obj (GameObj): The game object to be moved.
        direction (Direction): The direction in which to attempt moving the object.
        om (List[List[Union[str, GameObj]]]): The object map, representing the game grid.
        bm (List[List[Union[str, GameObj]]]): The boundary map, representing game boundaries.
        moved_objs (List): A list to track objects that have been moved in the current step.
        players: The list of player-controlled objects.
        pushables: The list of objects that can be pushed.
        phys: The list of physical objects in the game.
        sort_phys: A dictionary of objects per type.

    Returns:
        bool: True if the move was executed successfully; False otherwise.

    Process:
        - Checks if the object can be moved in the specified direction based on game rules.
        - If possible, calls `_execute_move` to move the object and update its state.
        - Otherwise, returns False.
    """
    if can_move(game_obj, direction, om, bm, moved_objs, players, pushables, phys, sort_phys):
        return _execute_move(game_obj, direction, om, moved_objs)
    else:
        return False


def move_obj_merge(o: GameObj, direction: Direction,
                   om: List[List[Union[str, GameObj]]], bm: List[List[Union[str, GameObj]]],
                   moved_objs: List, players, pushables, phys, sort_phys):
    if can_move(o, direction, om, bm, moved_objs, players, pushables, phys, sort_phys):
        return _execute_move(o, direction, om, moved_objs)
    else:
        om[o.y][o.x] = ' '
        return True


def move_players(direction: Direction, moved_objects: List, state: GameState):
    players = state.players
    pushables = state.pushables
    phys = state.phys
    sort_phys = state.sort_phys
    killers = state.killers
    sinkers = state.sinkers
    featured = state.featured

    # iterate over all player objects and move them in the designated direction
    for curPlayer in players:
        move_obj(curPlayer, direction, state.obj_map, state.back_map, moved_objects,
                 players, pushables, phys, sort_phys)

    # remove objects according to various rules
    destroy_objs(killed(players, killers), state)
    destroy_objs(drowned(phys, sinkers), state)
    destroy_objs(bad_feats(featured, sort_phys), state)


def move_auto_movers(mo: List, state: GameState):
    automovers = state.auto_movers
    om = state.obj_map
    bm = state.back_map
    players = state.players
    pushables = state.pushables
    phys = state.phys
    sort_phys = state.sort_phys
    killers = state.killers
    sinkers = state.sinkers
    featured = state.featured

    # iterate over all automovers and move them towards their current direction
    for curAuto in automovers:
        m = move_obj(curAuto, curAuto.dir, om, bm, mo, players, pushables, phys, sort_phys)
        if not m:
            curAuto.dir = Direction.opposite(curAuto.dir)  # walk towards the opposite direction

    # remove objects according to various rules
    destroy_objs(killed(players, killers), state)
    destroy_objs(drowned(phys, sinkers), state)
    destroy_objs(bad_feats(featured, sort_phys), state)


def assign_map_objs(game_state: GameState):
    """
    Populate the game state with objects from the object map.
    Objects can be physical (like "baba" representing the character) or
    word-based (like words in the rule "Baba is You").

    :param game_state: Current game state.
    :return: Boolean indicating success or failure.
    """
    game_state.sort_phys = {}
    game_state.phys = []
    game_state.words = []
    game_state.is_connectors = []

    game_map = game_state.obj_map
    phys = game_state.phys
    words = game_state.words
    sort_phys = game_state.sort_phys
    is_connectors = game_state.is_connectors

    if len(game_map) == 0:
        print("ERROR: Map not initialized yet")
        return False

    for r in range(len(game_map)):
        for c in range(len(game_map[0])):
            character = game_map[r][c]
            object_name = character_to_name[character]
            if "_" not in object_name:
                continue
            base_obj, word_type = object_name.split("_")

            # retrieve word-based objects
            if word_type == "word":
                if base_obj + "_obj" not in name_to_character:
                    # create keyword object
                    w = GameObj.create_keyword_object(base_obj, character, None, c, r)
                else:
                    w = GameObj.create_word_object(base_obj, character, None, c, r)

                # keep track of special key-words
                words.append(w)
                if base_obj == "is":
                    is_connectors.append(w)

                # replace character with object
                game_map[r][c] = w

            # retrieve physical-based objects
            elif word_type == "obj":
                # create object
                o = GameObj.create_physical_object(base_obj, character, c, r)

                # keep track of objects
                phys.append(o)
                if base_obj not in sort_phys:
                    sort_phys[base_obj] = []
                sort_phys[base_obj].append(o)

                # replace character with object
                game_map[r][c] = o


def init_empty_map(m):
    """
    Initialize an empty map based on the dimensions of the input map.

    :param m: The input map for which an empty map will be created.
    :return: A 2D list representing an empty map.
    """
    new_map = []
    for r in range(len(m)):
        new_row = []
        for c in range(len(m[0])):
            new_row.append(' ')

        new_map.append(new_row)
    return new_map


def split_map(m):
    """
    Split the map into two layers: background map and object map.

    :param m: The input 2D map of characters.
    :return: Tuple (background_map, object_map).
    """
    background_map = init_empty_map(m)
    object_map = init_empty_map(m)
    for r in range(len(m)):
        for c in range(len(m[0])):
            map_character = m[r][c]
            parts = character_to_name[map_character].split("_")

            # background
            if len(parts) == 1:
                background_map[r][c] = map_character
                object_map[r][c] = ' '
            # object
            else:
                background_map[r][c] = ' '
                object_map[r][c] = map_character

    return background_map, object_map


def double_map_to_string(object_map: List[List[Union[str, GameObj]]], background_map: List[List[Union[str, GameObj]]]):
    """
    Convert two 2D maps (object and background) into a combined string representation.

    :param object_map: A 2D list representing the object map.
    :param background_map: A 2D list representing the background map.
    :return: A string representation of the combined maps.
    """
    map_string = ""
    for row in range(len(object_map)):
        for column in range(len(object_map[0])):
            game_object = object_map[row][column]
            background = background_map[row][column]
            if row == 0 or column == 0 or row == len(object_map)-1 or column == len(object_map[0])-1:
                map_string += "_"
            elif game_object == " " and background == " ":
                map_string += "."
            elif game_object == " ":
                map_string += name_to_character[background.name + ("_word" if is_word(background) or is_key_word(game_object) else "_obj")]
            else:
                map_string += name_to_character[game_object.name + ("_word" if is_word(game_object) or is_key_word(game_object) else "_obj")]
        map_string += "\n"
    map_string = map_string.rstrip("\n")  # Remove the trailing newline
    return map_string


def map_to_string(game_map: List[List[str]]):
    """
    Generate a printable version of the map by converting it into a comma-separated string.

    :param game_map: A 2D list representing the game map.
    :return: A string representing the map.
    """
    map_arr = []
    for r in range(len(game_map)):
        row = ""
        for c in range(len(game_map[r])):
            row += game_map[r][c]
        map_arr.append(row)

    map_str = ""
    for row in map_arr:
        map_str += ",".join(row) + "\n"

    return map_str


# turns a string object back into a 2d array
def parse_map(map_string: str) -> List[List[str]]:
    """
    Parse a string into a 2D map.

    :param map_string: A string representing the map (e.g., '.' for empty, characters for objects).
    :return: A 2D list representing the parsed map.
    """
    new_map = []
    rows = map_string.split("\n")
    for row in rows:
        row = row.replace(".", " ")
        new_map.append(list(row))
    return new_map


def parse_map_wh(ms: str, w: int, h: int) -> List[List[str]]:
    """
    Parse a string map into a 2D list with specific width and height.

    :param ms: A string representation of the map.
    :param w: Width of the map.
    :param h: Height of the map.
    :return: A 2D list representing the parsed map.
    """
    new_map = []
    for r in range(h):
        row = ms[(r*w):(r*w) + w].replace(".", " ")
        new_map.append(list(row))
    return new_map


def make_level(game_map: List[List[str]]) -> GameState:
    game_state = GameState()
    clear_level(game_state)

    game_state.orig_map = game_map

    maps = split_map(game_state.orig_map)
    game_state.back_map = maps[0]
    game_state.obj_map = maps[1]

    assign_map_objs(game_state)
    interpret_rules(game_state)

    return game_state


def obj_at_pos(x: int, y: int, om: List[List[GameObj]]):
    """
    Get the object at a specific position in the object map.

    :param x: X coordinate.
    :param y: Y coordinate.
    :param om: The object map (2D list).
    :return: The object at the specified coordinates.
    """
    return om[y][x]


def is_word(e: Union[str, GameObj]):
    """
    Check if a given object is a word.

    :param e: The object or string to check.
    :return: True if the object is a word, False otherwise.
    """
    if isinstance(e, str):
        return False
    return e.object_type == GameObjectType.Word


def is_key_word(e: Union[str, GameObj]):
    """
    Check if a given object is a keyword (a word that doesn't have a corresponding object)
    e.g. is, win, melt, ...

    :param e: The object or string to check.
    :return: True if the object is a word, False otherwise.
    """
    if isinstance(e, str):
        return False
    return e.object_type == GameObjectType.Keyword


def is_phys(e: Union[str, GameObj]):
    """
    Check if a given object is a physical object.

    :param e: The object or string to check.
    :return: True if the object is a physical object, False otherwise.
    """
    if isinstance(e, str):
        return False
    return e.object_type == GameObjectType.Physical


def add_active_rules(word_a: GameObj, word_b: GameObj, is_connector: GameObj, rules: List, rule_objs: List):
    """
    Add active rules based on the word objects and their connectors (like "is" in "Baba is You").

    :param word_a: The first word object.
    :param word_b: The second word object.
    :param is_connector: The connector word ("is").
    :param rules: List of current active rules.
    :param rule_objs: List of rule objects in the game.
    """
    if (is_word(word_a) and not is_key_word(word_a)) and (is_word(word_b) or is_key_word(word_b)):
        # Add a new rule if not already made
        r = f"{word_a.name}-{is_connector.name}-{word_b.name}"
        if r not in rules:
            rules.append(r)

        # Add as a rule object
        rule_objs.append(word_a)
        rule_objs.append(is_connector)
        rule_objs.append(word_b)


def interpret_rules(game_state: GameState):
    """
    Interpret and apply the rules based on the current game state and the words in the map.

    :param game_state: The current game state.
    """
    # Reset the rules (since a word object was changed)
    game_state.rules = []
    game_state.rule_objs = []

    # Get all relevant fields
    om = game_state.obj_map
    bm = game_state.back_map
    is_connectors = game_state.is_connectors
    rules = game_state.rules
    rule_objs = game_state.rule_objs
    sort_phys = game_state.sort_phys
    phys = game_state.phys

    # iterate all is-connectors to identify rules
    for is_connector in is_connectors:
        # Horizontal position
        word_a = obj_at_pos(is_connector.x - 1, is_connector.y, om)
        word_b = obj_at_pos(is_connector.x + 1, is_connector.y, om)
        add_active_rules(word_a, word_b, is_connector, rules, rule_objs)

        # Vertical position
        word_c = obj_at_pos(is_connector.x, is_connector.y - 1, om)
        word_d = obj_at_pos(is_connector.x, is_connector.y + 1, om)
        add_active_rules(word_c, word_d, is_connector, rules, rule_objs)

    # Interpret sprite changing rules
    transformation(om, bm, rules, sort_phys, phys)

    # Reset the objects
    reset_all(game_state)


# Check if array contains string with a substring
def has(arr: List, ss: str):
    """
    Check if any item in the list contains a specific substring.

    :param arr: List of strings.
    :param ss: Substring to search for.
    :return: True if any string in the list contains the substring, False otherwise.
    """
    for item in arr:
        if ss in item:
            return True
    return False


def clear_level(game_state):
    """
    Clear the current level by resetting the game state.

    :param game_state: The current game state.
    """
    game_state.reset()


# Function resetAll
def reset_all(game_state: GameState):
    """
    Reset all object properties and reapply rules in the current game state.

    :param game_state: The current game state.
    """
    # Reset the objects
    reset_obj_props(game_state.phys)
    set_players(game_state)
    set_auto_movers(game_state)
    set_wins(game_state)
    set_pushes(game_state)
    set_stops(game_state)
    set_kills(game_state)
    set_sinks(game_state)
    set_overlaps(game_state)
    set_features(game_state)


# Set the player objects
def set_players(game_state: GameState):
    """
    Set the player objects based on the "you" rule.

    :param game_state: The current game state.
    """
    players = game_state.players
    rules = game_state.rules
    sort_phys = game_state.sort_phys

    players.clear()
    for rule in rules:
        if "you" in rule:
            players.extend(sort_phys.get(rule.split("-")[0], []))

    # Make all the players movable
    for player in players:
        player.is_movable = True


def set_auto_movers(game_state: GameState):
    """
    Set the automatically moving objects based on the "move" rule.

    :param game_state: The current game state.
    """
    game_state.auto_movers = []
    auto_movers = game_state.auto_movers
    rules = game_state.rules
    sort_phys = game_state.sort_phys

    for rule in rules:
        if "move" in rule:
            auto_movers.extend(sort_phys.get(rule.split("-")[0], []))

    # Make all the NPCs movable and set default direction
    for auto_mover in auto_movers:
        auto_mover.is_movable = True
        if auto_mover.dir == Direction.Undefined:
            auto_mover.dir = Direction.Right


# Set the winning objects
def set_wins(game_state: GameState):
    """
    Set the winnable objects based on the "win" rule.

    :param game_state: The current game state.
    """
    game_state.winnables = []
    winnables = game_state.winnables
    rules = game_state.rules
    sort_phys = game_state.sort_phys

    for rule in rules:
        if "win" in rule:
            winnables.extend(sort_phys.get(rule.split("-")[0], []))


# Set the pushable objects
def set_pushes(game_state: GameState):
    """
    Set the pushable objects based on the "push" rule.

    :param game_state: The current game state.
    """
    game_state.pushables = []
    pushables = game_state.pushables
    rules = game_state.rules
    sort_phys = game_state.sort_phys

    for rule in rules:
        if "push" in rule:
            pushables.extend(sort_phys.get(rule.split("-")[0], []))

    # Make all the pushables movable
    for pushable in pushables:
        pushable.is_movable = True


# Set the stopping objects
def set_stops(game_state: GameState):
    """
    Set the stopping objects based on the "stop" rule.

    :param game_state: The current game state.
    """
    game_state.stoppables = []
    stoppables = game_state.stoppables
    rules = game_state.rules
    sort_phys = game_state.sort_phys

    for rule in rules:
        if "stop" in rule:
            stoppables.extend(sort_phys.get(rule.split("-")[0], []))

    # Make all the stoppables stopped
    for stoppable in stoppables:
        stoppable.is_stopped = True


# Set the killable objects
def set_kills(game_state: GameState):
    """
    Set the killable objects based on the "kill" rule.

    :param game_state: The current game state.
    """
    game_state.killers = []
    killers = game_state.killers
    rules = game_state.rules
    sort_phys = game_state.sort_phys

    for rule in rules:
        if "kill" in rule or "sink" in rule:
            killers.extend(sort_phys.get(rule.split("-")[0], []))


# Set the sinkable objects
def set_sinks(game_state: GameState):
    """
    Set the sinkable objects based on the "sink" rule.

    :param game_state: The current game state.
    """
    game_state.sinkers = []
    sinkers = game_state.sinkers
    rules = game_state.rules
    sort_phys = game_state.sort_phys

    for rule in rules:
        if "sink" in rule:
            sinkers.extend(sort_phys.get(rule.split("-")[0], []))


# Set objects that are unmovable but not stoppable
def set_overlaps(game_state: GameState):
    """
    Set the objects that can overlap and remove the ones that can't.

    :param game_state: The current game state.
    """
    game_state.overlaps = []
    game_state.unoverlaps = []

    bm = game_state.back_map
    om = game_state.obj_map
    overlaps = game_state.overlaps
    unoverlaps = game_state.unoverlaps
    phys = game_state.phys
    words = game_state.words

    for p in phys:
        if not p.is_movable and not p.is_stopped:
            overlaps.append(p)
            bm[p.y][p.x] = p
            om[p.y][p.x] = ' '
        else:
            unoverlaps.append(p)
            om[p.y][p.x] = p
            bm[p.y][p.x] = ' '

    unoverlaps.extend(words)
    # Words will always be in the object layer
    for w in words:
        om[w.y][w.x] = w


# Check if an object is overlapping another
def overlapped(a: GameObj, b: GameObj):
    """
    Check if two game objects overlap.

    :param a: First game object.
    :param b: Second game object.
    :return: True if the objects overlap, False otherwise.
    """
    return a == b or (a.x == b.x and a.y == b.y)


# Reset all the properties of every object
def reset_obj_props(phys: List[GameObj]):
    """
    Reset the properties of physical objects.

    :param phys: List of physical game objects.
    """
    for p in phys:
        p.is_movable = False
        p.is_stopped = False
        p.feature = ""


# Check if the player has stepped on a kill object
def killed(players, killers):
    """
    Check if any player has been killed by a killer object.

    :param players: List of player objects.
    :param killers: List of killer objects.
    :return: List of killed objects.
    """
    dead = []
    for player in players:
        for killer in killers:
            if overlapped(player, killer):
                dead.append([player, killer])
                # Todo, I assume we can break here
    return dead


# Check if an object has drowned
def drowned(phys, sinkers):
    """
    Check if any objects have drowned by falling into sinkers.

    :param phys: List of physical objects.
    :param sinkers: List of sinker objects.
    :return: List of drowned objects.
    """
    dead = []
    for p in phys:
        for sinker in sinkers:
            if p != sinker and overlapped(p, sinker):
                dead.append([p, sinker])
    return dead


# Destroy objects
def destroy_objs(dead, game_state: GameState):
    """
    Remove dead objects (killed or drowned) from the game state.

    :param dead: List of objects to be removed.
    :param game_state: The current game state.
    """
    bm = game_state.back_map
    om = game_state.obj_map
    phys = game_state.phys
    sort_phys = game_state.sort_phys

    for p, o in dead:
        # Remove reference of the player and the murder object
        if p in phys:
            phys.remove(p)
        if o in phys:
            phys.remove(o)
        if p in sort_phys[p.name]:
            sort_phys[p.name].remove(p)
        if o in sort_phys[o.name]:
            sort_phys[o.name].remove(o)

        # Clear the space
        bm[o.y][o.x] = ' '
        om[p.y][p.x] = ' '

    # Reset the objects
    if dead:
        reset_all(game_state)


# Check if the player has entered a win state
def check_win(game_state: GameState) -> bool:
    """
    Check if any player object has reached a winning state.

    :param game_state: The current game state.
    :return: True if the game has been won, False otherwise.
    """
    if "win" not in game_state.lazy_evaluation_properties:
        for player in game_state.players:
            for winnable in game_state.winnables:
                if overlapped(player, winnable):
                    game_state.lazy_evaluation_properties["win"] = True
                    return True
        else:
            game_state.lazy_evaluation_properties["win"] = False
    return game_state.lazy_evaluation_properties["win"]


def is_obj_word(w: str):
    """
    Check if a given word string represents a word object in the game.

    :param w: Word string.
    :return: True if it's a word object, False otherwise.
    """
    if w+"_obj" in name_to_character:
        return True


# Turns all of one object type into all of another object type
def transformation(om, bm, rules, sort_phys, phys):
    """
    Apply transformation rules (e.g., "Baba is Flag") to change object types in the game state.

    :param om: Object map.
    :param bm: Background map.
    :param rules: List of active rules.
    :param sort_phys: Dictionary of sorted physical objects by type.
    :param phys: List of physical objects.
    """
    # x-is-x takes priority and makes it immutable
    x_is_x = []
    for rule in rules:
        parts = rule.split("-")
        if parts[0] == parts[2] and parts[0] in sort_phys:
            x_is_x.append(parts[0])

    # Transform sprites (x-is-y == x becomes y)
    for rule in rules:
        parts = rule.split("-")
        if parts[0] not in x_is_x and is_obj_word(parts[0]) and is_obj_word(parts[2]):
            all_objs = sort_phys.get(parts[0], []).copy()
            for obj in all_objs:
                change_sprite(obj, parts[2], om, bm, phys, sort_phys)


# Changes a sprite from one thing to another
def change_sprite(o, w, om, bm, phys, sort_phys):
    """
    Change the sprite of a game object to a different type based on rules.

    :param o: Original game object.
    :param w: New object type name.
    :param om: Object map.
    :param bm: Background map.
    :param phys: List of physical objects.
    :param sort_phys: Dictionary of sorted physical objects by type.
    """
    character = name_to_character[w + "_obj"]
    o2 = GameObj.create_physical_object(w, character, o.x, o.y)
    phys.append(o2)  # in with the new...

    # Replace object on obj_map/back_map
    if obj_at_pos(o.x, o.y, om) == o:
        om[o.y][o.x] = o2
    else:
        bm[o.y][o.x] = o2

    # Add to the list of objects under a certain name
    if w not in sort_phys:
        sort_phys[w] = [o2]
    else:
        sort_phys[w].append(o2)

    phys.remove(o)  # ...out with the old
    sort_phys[o.name].remove(o)


# Adds a feature to word groups based on ruleset
def set_features(game_state: GameState):
    """
    Set features (e.g., "hot", "melt") on objects based on the rules in the game state.

    :param game_state: The current game state
    """
    game_state.featured = {}
    featured = game_state.featured
    rules = game_state.rules
    sort_phys = game_state.sort_phys

    # Add a feature to the sprites (x-is-y == x has y feature)
    for rule in rules:
        parts = rule.split("-")
        # Add a feature to a sprite set
        if parts[2] in features and parts[0] not in features and parts[0] in sort_phys:
            # No feature set yet so make a new array
            if parts[2] not in featured:
                featured[parts[2]] = [parts[0]]
            else:
                # Or append it
                featured[parts[2]].append(parts[0])

            # Set the physical objects' features
            ps = sort_phys[parts[0]]
            for p in ps:
                p.feature = parts[2]


# Similar to killed() check if feat pairs are overlapped and destroy both
def bad_feats(featured, sort_phys):
    baddies = []

    for pair in featPairs:
        # Check if both features have object sets
        if pair[0] in featured and pair[1] in featured:
            # Get all the sprites for each group
            a_set = []
            b_set = []

            for feature in featured[pair[0]]:
                a_set.extend(sort_phys[feature])

            for feature in featured[pair[1]]:
                b_set.extend(sort_phys[feature])

            # Check for overlaps between objects
            for a in a_set:
                for b in b_set:
                    if overlapped(a, b):
                        baddies.append([a, b])

    return baddies







##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################


# Returns True if the player(s) in next_state are closer to a win object than in prev_state
def state_brings_player_closer_to_win(prev_state: GameState, next_state: GameState) -> bool:
    """
    Returns True if the minimum Manhattan distance from any player to any win object
    in next_state is less than in prev_state.

    :param prev_state: The game state before the move.
    :param next_state: The game state after the move.
    :return: True if the player(s) are closer to a win object, False otherwise.
    """
    def min_player_win_distance(state):
        if not state.players or not state.winnables:
            return float('inf')
        min_dist = float('inf')
        for player in state.players:
            for win_obj in state.winnables:
                dist = abs(player.x - win_obj.x) + abs(player.y - win_obj.y)
                if dist < min_dist:
                    min_dist = dist
        return min_dist
    return min_player_win_distance(next_state) < min_player_win_distance(prev_state)





# Returns True if a move creates a new positive rule (e.g., a new 'is you', 'is win', etc.)
def move_creates_positive_rule(prev_state: 'GameState', next_state: 'GameState') -> bool:
    """
    Returns True if next_state.rules contains a new 'is' rule with a positive effect (win, you, push, open, etc.)
    that was not present in prev_state.rules. Only rules that actually affect objects on the map are considered.
    """
    positive_keywords = {"win", "you", "push", "open", "shut", "hot", "melt", "move"}
    prev_rules = set(prev_state.rules)
    next_rules = set(next_state.rules)
    new_rules = next_rules - prev_rules
    # Only consider rules that are of the form 'object-is-keyword' and the object exists in next_state.phys
    phys_names = set(obj.name for obj in next_state.phys)
    for rule in new_rules:
        parts = rule.split("-")
        if len(parts) == 3 and parts[1] == "is" and parts[2] in positive_keywords and parts[0] in phys_names:
            return True
    return False

# #AGGIUNTA
# def move_creates_win_condition(prev_state: 'GameState', next_state: 'GameState') -> bool:
#     """
#     Returns True if the move creates a winning condition by having a player overlap with a winnable object.
#     """
#     if not next_state.players or not next_state.winnables:
#         return False
    
#     for player in next_state.players:
#         for winnable in next_state.winnables:
#             if player.x == winnable.x and player.y == winnable.y:
#                 return True
#     return False

# #AGGIUNTA
# def move_creates_player_rule(prev_state: 'GameState', next_state: 'GameState') -> bool:
#     """
#     Returns True if a new "X is you" rule is created, potentially giving the player more control.
#     """
#     prev_rules = set(prev_state.rules)
#     next_rules = set(next_state.rules)
#     new_rules = next_rules - prev_rules
    
#     phys_names = set(obj.name for obj in next_state.phys)
#     for rule in new_rules:
#         parts = rule.split("-")
#         if len(parts) == 3 and parts[1] == "is" and parts[2] == "you" and parts[0] in phys_names:
#             return True
#     return False

# # AGGIUNTA
# def move_creates_win_rule(prev_state: 'GameState', next_state: 'GameState') -> bool:
#     """
#     Returns True if a new "X is win" rule is created, potentially creating new winning objectives.
#     """
#     prev_rules = set(prev_state.rules)
#     next_rules = set(next_state.rules)
#     new_rules = next_rules - prev_rules
    
#     phys_names = set(obj.name for obj in next_state.phys)
#     for rule in new_rules:
#         parts = rule.split("-")
#         if len(parts) == 3 and parts[1] == "is" and parts[2] == "win" and parts[0] in phys_names:
#             return True
#     return False

# #AGGIUNTA
# def move_removes_obstacle_rule(prev_state: 'GameState', next_state: 'GameState') -> bool:
#     """
#     Returns True if a rule that makes objects block movement (stop) is removed.
#     """
#     prev_rules = set(prev_state.rules)
#     next_rules = set(next_state.rules)
#     removed_rules = prev_rules - next_rules
    
#     phys_names = set(obj.name for obj in prev_state.phys)
#     for rule in removed_rules:
#         parts = rule.split("-")
#         if len(parts) == 3 and parts[1] == "is" and parts[2] == "stop" and parts[0] in phys_names:
#             return True
#     return False

# Funzioni euristiche aggiuntive per migliorare l'algoritmo MCTS in Baba is You

def move_creates_win_condition(prev_state: 'GameState', next_state: 'GameState') -> bool:
    """
    Returns True if the move creates a winning condition by having a player overlap with a winnable object.
    This is the most important heuristic as it represents achieving the goal.
    """
    if not next_state.players or not next_state.winnables:
        return False
    
    for player in next_state.players:
        for winnable in next_state.winnables:
            if player.x == winnable.x and player.y == winnable.y:
                return True
    return False

def move_creates_player_rule(prev_state: 'GameState', next_state: 'GameState') -> bool:
    """
    Returns True if a new "X is you" rule is created, potentially giving the player more control.
    """
    prev_rules = set(prev_state.rules)
    next_rules = set(next_state.rules)
    new_rules = next_rules - prev_rules
    
    phys_names = set(obj.name for obj in next_state.phys)
    for rule in new_rules:
        parts = rule.split("-")
        if len(parts) == 3 and parts[1] == "is" and parts[2] == "you" and parts[0] in phys_names:
            return True
    return False

def move_creates_win_rule(prev_state: 'GameState', next_state: 'GameState') -> bool:
    """
    Returns True if a new "X is win" rule is created, potentially creating new winning objectives.
    """
    prev_rules = set(prev_state.rules)
    next_rules = set(next_state.rules)
    new_rules = next_rules - prev_rules
    
    phys_names = set(obj.name for obj in next_state.phys)
    for rule in new_rules:
        parts = rule.split("-")
        if len(parts) == 3 and parts[1] == "is" and parts[2] == "win" and parts[0] in phys_names:
            return True
    return False

def move_removes_obstacle_rule(prev_state: 'GameState', next_state: 'GameState') -> bool:
    """
    Returns True if a rule that makes objects block movement (stop) is removed.
    """
    prev_rules = set(prev_state.rules)
    next_rules = set(next_state.rules)
    removed_rules = prev_rules - next_rules
    
    phys_names = set(obj.name for obj in prev_state.phys)
    for rule in removed_rules:
        parts = rule.split("-")
        if len(parts) == 3 and parts[1] == "is" and parts[2] == "stop" and parts[0] in phys_names:
            return True
    return False

def move_creates_push_rule(prev_state: 'GameState', next_state: 'GameState') -> bool:
    """
    Returns True if a new "X is push" rule is created, making objects movable.
    """
    prev_rules = set(prev_state.rules)
    next_rules = set(next_state.rules)
    new_rules = next_rules - prev_rules
    
    phys_names = set(obj.name for obj in next_state.phys)
    for rule in new_rules:
        parts = rule.split("-")
        if len(parts) == 3 and parts[1] == "is" and parts[2] == "push" and parts[0] in phys_names:
            return True
    return False

def move_brings_rule_words_closer(prev_state: 'GameState', next_state: 'GameState') -> bool:
    """
    Returns True if word objects that could form beneficial rules are brought closer together.
    Specifically looks for words that could form "X is you" or "X is win" rules.
    """
    def get_rule_potential_distance(state):
        if not state.words:
            return float('inf')
        
        min_dist = float('inf')
        beneficial_patterns = [("you", "win"), ("push", "you"), ("push", "win")]
        
        # Find potential rule formations
        for word1 in state.words:
            if word1.obj in ["you", "win", "push"]:
                for word2 in state.words:
                    if word2.obj == "is":
                        for word3 in state.words:
                            if word3.obj != word1.obj and word3.obj != "is":
                                # Check if this would form a beneficial rule
                                pattern = (word1.obj, word3.obj)
                                reverse_pattern = (word3.obj, word1.obj)
                                
                                if pattern in beneficial_patterns or reverse_pattern in beneficial_patterns:
                                    # Calculate distance for potential rule formation
                                    dist = abs(word1.x - word2.x) + abs(word1.y - word2.y) + \
                                           abs(word2.x - word3.x) + abs(word2.y - word3.y)
                                    min_dist = min(min_dist, dist)
        
        return min_dist
    
    prev_dist = get_rule_potential_distance(prev_state)
    next_dist = get_rule_potential_distance(next_state)
    
    return next_dist < prev_dist

def move_avoids_danger(prev_state: 'GameState', next_state: 'GameState') -> bool:
    """
    Returns True if the move increases the minimum distance from players to dangerous objects.
    """
    def min_player_danger_distance(state):
        if not state.players:
            return float('inf')
        
        dangerous_objects = state.killers + state.sinkers
        if not dangerous_objects:
            return float('inf')
        
        min_dist = float('inf')
        for player in state.players:
            for danger in dangerous_objects:
                dist = abs(player.x - danger.x) + abs(player.y - danger.y)
                min_dist = min(min_dist, dist)
        
        return min_dist
    
    prev_danger_dist = min_player_danger_distance(prev_state)
    next_danger_dist = min_player_danger_distance(next_state)
    
    return next_danger_dist > prev_danger_dist

def move_creates_tool_access(prev_state: 'GameState', next_state: 'GameState') -> bool:
    """
    Returns True if the move brings players closer to pushable objects that could be used as tools.
    """
    def min_player_tool_distance(state):
        if not state.players or not state.pushables:
            return float('inf')
        
        min_dist = float('inf')
        for player in state.players:
            for pushable in state.pushables:
                # Don't consider the player itself as a tool
                if pushable not in state.players:
                    dist = abs(player.x - pushable.x) + abs(player.y - pushable.y)
                    min_dist = min(min_dist, dist)
        
        return min_dist
    
    prev_tool_dist = min_player_tool_distance(prev_state)
    next_tool_dist = min_player_tool_distance(next_state)
    
    return next_tool_dist < prev_tool_dist

def move_preserves_control(prev_state: 'GameState', next_state: 'GameState') -> bool:
    """
    Returns True if the move maintains or increases the number of controllable objects.
    """
    return len(next_state.players) >= len(prev_state.players)

def move_improves_rule_formation_potential(prev_state: 'GameState', next_state: 'GameState') -> bool:
    """
    Returns True if the move improves the potential for forming beneficial rules by organizing words.
    """
    def count_aligned_words(state):
        """Count words that are aligned horizontally or vertically in groups of 3."""
        if len(state.words) < 3:
            return 0
        
        aligned_count = 0
        
        # Check horizontal alignments
        for y in range(len(state.obj_map)):
            words_in_row = [w for w in state.words if w.y == y]
            words_in_row.sort(key=lambda w: w.x)
            
            for i in range(len(words_in_row) - 2):
                if (words_in_row[i + 1].x == words_in_row[i].x + 1 and 
                    words_in_row[i + 2].x == words_in_row[i + 1].x + 1):
                    aligned_count += 1
        
        # Check vertical alignments
        for x in range(len(state.obj_map[0])):
            words_in_col = [w for w in state.words if w.x == x]
            words_in_col.sort(key=lambda w: w.y)
            
            for i in range(len(words_in_col) - 2):
                if (words_in_col[i + 1].y == words_in_col[i].y + 1 and 
                    words_in_col[i + 2].y == words_in_col[i + 1].y + 1):
                    aligned_count += 1
        
        return aligned_count
    
    prev_alignment = count_aligned_words(prev_state)
    next_alignment = count_aligned_words(next_state)
    
    return next_alignment > prev_alignment

# Funzione principale per combinare tutte le euristiche
def evaluate_move_heuristic(prev_state: 'GameState', next_state: 'GameState') -> float:
    """
    Combines all heuristic functions into a single score.
    Higher scores indicate better moves.
    """
    score = 0.0
    
    # Victory condition - highest priority
    if move_creates_win_condition(prev_state, next_state):
        score += 1000.0
    
    # Rule creation - high priority
    if move_creates_positive_rule(prev_state, next_state):
        score += 50.0
    
    if move_creates_player_rule(prev_state, next_state):
        score += 80.0
    
    if move_creates_win_rule(prev_state, next_state):
        score += 70.0
    
    if move_creates_push_rule(prev_state, next_state):
        score += 30.0
    
    # Rule removal - medium priority
    if destructive_rule_removed(prev_state, next_state):
        score += 40.0
    
    if move_removes_obstacle_rule(prev_state, next_state):
        score += 35.0
    
    # Movement and positioning - lower priority
    if state_brings_player_closer_to_win(prev_state, next_state):
        score += 20.0
    
    if move_brings_rule_words_closer(prev_state, next_state):
        score += 25.0
    
    if move_creates_tool_access(prev_state, next_state):
        score += 15.0
    
    if move_improves_rule_formation_potential(prev_state, next_state):
        score += 10.0
    
    # Safety considerations
    if move_avoids_danger(prev_state, next_state):
        score += 30.0
    
    if move_preserves_control(prev_state, next_state):
        score += 25.0
    
    # Negative scoring for potentially bad moves
    if something_sank(prev_state, next_state):
        score -= 20.0
    
    # Bonus for meaningful interaction
    if something_was_pushed(prev_state, next_state):
        score += 5.0
    
    return score


# Check if the player is in a deadlock state (cannot move, has drowned, or has died)
def check_player_deadlocked(game_state: GameState) -> bool:
    """
    Returns True if all player objects are either dead (not present), have drowned, or cannot make any legal move.

    :param game_state: The current game state.
    :return: True if all players are deadlocked, False otherwise.
    """
    # If there are no players left, they have all died
    if not game_state.players:
        return True

    # If any player is overlapping with a sinker, or killed, consider deadlocked
    for player in game_state.players:
        for sinker in game_state.sinkers:
            if player != sinker and overlapped(player, sinker):
                return True
    # Check if any player is overlapping with a killer
    for player in game_state.players:
        for killer in game_state.killers:
            if overlapped(player, killer):
                return True

    # Check if all players are unable to move in any direction
    from enum import Enum
    class _Direction(Enum):
        Left = 'l'
        Right = 'r'
        Up = 'u'
        Down = 'd'
        Wait = 's'
        Undefined = None

    def can_player_move(player, state):
        for direction in [getattr(state, 'Direction', _Direction).Left,
                          getattr(state, 'Direction', _Direction).Right,
                          getattr(state, 'Direction', _Direction).Up,
                          getattr(state, 'Direction', _Direction).Down]:
            # Try to move the player in each direction
            # Use the can_move function if available
            try:
                if can_move(player, direction, state.obj_map, state.back_map, [], state.players, state.pushables, state.phys, state.sort_phys):
                    return True
            except Exception:
                continue
        return False

    # If all players cannot move, it's a deadlock
    if all(not can_player_move(player, game_state) for player in game_state.players):
        return True
    return False





# Returns True if any non-player, non-word, non-keyword physical object was pushed (moved position) from prev_state to next_state
def something_was_pushed(prev_state: 'GameState', next_state: 'GameState') -> bool:
    """
    Returns True if any non-player, non-word, non-keyword physical object changed position between prev_state and next_state.
    """
    def is_pushable_nonplayer(obj, state):
        return (
            is_phys(obj)
            and obj not in state.players
            and not is_word(obj)
            and not is_key_word(obj)
        )
    prev_objs = {(obj.name, obj.x, obj.y) for obj in prev_state.phys if is_pushable_nonplayer(obj, prev_state)}
    next_objs = {(obj.name, obj.x, obj.y) for obj in next_state.phys if is_pushable_nonplayer(obj, next_state)}
    # If any object changed position, the sets will differ
    return prev_objs != next_objs




# Returns True if a destructive rule (kill/sink/drown) is removed for any object present in prev_state.phys

def destructive_rule_removed(prev_state: 'GameState', next_state: 'GameState') -> bool:
    """
    Returns True if a destructive rule (e.g., 'x-is-kill', 'x-is-sink', 'x-is-drown', 'x-is-stop')
    is present in prev_state.rules but not in next_state.rules, and the object is present in prev_state.phys.
    """
    destructive_keywords = {"kill", "sink", "drown", "stop"}
    prev_rules = set(prev_state.rules)
    next_rules = set(next_state.rules)
    removed_rules = prev_rules - next_rules
    phys_names = set(obj.name for obj in prev_state.phys)
    for rule in removed_rules:
        parts = rule.split("-")
        if len(parts) == 3 and parts[1] == "is" and parts[2] in destructive_keywords and parts[0] in phys_names:
            return True
    return False




# Returns True if any object and a sinker overlapped and both disappeared (i.e., a sink event occurred)
def move_closer_to_destructive_rule(prev_state: 'GameState', next_state: 'GameState') -> bool:
    """
    Returns True if the minimum Manhattan distance from any player to any object affected by a destructive rule
    (kill, sink, drown, stop) that was present in prev_state.rules but not in next_state.rules is less in next_state than in prev_state.
    Only considers objects present in next_state.phys.
    """
    destructive_keywords = {"kill", "sink", "drown", "stop"}
    prev_rules = set(prev_state.rules)
    next_rules = set(next_state.rules)
    removed_rules = prev_rules - next_rules
    # Find all object names affected by removed destructive rules
    affected_names = set()
    for rule in removed_rules:
        parts = rule.split("-")
        if len(parts) == 3 and parts[1] == "is" and parts[2] in destructive_keywords:
            affected_names.add(parts[0])
    if not affected_names:
        return False
    # Find all objects in next_state.phys with those names
    affected_objs_next = [obj for obj in next_state.phys if obj.name in affected_names]
    affected_objs_prev = [obj for obj in prev_state.phys if obj.name in affected_names]
    if not affected_objs_next or not next_state.players:
        return False
    def min_dist(players, objs):
        min_d = float('inf')
        for player in players:
            for obj in objs:
                dist = abs(player.x - obj.x) + abs(player.y - obj.y)
                if dist < min_d:
                    min_d = dist
        return min_d
    dist_prev = min_dist(prev_state.players, affected_objs_prev) if prev_state.players and affected_objs_prev else float('inf')
    dist_next = min_dist(next_state.players, affected_objs_next)
    return dist_next < dist_prev

def something_sank(prev_state: 'GameState', next_state: 'GameState') -> bool:
    """
    Returns True if, between prev_state and next_state, any sinker and any other object (not a player, not a word/keyword)
    overlapped in prev_state and both are missing in next_state.phys at that position.
    """
    next_phys_set = set((obj.name, obj.x, obj.y) for obj in next_state.phys)
    for sinker in prev_state.sinkers:
        for obj in prev_state.phys:
            if obj is sinker or is_word(obj) or is_key_word(obj) or obj in prev_state.players:
                continue
            if (obj.x, obj.y) == (sinker.x, sinker.y):
                sinker_pos = (sinker.name, sinker.x, sinker.y)
                obj_pos = (obj.name, obj.x, obj.y)
                if sinker_pos not in next_phys_set and obj_pos not in next_phys_set:
                    return True
    return False


##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################
##############################################################################################################################



if __name__ == "__main__":
    pass
