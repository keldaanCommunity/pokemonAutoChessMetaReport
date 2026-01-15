"""Utility functions and constants"""

import os
import math
import random
import requests
from dotenv import load_dotenv
from warnings import simplefilter

# Ignore sklearn warnings
simplefilter(action='ignore', category=FutureWarning)

load_dotenv()


class ColorGenerator:
    """
    Generate visually distinct colors using the golden ratio.
    
    Uses the golden ratio to distribute colors evenly across the hue spectrum,
    with random saturation and lightness variations for visual distinction.
    """
    def __init__(self):
        self.__phi = (1 + math.sqrt(5)) / 2
        self.__x = random.random()

    def __hsl2hex(self, h, s, l):
        """
        Convert HSL (Hue, Saturation, Lightness) color values to hexadecimal format.
        
        Args:
            h (float): Hue value (0-360 degrees)
            s (float): Saturation value (0-1)
            l (float): Lightness value (0-1)
        
        Returns:
            str: Hexadecimal color string (e.g., '#FF5733')
        """
        t = s * min(l, 1-l)
        k1 = (h/30) % 12
        k2 = (8 + h/30) % 12
        k3 = (4 + h/30) % 12
        r = l - t * max(-1, min(1, k1-3, 9-k1))
        g = l - t * max(-1, min(1, k2-3, 9-k2))
        b = l - t * max(-1, min(1, k3-3, 9-k3))
        red = math.floor(r*256)
        green = math.floor(g*256)
        blue = math.floor(b*256)
        return f"#{red:02X}{green:02X}{blue:02X}"

    def next(self):
        """
        Generate and return the next color in the sequence.
        
        Returns:
            str: Hexadecimal color string representing the next color in the golden ratio sequence
        """
        self.__x = (self.__x + self.__phi) % 1
        h = self.__x * 360
        s = (60 + 40 * random.random()) / 100
        l = (50 + s*0.2 * (random.random()*2-1)) / 100
        return self.__hsl2hex(h, s, l)


# Load Pokemon data from API
def load_pokemon_constants():
    """
    Load Pokemon types, items, and related data from the Pokemon Auto Chess API.
    
    Fetches comprehensive game data including: Pokemon list, type system, type triggers,
    and items. Builds a reverse mapping of Pokemon to their types.
    
    Returns:
        dict: Dictionary containing:
            - LIST_POKEMON: List of all Pokemon names
            - TYPE_POKEMON: Mapping of types to their Pokemon
            - TYPE_TRIGGER: Type trigger effects and conditions
            - ITEM: All available items
            - POKEMON_TYPE: Reverse mapping of Pokemon names to their types
            - LIST_TYPE: List of all type names
    """
    try:
        request_pokemons = requests.get('https://pokemon-auto-chess.com/pokemons', timeout=15)
        request_pokemons.raise_for_status()
        LIST_POKEMON = [p for p in request_pokemons.json().values()]
    except Exception as e:
        print(f"Warning: Could not fetch Pokemon list from API: {e}")
        LIST_POKEMON = []

    try:
        types_pokemons = requests.get('https://pokemon-auto-chess.com/types', timeout=15)
        types_pokemons.raise_for_status()
        TYPE_POKEMON = types_pokemons.json()
    except Exception as e:
        print(f"Warning: Could not fetch Pokemon types from API: {e}")
        TYPE_POKEMON = {}

    try:
        trigger = requests.get('https://pokemon-auto-chess.com/types-trigger', timeout=15)
        trigger.raise_for_status()
        TYPE_TRIGGER = trigger.json()
    except Exception as e:
        print(f"Warning: Could not fetch trigger thresholds from API: {e}")
        TYPE_TRIGGER = {}

    try:
        items = requests.get('https://pokemon-auto-chess.com/items', timeout=15)
        items.raise_for_status()
        ITEM = items.json()
    except Exception as e:
        print(f"Warning: Could not fetch items from API: {e}")
        ITEM = {}

    # Get list of types for each pokemon
    POKEMON_TYPE = {}
    for pkm in LIST_POKEMON:
        type_list = []
        for t in TYPE_POKEMON:
            type_name = t.lower()
            type_pokemons = TYPE_POKEMON[t]
            if pkm in type_pokemons:
                type_list.append(type_name)
        POKEMON_TYPE[pkm] = type_list

    LIST_TYPE = [k.lower() for k in TYPE_POKEMON.keys()] if TYPE_POKEMON else []

    return {
        'LIST_POKEMON': LIST_POKEMON,
        'TYPE_POKEMON': TYPE_POKEMON,
        'TYPE_TRIGGER': TYPE_TRIGGER,
        'ITEM': ITEM,
        'POKEMON_TYPE': POKEMON_TYPE,
        'LIST_TYPE': LIST_TYPE,
    }


# Load constants
CONSTANTS = load_pokemon_constants()
LIST_POKEMON = CONSTANTS['LIST_POKEMON']
TYPE_POKEMON = CONSTANTS['TYPE_POKEMON']
TYPE_TRIGGER = CONSTANTS['TYPE_TRIGGER']
ITEM = CONSTANTS['ITEM']
POKEMON_TYPE = CONSTANTS['POKEMON_TYPE']
LIST_TYPE = CONSTANTS['LIST_TYPE']

# Synergy trigger thresholds normalized to lowercase (from TYPE_TRIGGER API response)
SYNERGY_TRIGGERS = {k.lower(): v for k, v in TYPE_TRIGGER.items()}

# Database configuration
DB_NAME = os.environ.get("DATABASE_ENV", "dev")

