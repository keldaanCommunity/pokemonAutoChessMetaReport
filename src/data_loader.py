"""Data loading and DataFrame creation"""

import os
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from .utils import LIST_POKEMON, ITEM, DB_NAME, SYNERGY_TRIGGERS


def load_data_mongodb(time_limit, db_name=DB_NAME, limit=None):
    """
    Load match data from MongoDB database with optimizations for large datasets.

    Uses MongoDB projection to fetch only required fields and ensures index exists 
    on 'time' field for fast filtering. Sorts results by time descending.

    Args:
        time_limit (int): Timestamp in milliseconds; only loads matches with time > time_limit
        db_name (str): Name of the MongoDB database to connect to (default: DB_NAME from utils)
        limit (int): Maximum number of documents to retrieve; None for no limit (default: None)

    Returns:
        list: List of match documents from MongoDB
    """
    uri = os.environ.get("MONGO_URI")
    client = MongoClient(uri)
    db = client[db_name]
    collection = db['detailledstatisticv2']

    # Ensure index exists on 'time' field for fast filtering
    try:
        collection.create_index("time")
    except:
        pass  # Index may already exist

    # Use projection to fetch only required fields for better performance
    projection = {
        "rank": 1,
        "nbplayers": 1,
        "synergies": 1,
        "pokemons": 1,
        "items": 1,
        "elo": 1,
        "time": 1,
        "regions": 1
    }

    cursor = collection.find(
        {"time": {"$gt": time_limit}}, projection=projection).sort("time", -1)
    if limit:
        cursor = cursor.limit(limit)
    result = list(cursor)
    client.close()
    return result


def create_dataframe(json_data):
    """
    Convert match JSON data to DataFrame with synergies capped at activation levels.

    For each synergy, stores the highest trigger threshold it meets or exceeds.
    For example, FIRE [2,4,6,8] with count=7 → stored as 6 (highest threshold met).
    This captures synergy activation tiers without feature explosion.

    Args:
        json_data (list): List of match documents from MongoDB, each containing rank, nbplayers, synergies, and pokemons

    Returns:
        pd.DataFrame: DataFrame where each row is a match with synergy activation levels and Pokemon list
    """
    list_match = []
    for i in range(len(json_data)):
        data = json_data[i]
        match = {}
        match["rank"] = data["rank"]
        match["nbplayers"] = data["nbplayers"] if "nbplayers" in data else 8
        match["elo"] = data.get("elo", 0)

        # Extract Pokemon data (keep full objects with items for later analysis)
        pokemons_with_items = []
        match_items = []  # All items from all Pokemon in this match
        if "pokemons" in data and data["pokemons"]:
            for pokemon in data["pokemons"]:
                if isinstance(pokemon, dict) and "name" in pokemon:
                    # Keep full pokemon object with items
                    pokemons_with_items.append({
                        "name": pokemon.get("name"),
                        "items": pokemon.get("items", []) if isinstance(pokemon.get("items"), list) else []
                    })
                    # Collect items for match-level tracking
                    if "items" in pokemon and isinstance(pokemon["items"], list):
                        match_items.extend(pokemon["items"])
                elif isinstance(pokemon, str):
                    # Fallback: if pokemon is just a string name
                    pokemons_with_items.append({
                        "name": pokemon,
                        "items": []
                    })

        match["pokemons"] = pokemons_with_items
        match["items"] = match_items

        # If no items found in Pokemon objects, try match-level items
        if not match_items and "items" in data and isinstance(data["items"], list):
            match["items"] = data["items"]

        # Use pre-computed synergies from the database
        # For each synergy, store the highest trigger threshold it meets
        if "synergies" in data and data["synergies"]:
            for synergy, count in data["synergies"].items():
                synergy_lower = synergy.lower()

                # Get trigger thresholds for this synergy
                thresholds = SYNERGY_TRIGGERS.get(synergy_lower)
                if thresholds:
                    # Find the highest threshold this count meets or exceeds
                    effective_level = 0
                    for threshold in thresholds:
                        if count >= threshold:
                            effective_level = threshold
                        else:
                            break  # Thresholds are in ascending order

                    # Store the effective level (highest threshold met)
                    match[synergy_lower] = effective_level

        list_match.append(match)

    dataframe = pd.DataFrame(list_match)
    dataframe.fillna(0, inplace=True)
    return dataframe


def create_item_data_elo_threshold(json_data):
    """
    Generate item statistics filtered by ELO rating thresholds.

    Groups items by ELO tier and calculates usage statistics for each item at that tier.
    For each item, tracks: appearance count, average rank, and top 3 Pokemon that carry it.
    Includes timestamp for history tracking.

    Args:
        json_data (list): List of match documents with Pokemon items and ELO ratings

    Returns:
        dict_values: Collection of tier dictionaries, each containing tier name, timestamp, and items statistics
    """
    # Get current timestamp for this generation
    current_timestamp = datetime.now().isoformat()

    elo_threshold_stats = {
        "LEVEL_BALL": {"tier": "LEVEL_BALL", "timestamp": current_timestamp, "items": {}},
        "NET_BALL": {"tier": "NET_BALL", "timestamp": current_timestamp, "items": {}},
        "SAFARI_BALL": {"tier": "SAFARI_BALL", "timestamp": current_timestamp, "items": {}},
        "LOVE_BALL": {"tier": "LOVE_BALL", "timestamp": current_timestamp, "items": {}},
        "PREMIER_BALL": {"tier": "PREMIER_BALL", "timestamp": current_timestamp, "items": {}},
        "QUICK_BALL": {"tier": "QUICK_BALL", "timestamp": current_timestamp, "items": {}},
        "POKE_BALL": {"tier": "POKE_BALL", "timestamp": current_timestamp, "items": {}},
        "SUPER_BALL": {"tier": "SUPER_BALL", "timestamp": current_timestamp, "items": {}},
        "ULTRA_BALL": {"tier": "ULTRA_BALL", "timestamp": current_timestamp, "items": {}},
        "MASTER_BALL": {"tier": "MASTER_BALL", "timestamp": current_timestamp, "items": {}},
        "BEAST_BALL": {"tier": "BEAST_BALL", "timestamp": current_timestamp, "items": {}},
    }

    thresholds = {
        "BEAST_BALL": 1700,
        "MASTER_BALL": 1500,
        "ULTRA_BALL": 1400,
        "SUPER_BALL": 1350,
        "POKE_BALL": 1300,
        "QUICK_BALL": 1250,
        "PREMIER_BALL": 1200,
        "LOVE_BALL": 1150,
        "SAFARI_BALL": 1100,
        "NET_BALL": 1050,
        "LEVEL_BALL": 0,
    }

    for threshold in thresholds:
        elo_threshold = thresholds[threshold]
        item_stats = {}
        for item in ITEM:
            item_stats[item] = {"pokemons": {},
                                "rank": 0, "count": 1, "name": item}
        for match in json_data:
            nbPlayers = match["nbplayers"] if "nbplayers" in match else 8
            if match["elo"] >= elo_threshold:
                for pokemon in match["pokemons"]:
                    for item in pokemon["items"]:
                        if item not in ["HARD_STONE", "TINY_MUSHROOM"]:
                            item_stats[item]["count"] += 1
                            item_stats[item]["rank"] += 1 + \
                                (match["rank"] - 1) * 7 / (nbPlayers - 1)
                            name = pokemon["name"]
                            if name in item_stats[item]["pokemons"]:
                                item_stats[item]["pokemons"][name] += 1
                            else:
                                item_stats[item]["pokemons"][name] = 1

        for item in item_stats:
            item_stats[item]["rank"] = round(
                item_stats[item]["rank"] / item_stats[item]["count"], 2)
            item_stats[item]["pokemons"] = dict(
                sorted(item_stats[item]["pokemons"].items(), key=lambda x: x[1], reverse=True))
            item_stats[item]["pokemons"] = list(
                item_stats[item]["pokemons"])[:3]
        elo_threshold_stats[threshold]["items"] = item_stats

    return elo_threshold_stats.values()


def create_pokemon_data_elo_threshold(json_data):
    """
    Generate Pokemon statistics filtered by ELO rating thresholds.

    Groups Pokemon stats by ELO tier. For each tier, calculates per-Pokemon statistics 
    including: appearance count, average rank, average item count, and top 3 items.
    Includes timestamp for history tracking.

    Args:
        json_data (list): List of match documents with Pokemon data and ELO ratings

    Returns:
        dict_values: Collection of tier dictionaries, each containing tier name, timestamp, and Pokemon statistics
    """
    # Get current timestamp for this generation
    current_timestamp = datetime.now().isoformat()

    elo_threshold_stats = {
        "LEVEL_BALL": {"tier": "LEVEL_BALL", "timestamp": current_timestamp, "pokemons": {}},
        "NET_BALL": {"tier": "NET_BALL", "timestamp": current_timestamp, "pokemons": {}},
        "SAFARI_BALL": {"tier": "SAFARI_BALL", "timestamp": current_timestamp, "pokemons": {}},
        "LOVE_BALL": {"tier": "LOVE_BALL", "timestamp": current_timestamp, "pokemons": {}},
        "PREMIER_BALL": {"tier": "PREMIER_BALL", "timestamp": current_timestamp, "pokemons": {}},
        "QUICK_BALL": {"tier": "QUICK_BALL", "timestamp": current_timestamp, "pokemons": {}},
        "POKE_BALL": {"tier": "POKE_BALL", "timestamp": current_timestamp, "pokemons": {}},
        "SUPER_BALL": {"tier": "SUPER_BALL", "timestamp": current_timestamp, "pokemons": {}},
        "ULTRA_BALL": {"tier": "ULTRA_BALL", "timestamp": current_timestamp, "pokemons": {}},
        "MASTER_BALL": {"tier": "MASTER_BALL", "timestamp": current_timestamp, "pokemons": {}},
        "BEAST_BALL": {"tier": "BEAST_BALL", "timestamp": current_timestamp, "pokemons": {}},
    }

    thresholds = {
        "BEAST_BALL": 1700,
        "MASTER_BALL": 1500,
        "ULTRA_BALL": 1400,
        "SUPER_BALL": 1350,
        "POKE_BALL": 1300,
        "QUICK_BALL": 1250,
        "PREMIER_BALL": 1200,
        "LOVE_BALL": 1150,
        "SAFARI_BALL": 1100,
        "NET_BALL": 1050,
        "LEVEL_BALL": 0,
    }

    for threshold in thresholds:
        elo_threshold = thresholds[threshold]
        pokemon_stats = {}
        for pokemon in LIST_POKEMON:
            pokemon_stats[pokemon] = {"items": {},
                                      "rank": 0, "count": 0, "name": pokemon, "item_count": 0}

        for match in json_data:
            nbPlayers = match["nbplayers"] if "nbplayers" in match else 8
            if match["elo"] >= elo_threshold:
                for pokemon in match["pokemons"]:
                    name = pokemon["name"]
                    pokemon_stats[name]["rank"] += 1 + \
                        (match["rank"] - 1) * 7 / (nbPlayers - 1)
                    pokemon_stats[name]["item_count"] += len(pokemon["items"])
                    pokemon_stats[name]["count"] += 1
                    for item in pokemon["items"]:
                        if item in pokemon_stats[name]["items"]:
                            pokemon_stats[name]["items"][item] += 1
                        else:
                            pokemon_stats[name]["items"][item] = 1

        for pokemon in pokemon_stats:
            if pokemon_stats[pokemon]["count"] == 0:
                pokemon_stats[pokemon]["rank"] = 9
            else:
                pokemon_stats[pokemon]["rank"] = round(
                    pokemon_stats[pokemon]["rank"] / pokemon_stats[pokemon]["count"], 2)
                pokemon_stats[pokemon]["item_count"] = round(
                    pokemon_stats[pokemon]["item_count"] / pokemon_stats[pokemon]["count"], 2)
            pokemon_stats[pokemon]["items"] = dict(sorted(
                pokemon_stats[pokemon]["items"].items(), key=lambda x: x[1], reverse=True))
            pokemon_stats[pokemon]["items"] = list(
                pokemon_stats[pokemon]["items"])[:3]
        elo_threshold_stats[threshold]["pokemons"] = pokemon_stats

    return elo_threshold_stats.values()


def create_region_data(json_data):
    """
    Generate region statistics from match data.

    Calculates per-region statistics including: appearance count, average rank, 
    average ELO, and top 3 most common Pokemon in that region.

    Args:
        json_data (list): List of match documents with region information

    Returns:
        dict_values: Collection of region stat dictionaries for each region in the data
    """
    region_stats = {}

    # Collect all unique regions from data and initialize stats
    for match in json_data:
        if "regions" in match:
            for region in match["regions"]:
                if region not in region_stats:
                    region_stats[region] = {
                        "name": region,
                        "count": 0,
                        "rank": 0,
                        "elo": 0,
                        "pokemons": {}
                    }

    # Aggregate stats
    for match in json_data:
        nbPlayers = match["nbplayers"] if "nbplayers" in match else 8
        if "regions" in match:
            for region in match["regions"]:
                region_stats[region]["count"] += 1
                region_stats[region]["rank"] += 1 + \
                    (match["rank"] - 1) * 7 / (nbPlayers - 1)
                region_stats[region]["elo"] += match["elo"]

                # Track pokemons with this region
                for pokemon in match["pokemons"]:
                    name = pokemon["name"]
                    if name in region_stats[region]["pokemons"]:
                        region_stats[region]["pokemons"][name] += 1
                    else:
                        region_stats[region]["pokemons"][name] = 1

    # Calculate means and format
    for region in region_stats:
        count = region_stats[region]["count"]
        if count > 0:
            region_stats[region]["rank"] = round(
                region_stats[region]["rank"] / count, 2)
            region_stats[region]["elo"] = round(
                region_stats[region]["elo"] / count, 2)
        region_stats[region]["pokemons"] = dict(
            sorted(region_stats[region]["pokemons"].items(), key=lambda x: x[1], reverse=True))
        region_stats[region]["pokemons"] = list(
            region_stats[region]["pokemons"])[:3]

    return region_stats.values()
