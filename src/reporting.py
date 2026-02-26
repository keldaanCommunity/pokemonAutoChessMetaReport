"""Meta report generation and MongoDB export"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pymongo import MongoClient
from .utils import DB_NAME


def get_meta_report(df):
    """
    Generate meta composition report from clustered data.

    For each cluster (excluding noise points labeled -1), calculates: composition metrics 
    (active synergies and their effective levels), performance metrics (winrate, mean rank), 
    and a representative mean team (average Pokemon and synergy composition).

    Args:
        df (pd.DataFrame): DataFrame with cluster assignments ('cluster_id') and match features 
                          including rank, synergy activation levels (capped at trigger thresholds),
                          and t-SNE coordinates (x, y)

    Returns:
        list: List of meta report dictionaries, each containing cluster synergy composition and mean team
    """
    from .utils import SYNERGY_TRIGGERS

    n_row_total = df.shape[0]
    list_cluster_id = df["cluster_id"].unique().tolist()
    if '-1' in list_cluster_id:
        list_cluster_id.remove('-1')

    # Get list of synergy columns (all except rank, nbplayers, pokemons, items, x, y, cluster_id)
    synergy_columns = [c for c in df.columns
                       if c not in ["rank", "nbplayers", "pokemons", "items", "x", "y", "cluster_id"]]

    list_meta_report = []
    for cluster_id in list_cluster_id:
        df_sub_cluster = df[df["cluster_id"] == cluster_id]
        meta_report = {}
        meta_report["cluster_id"] = cluster_id
        n_row = df_sub_cluster.shape[0]
        size_ratio = 100 * n_row / n_row_total
        meta_report["count"] = n_row
        meta_report["ratio"] = round(size_ratio, 5)

        n_rank1 = df_sub_cluster[df_sub_cluster["rank"] == 1].shape[0]
        winrate = 100 * n_rank1 / n_row
        meta_report["winrate"] = round(winrate, 5)
        mean_rank = df_sub_cluster["rank"].mean()
        meta_report["mean_rank"] = round(mean_rank, 5)

        if not synergy_columns:
            print(f"\tskip undefined cluster {cluster_id} with size {n_row}")
            continue

        # Calculate mean synergy level for each synergy in this cluster
        s_mean_synergy = df_sub_cluster[synergy_columns].mean()

        # Round to nearest valid threshold value for each synergy
        synergy_activations = {}
        for synergy_name, mean_value in s_mean_synergy.items():
            if mean_value > 0:
                # Find nearest valid threshold
                thresholds = SYNERGY_TRIGGERS.get(synergy_name, [])
                if thresholds:
                    # Find the highest threshold <= mean_value
                    effective_level = 0
                    for threshold in thresholds:
                        if mean_value >= threshold:
                            effective_level = threshold
                        else:
                            break
                    if effective_level > 0:
                        synergy_activations[synergy_name] = effective_level

        meta_report["synergies"] = synergy_activations

        # Create mean team representing the cluster
        mean_team = {}
        mean_team["cluster_id"] = cluster_id
        mean_team["rank"] = round(df_sub_cluster["rank"].mean(), 2)

        # Calculate mean Pokemon composition with items
        pokemon_data = {}  # {pokemon_name: {count: int, items: []}}

        for _, row in df_sub_cluster.iterrows():
            if "pokemons" in row and row["pokemons"]:
                pokemons_list = row["pokemons"]

                # Check if pokemons contain item data (list of dicts) or just names (list of strings)
                for pokemon_entry in pokemons_list:
                    if isinstance(pokemon_entry, dict):
                        # Data structure: {"name": "Pikachu", "items": ["Sword", "Shield"]}
                        pokemon_name = pokemon_entry.get("name")
                        pokemon_items = pokemon_entry.get("items", [])
                    else:
                        # Fallback: just pokemon name (string)
                        pokemon_name = pokemon_entry
                        pokemon_items = []

                    if pokemon_name:
                        if pokemon_name not in pokemon_data:
                            pokemon_data[pokemon_name] = {
                                "count": 0, "items": []}
                        pokemon_data[pokemon_name]["count"] += 1
                        # Add actual items for this Pokemon
                        if pokemon_items:
                            pokemon_data[pokemon_name]["items"].extend(
                                pokemon_items)

        # Sort by frequency and keep top 10
        sorted_pokemons = sorted(pokemon_data.items(
        ), key=lambda x: x[1]["count"], reverse=True)[:10]

        # Convert to mean format with items
        mean_pokemons = {}
        if sorted_pokemons:
            for pokemon, data in sorted_pokemons:
                frequency = data["count"] / n_row

                # Calculate mean items held and get top items
                items_list = data["items"]
                if items_list:
                    # Calculate mean items per Pokemon occurrence
                    mean_items = len(items_list) / \
                        data["count"] if data["count"] > 0 else 0
                    # Cap at 3 since Pokemon can only hold 0-3 items
                    mean_items = min(mean_items, 3.0)
                    # Round to nearest integer
                    num_items = int(round(mean_items))

                    # Count item frequencies
                    from collections import Counter
                    item_counts = Counter(items_list)
                    top_items = [item for item,
                                 _ in item_counts.most_common(num_items)]
                else:
                    mean_items = 0
                    top_items = []

                mean_pokemons[pokemon] = {
                    "frequency": round(frequency, 3),
                    "mean_items": round(mean_items, 2),
                    "items": top_items
                }

        mean_team["pokemons"] = mean_pokemons

        # Calculate mean synergy levels for the team
        team_synergies = {}
        for synergy_name in synergy_columns:
            synergy_value = s_mean_synergy[synergy_name]
            if synergy_value > 0:
                team_synergies[synergy_name] = round(synergy_value, 2)

        mean_team["synergies"] = team_synergies

        meta_report["mean_team"] = mean_team

        # Calculate mean items for the cluster
        all_items = []  # Collect all items from all Pokemon in cluster
        for _, row in df_sub_cluster.iterrows():
            if "pokemons" in row and row["pokemons"]:
                pokemons_list = row["pokemons"]
                for pokemon_entry in pokemons_list:
                    if isinstance(pokemon_entry, dict):
                        pokemon_items = pokemon_entry.get("items", [])
                    else:
                        pokemon_items = []

                    if pokemon_items:
                        all_items.extend(pokemon_items)

        # Calculate item frequencies (top 5 only)
        mean_items_list = []
        if all_items:
            from collections import Counter
            item_counts = Counter(all_items)
            total_items = len(all_items)

            # Sort by frequency descending and keep top 5
            for item, count in item_counts.most_common(5):
                mean_items_list.append({
                    "item": item,
                    "frequency": round(count / total_items, 3)
                })

        meta_report["mean_items"] = mean_items_list

        # Get top 6 teams (ranked by placement/winrate)
        top_teams = []
        df_sorted = df_sub_cluster.sort_values("rank").head(6)

        for _, row in df_sorted.iterrows():
            pokemons_list = row["pokemons"] if "pokemons" in row and row["pokemons"] else [
            ]

            # Create pokemon entries with their actual items
            pokemons_with_items = []
            for pokemon_entry in pokemons_list:
                if isinstance(pokemon_entry, dict):
                    # Data structure: {"name": "Pikachu", "items": ["Sword", "Shield"]}
                    pokemon_name = pokemon_entry.get("name")
                    pokemon_items = pokemon_entry.get("items", [])
                else:
                    # Fallback: just pokemon name (string)
                    pokemon_name = pokemon_entry
                    pokemon_items = []

                if pokemon_name:
                    pokemons_with_items.append({
                        "name": pokemon_name,
                        "items": pokemon_items if pokemon_items else []
                    })

            team_entry = {
                "rank": int(row["rank"]),
                "elo": int(row["elo"]) if "elo" in row else 0,
                "pokemons": pokemons_with_items
            }
            top_teams.append(team_entry)

        meta_report["top_teams"] = top_teams

        # Compute convex hull for cluster boundary
        from scipy.spatial import ConvexHull
        points = df_sub_cluster[["x", "y"]].values

        hull_points = []
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                # Get hull vertices in order
                hull_points = [[float(points[i][0]), float(
                    points[i][1])] for i in hull.vertices]
            except Exception as e:
                # If convex hull fails (e.g., colinear points), use a simple circle
                print(
                    f"\tWarning: Could not compute convex hull for cluster {cluster_id}: {e}")
                hull_points = []

        meta_report["hull"] = hull_points

        # Calculate cluster center position
        meta_report["x"] = np.mean([k for k in df_sub_cluster["x"]])
        meta_report["y"] = np.mean([k for k in df_sub_cluster["y"]])

        list_meta_report.append(meta_report)

    return list_meta_report


def create_metadata(json_data, time_limit):
    """
    Create metadata about the analysis run.

    Records when the analysis was created, how many documents were processed, 
    and what time window was used for the analysis.

    Args:
        json_data (list): List of match documents that were analyzed
        time_limit (int): Timestamp in milliseconds representing the cutoff time for data inclusion

    Returns:
        list: Single-element list containing metadata dictionary
    """
    metadata = {}
    metadata["created_at"] = datetime.now().isoformat()
    metadata["count"] = len(json_data)
    metadata["time_limit"] = datetime.fromtimestamp(
        time_limit / 1000).isoformat()
    return [metadata]


def export_data_mongodb(list_data, db_name, collection_name):
    """
    Export data to MongoDB.

    Clears existing documents in the target collection and inserts new data.
    Automatically closes the connection after export.

    Args:
        list_data (list): List of documents to insert into MongoDB
        db_name (str): Name of the MongoDB database
        collection_name (str): Name of the collection to export to

    Returns:
        None: Data is written directly to MongoDB
    """
    uri = os.getenv("MONGO_URI")
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    collection.delete_many({})
    collection.insert_many(list_data)
    client.close()


def export_pokemon_data_with_history_mongodb(list_data, db_name, collection_name, max_history=15):
    """
    Export Pokemon statistics to MongoDB with history tracking.

    Updates existing documents by appending current metrics to history arrays.
    Maintains a rolling window of the last `max_history` entries per Pokemon.
    If document doesn't exist, creates it with empty history.

    Args:
        list_data (list): List of tier documents, each containing tier name, timestamp, and Pokemon stats
        db_name (str): Name of the MongoDB database
        collection_name (str): Name of the collection to export to
        max_history (int): Maximum number of history entries to keep per Pokemon (default: 15)

    Returns:
        None: Data is written directly to MongoDB
    """
    uri = os.getenv("MONGO_URI")
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]

    for tier_data in list_data:
        tier_name = tier_data["tier"]
        timestamp = tier_data["timestamp"]
        pokemons_data = tier_data["pokemons"]

        # Check if tier document exists
        existing_doc = collection.find_one({"tier": tier_name})

        if existing_doc is None:
            # Create new document with empty history for each Pokemon
            new_doc = {
                "tier": tier_name,
                "pokemons": {}
            }

            for pokemon_name, pokemon_stats in pokemons_data.items():
                new_doc["pokemons"][pokemon_name] = {
                    "name": pokemon_stats["name"],
                    "rank": pokemon_stats["rank"],
                    "count": pokemon_stats["count"],
                    "item_count": pokemon_stats["item_count"],
                    "items": pokemon_stats["items"],
                    "rank_history": [{"date": timestamp, "value": pokemon_stats["rank"]}],
                    "count_history": [{"date": timestamp, "value": pokemon_stats["count"]}],
                    "item_count_history": [{"date": timestamp, "value": pokemon_stats["item_count"]}]
                }

            collection.insert_one(new_doc)
        else:
            # Update existing document with new data and append to history
            existing_pokemons = existing_doc.get("pokemons", {})

            for pokemon_name, pokemon_stats in pokemons_data.items():
                if pokemon_name in existing_pokemons:
                    # Pokemon exists - append to history and update current values
                    existing_pokemon = existing_pokemons[pokemon_name]

                    # Get existing history or create empty lists
                    rank_history = existing_pokemon.get("rank_history", [])
                    count_history = existing_pokemon.get("count_history", [])
                    item_count_history = existing_pokemon.get(
                        "item_count_history", [])

                    # Append new values to history
                    rank_history.append(
                        {"date": timestamp, "value": pokemon_stats["rank"]})
                    count_history.append(
                        {"date": timestamp, "value": pokemon_stats["count"]})
                    item_count_history.append(
                        {"date": timestamp, "value": pokemon_stats["item_count"]})

                    # Keep only last max_history entries
                    rank_history = rank_history[-max_history:]
                    count_history = count_history[-max_history:]
                    item_count_history = item_count_history[-max_history:]

                    # Update the Pokemon data
                    collection.update_one(
                        {"tier": tier_name},
                        {
                            "$set": {
                                f"pokemons.{pokemon_name}.rank": pokemon_stats["rank"],
                                f"pokemons.{pokemon_name}.count": pokemon_stats["count"],
                                f"pokemons.{pokemon_name}.item_count": pokemon_stats["item_count"],
                                f"pokemons.{pokemon_name}.items": pokemon_stats["items"],
                                f"pokemons.{pokemon_name}.rank_history": rank_history,
                                f"pokemons.{pokemon_name}.count_history": count_history,
                                f"pokemons.{pokemon_name}.item_count_history": item_count_history
                            }
                        }
                    )
                else:
                    # New Pokemon - add with initial history entry
                    collection.update_one(
                        {"tier": tier_name},
                        {
                            "$set": {
                                f"pokemons.{pokemon_name}": {
                                    "name": pokemon_stats["name"],
                                    "rank": pokemon_stats["rank"],
                                    "count": pokemon_stats["count"],
                                    "item_count": pokemon_stats["item_count"],
                                    "items": pokemon_stats["items"],
                                    "rank_history": [{"date": timestamp, "value": pokemon_stats["rank"]}],
                                    "count_history": [{"date": timestamp, "value": pokemon_stats["count"]}],
                                    "item_count_history": [{"date": timestamp, "value": pokemon_stats["item_count"]}]
                                }
                            }
                        }
                    )

    client.close()


def export_item_data_with_history_mongodb(list_data, db_name, collection_name, max_history=15):
    """
    Export item statistics to MongoDB with history tracking.

    Updates existing documents by appending current metrics to history arrays.
    Maintains a rolling window of the last `max_history` entries per item.
    If document doesn't exist, creates it with empty history.

    Args:
        list_data (list): List of tier documents, each containing tier name, timestamp, and item stats
        db_name (str): Name of the MongoDB database
        collection_name (str): Name of the collection to export to
        max_history (int): Maximum number of history entries to keep per item (default: 15)

    Returns:
        None: Data is written directly to MongoDB
    """
    uri = os.getenv("MONGO_URI")
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]

    for tier_data in list_data:
        tier_name = tier_data["tier"]
        timestamp = tier_data["timestamp"]
        items_data = tier_data["items"]

        # Check if tier document exists
        existing_doc = collection.find_one({"tier": tier_name})

        if existing_doc is None:
            # Create new document with empty history for each item
            new_doc = {
                "tier": tier_name,
                "items": {}
            }

            for item_name, item_stats in items_data.items():
                new_doc["items"][item_name] = {
                    "name": item_stats["name"],
                    "rank": item_stats["rank"],
                    "count": item_stats["count"],
                    "pokemons": item_stats["pokemons"],
                    "rank_history": [{"date": timestamp, "value": item_stats["rank"]}],
                    "count_history": [{"date": timestamp, "value": item_stats["count"]}]
                }

            collection.insert_one(new_doc)
        else:
            # Update existing document with new data and append to history
            existing_items = existing_doc.get("items", {})

            for item_name, item_stats in items_data.items():
                if item_name in existing_items:
                    # Item exists - append to history and update current values
                    existing_item = existing_items[item_name]

                    # Get existing history or create empty lists
                    rank_history = existing_item.get("rank_history", [])
                    count_history = existing_item.get("count_history", [])

                    # Append new values to history
                    rank_history.append(
                        {"date": timestamp, "value": item_stats["rank"]})
                    count_history.append(
                        {"date": timestamp, "value": item_stats["count"]})

                    # Keep only last max_history entries
                    rank_history = rank_history[-max_history:]
                    count_history = count_history[-max_history:]

                    # Update the item data
                    collection.update_one(
                        {"tier": tier_name},
                        {
                            "$set": {
                                f"items.{item_name}.rank": item_stats["rank"],
                                f"items.{item_name}.count": item_stats["count"],
                                f"items.{item_name}.pokemons": item_stats["pokemons"],
                                f"items.{item_name}.rank_history": rank_history,
                                f"items.{item_name}.count_history": count_history
                            }
                        }
                    )
                else:
                    # New item - add with initial history entry
                    collection.update_one(
                        {"tier": tier_name},
                        {
                            "$set": {
                                f"items.{item_name}": {
                                    "name": item_stats["name"],
                                    "rank": item_stats["rank"],
                                    "count": item_stats["count"],
                                    "pokemons": item_stats["pokemons"],
                                    "rank_history": [{"date": timestamp, "value": item_stats["rank"]}],
                                    "count_history": [{"date": timestamp, "value": item_stats["count"]}]
                                }
                            }
                        }
                    )

    client.close()


def export_meta_report_text(meta_report, output_dir):
    """
    Export meta report as a human-readable text file.

    Creates a comprehensive text summary of all clusters including:
    - Cluster ID, size, and performance metrics
    - Active synergies and their levels
    - Mean team with top 10 Pokemon and their items
    - Winrate and average rank

    Args:
        meta_report (list): List of cluster dictionaries from get_meta_report()
        output_dir (str): Directory to save the report file

    Returns:
        str: Path to the generated report file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"meta_report_{timestamp}.txt")

    with open(filepath, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("POKEMON AUTO CHESS META REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")

        for cluster in meta_report:
            cluster_id = cluster["cluster_id"]
            count = cluster["count"]
            ratio = cluster["ratio"]
            winrate = cluster["winrate"]
            mean_rank = cluster["mean_rank"]
            synergies = cluster.get("synergies", {})
            mean_team = cluster.get("mean_team", {})

            f.write(f"CLUSTER {cluster_id}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Size: {count} teams ({ratio}% of total)\n")
            f.write(f"  Winrate: {winrate}%\n")
            f.write(f"  Average Rank: {mean_rank}\n")
            f.write(
                f"  Position: x={cluster.get('x', 0):.2f}, y={cluster.get('y', 0):.2f}\n")
            f.write("\n")

            f.write(f"  Synergy Activations:\n")
            if synergies:
                for synergy_name in sorted(synergies.keys()):
                    level = synergies[synergy_name]
                    f.write(f"    - {synergy_name.upper()}: Level {level}\n")
            else:
                f.write(f"    (None)\n")
            f.write("\n")

            f.write(f"  Mean Team (Top 10 Pokemon):\n")
            if mean_team and "pokemons" in mean_team:
                pokemon_list = mean_team["pokemons"]
                for pokemon, data in pokemon_list.items():
                    frequency = data.get("frequency", 0) * 100
                    mean_items = data.get("mean_items", 0)
                    items = data.get("items", [])

                    f.write(
                        f"    - {pokemon}: {frequency:.1f}% frequency, {mean_items:.1f} avg items\n")
                    if items:
                        f.write(f"      Items: {', '.join(items)}\n")
            else:
                f.write(f"    (None)\n")
            f.write("\n\n")

    print(f"{datetime.now().time()} exported meta report to {filepath}")
    return filepath


def export_meta_report_json(meta_report, output_dir):
    """
    Export meta report as formatted JSON file.

    Serializes cluster analysis including:
    - Cluster metrics (size, winrate, rank)
    - Synergy activations
    - Mean team with Pokemon and items

    Args:
        meta_report (list): List of cluster dictionaries from get_meta_report()
        output_dir (str): Directory to save the JSON file

    Returns:
        str: Path to the generated JSON file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"meta_report_{timestamp}.json")

    def make_json_serializable(obj):
        """Convert numpy types and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        return obj

    # Convert to JSON-serializable format
    report_data = [make_json_serializable(cluster) for cluster in meta_report]

    with open(filepath, 'w') as f:
        json.dump(report_data, f, indent=2)

    print(f"{datetime.now().time()} exported meta report JSON to {filepath}")
    return filepath


def visualize_meta_report(meta_report, output_dir):
    """
    Create visualizations of the meta report.

    Generates multiple plots:
    - Cluster composition by synergy
    - Cluster performance metrics (winrate, mean rank)
    - Cluster distribution in t-SNE space

    Args:
        meta_report (list): List of cluster dictionaries from get_meta_report()
        output_dir (str): Directory to save visualization files

    Returns:
        list: Paths to generated visualization files
    """
    filepaths = []

    # Skip visualization if no clusters to visualize
    if not meta_report or len(meta_report) == 0:
        print(f"{datetime.now().time()} no clusters to visualize")
        return filepaths

    # 1. Cluster positioning and size
    fig, ax = plt.subplots(figsize=(12, 8))
    cluster_ids = [str(c["cluster_id"]) for c in meta_report]
    x_positions = [c.get("x", 0) for c in meta_report]
    y_positions = [c.get("y", 0) for c in meta_report]
    sizes = [c["count"] * 2 for c in meta_report]  # Scale for visibility
    colors = [c["winrate"] for c in meta_report]  # Color by winrate

    scatter = ax.scatter(x_positions, y_positions, s=sizes, c=colors,
                         cmap='RdYlGn', alpha=0.6, edgecolors='black')

    for i, cluster_id in enumerate(cluster_ids):
        ax.annotate(cluster_id, (x_positions[i], y_positions[i]),
                    ha='center', va='center', fontweight='bold')

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("Meta Clusters - Size and Winrate")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Winrate (%)")
    ax.grid(True, alpha=0.3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath1 = os.path.join(
        output_dir, f"meta_clusters_positioning_{timestamp}.png")
    plt.savefig(filepath1, dpi=150, bbox_inches='tight')
    filepaths.append(filepath1)
    plt.close()

    # 2. Performance metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cluster_labels = [f"C{c['cluster_id']}" for c in meta_report]
    winrates = [c["winrate"] for c in meta_report]
    mean_ranks = [c["mean_rank"] for c in meta_report]

    ax1.bar(cluster_labels, winrates, color='skyblue', edgecolor='black')
    ax1.set_ylabel("Winrate (%)")
    ax1.set_title("Cluster Winrates")
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=np.mean(winrates), color='r',
                linestyle='--', label='Average')
    ax1.legend()

    ax2.bar(cluster_labels, mean_ranks, color='lightcoral', edgecolor='black')
    ax2.set_ylabel("Average Rank")
    ax2.set_title("Cluster Mean Ranks (Lower is Better)")
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=np.mean(mean_ranks), color='r',
                linestyle='--', label='Average')
    ax2.legend()
    ax2.invert_yaxis()

    plt.tight_layout()
    filepath2 = os.path.join(output_dir, f"meta_performance_{timestamp}.png")
    plt.savefig(filepath2, dpi=150, bbox_inches='tight')
    filepaths.append(filepath2)
    plt.close()

    # 3. Synergy composition heatmap
    if meta_report:
        all_synergies = set()
        for cluster in meta_report:
            all_synergies.update(cluster.get("synergies", {}).keys())

        if all_synergies:
            synergy_list = sorted(all_synergies)
            synergy_matrix = []
            cluster_labels_short = [f"C{c['cluster_id']}" for c in meta_report]

            for cluster in meta_report:
                synergies = cluster.get("synergies", {})
                row = [synergies.get(s, 0) for s in synergy_list]
                synergy_matrix.append(row)

            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(np.array(synergy_matrix).T,
                           cmap='YlOrRd', aspect='auto')

            ax.set_xticks(range(len(cluster_labels_short)))
            ax.set_xticklabels(cluster_labels_short)
            ax.set_yticks(range(len(synergy_list)))
            ax.set_yticklabels(synergy_list)

            ax.set_xlabel("Cluster")
            ax.set_ylabel("Synergy")
            ax.set_title("Synergy Activation Levels by Cluster")

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Activation Level")

            plt.tight_layout()
            filepath3 = os.path.join(
                output_dir, f"meta_synergies_heatmap_{timestamp}.png")
            plt.savefig(filepath3, dpi=150, bbox_inches='tight')
            filepaths.append(filepath3)
            plt.close()

    print(f"{datetime.now().time()} generated {len(filepaths)} visualizations in {output_dir}")
    return filepaths


def export_meta_report_mongodb(meta_report, db_name):
    """
    Export meta report to MongoDB for production consumption.

    Exports cluster data to meta-report-v2 collection (1100+ tier only).
    Uses the v2 schema with mean_team instead of individual teams.

    Args:
        meta_report (list): List of cluster dictionaries from get_meta_report()
        db_name (str): Name of the MongoDB database

    Returns:
        None: Data is written directly to MongoDB
    """
    uri = os.getenv("MONGO_URI")
    client = MongoClient(uri)
    db = client[db_name]

    collection = db["meta-report-v2"]

    # Add timestamp to each report
    reports_with_metadata = []
    for report in meta_report:
        report_doc = report.copy()
        report_doc["generated_at"] = datetime.now().isoformat()
        reports_with_metadata.append(report_doc)

    # Clear existing data and insert new reports
    collection.delete_many({})
    if reports_with_metadata:
        collection.insert_many(reports_with_metadata)

    print(f"{datetime.now().time()} exported {len(reports_with_metadata)} clusters to meta-report-v2")
    client.close()


def export_dendrogram_mongodb(df_2d, df_synergies, df_full, n_clusters, linkage_method, db_name):
    """
    Export dendrogram data to MongoDB for rendering in external applications.

    Computes hierarchical clustering and exports the full linkage matrix,
    cluster profiles with synergy characterization, branch profiles for
    internal nodes, and dendrogram layout coordinates for visualization.

    Args:
        df_2d (pd.DataFrame): DataFrame with 'x' and 'y' columns (reduced coordinates)
        df_synergies (pd.DataFrame): Original synergy features DataFrame
        df_full (pd.DataFrame): Full original DataFrame with pokemons and items data
        n_clusters (int): Number of leaf clusters
        linkage_method (str): Linkage criterion ('ward', 'complete', 'average', 'single')
        db_name (str): Name of the MongoDB database

    Returns:
        dict: The dendrogram document that was exported
    """
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print(f"{datetime.now().time()} Computing dendrogram for MongoDB export...")

    # Compute linkage matrix on 2D coordinates
    Z = linkage(df_2d[["x", "y"]].values, method=linkage_method)
    n_samples = len(df_2d)

    # Get cluster assignments for n_clusters
    cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')

    from .utils import SYNERGY_TRIGGERS

    # Build cluster profiles (leaf nodes)
    cluster_profiles = []
    for cluster_id in range(1, n_clusters + 1):
        mask = cluster_labels == cluster_id
        cluster_size = int(mask.sum())

        if cluster_size > 0:
            # Get mean synergies for this cluster
            cluster_means = df_synergies[mask].mean()

            # Calculate effective synergy levels (like get_meta_report)
            synergy_activations = {}
            for synergy_name, mean_value in cluster_means.items():
                if mean_value > 0:
                    thresholds = SYNERGY_TRIGGERS.get(synergy_name, [])
                    if thresholds:
                        effective_level = 0
                        for threshold in thresholds:
                            if mean_value >= threshold:
                                effective_level = threshold
                            else:
                                break
                        if effective_level > 0:
                            synergy_activations[synergy_name] = effective_level

            # Sort synergies by effective level (highest first) and take top 5
            sorted_synergies = sorted(
                synergy_activations.items(), key=lambda x: -x[1])[:5]

            # Get original dataframe rows for this cluster to extract pokemons
            cluster_indices = np.where(mask)[0]

            # Calculate top 10 popular pokemons
            pokemon_counts = {}
            for idx in cluster_indices:
                if idx < len(df_full) and 'pokemons' in df_full.columns:
                    pokemons_list = df_full.iloc[idx].get('pokemons', [])
                    if pokemons_list:
                        for pokemon_entry in pokemons_list:
                            if isinstance(pokemon_entry, dict):
                                pokemon_name = pokemon_entry.get('name')
                            else:
                                pokemon_name = pokemon_entry
                            if pokemon_name:
                                pokemon_counts[pokemon_name] = pokemon_counts.get(
                                    pokemon_name, 0) + 1

            # Sort by count and get top 10 with frequency
            sorted_pokemons = sorted(
                pokemon_counts.items(), key=lambda x: -x[1])[:10]
            top_pokemons = [
                {'name': name, 'frequency': round(count / cluster_size, 3)}
                for name, count in sorted_pokemons
            ]

            profile = {
                'cluster_id': cluster_id,
                'size': cluster_size,
                'synergies': {syn: level for syn, level in sorted_synergies},
                'top_pokemons': top_pokemons
            }
            cluster_profiles.append(profile)

    # Generate dendrogram to get layout coordinates
    fig, ax = plt.subplots(figsize=[1, 1])
    dendro_data = dendrogram(
        Z,
        ax=ax,
        truncate_mode='lastp',
        p=n_clusters,
        no_plot=False,
        get_leaves=True
    )
    plt.close(fig)

    # Helper function to get all leaf indices under a node
    def get_leaves_for_node(node_idx, Z, n_samples):
        """Recursively get all original sample indices under a node."""
        if node_idx < n_samples:
            return [int(node_idx)]
        else:
            merge_idx = int(node_idx - n_samples)
            left = int(Z[merge_idx, 0])
            right = int(Z[merge_idx, 1])
            return get_leaves_for_node(left, Z, n_samples) + get_leaves_for_node(right, Z, n_samples)

    # Build branch profiles (internal nodes)
    branch_profiles = []
    icoord_list = dendro_data.get('icoord', [])
    dcoord_list = dendro_data.get('dcoord', [])

    for branch_idx, (icoord, dcoord) in enumerate(zip(icoord_list, dcoord_list)):
        merge_height = dcoord[1]  # Height of the merge (top of U)

        # Find which merge this corresponds to by height
        height_matches = np.where(np.isclose(
            Z[:, 2], merge_height, rtol=1e-9))[0]

        if len(height_matches) > 0:
            merge_idx = int(height_matches[0])
            node_idx = n_samples + merge_idx

            # Get all original sample indices under this branch
            sample_indices = get_leaves_for_node(node_idx, Z, n_samples)
            total_size = len(sample_indices)

            # Get cluster IDs for these samples
            leaf_cluster_ids = sorted(
                list(set(cluster_labels[sample_indices])))

            # Compute synergy profile for this branch - get synergy with highest mean value
            branch_data = df_synergies.iloc[sample_indices]
            branch_means = branch_data.mean()

            # Find synergy with highest mean value
            dominant_synergy = None
            max_value = 0
            for synergy_name, mean_value in branch_means.items():
                if mean_value > max_value:
                    max_value = mean_value
                    dominant_synergy = synergy_name

            # Count pokemons in this branch and get top 5 most used
            pokemon_counts = {}
            for idx in sample_indices:
                if idx < len(df_full) and 'pokemons' in df_full.columns:
                    pokemons_list = df_full.iloc[idx].get('pokemons', [])
                    if pokemons_list:
                        for pokemon_entry in pokemons_list:
                            if isinstance(pokemon_entry, dict):
                                pokemon_name = pokemon_entry.get('name')
                            else:
                                pokemon_name = pokemon_entry
                            if pokemon_name:
                                pokemon_counts[pokemon_name] = pokemon_counts.get(
                                    pokemon_name, 0) + 1

            # Sort by count and get top 5
            sorted_pokemons = sorted(
                pokemon_counts.items(), key=lambda x: -x[1])[:5]
            top_pokemons = [
                {'name': name, 'count': count}
                for name, count in sorted_pokemons
            ]

            branch_profile = {
                'branch_index': branch_idx,
                'merge_index': merge_idx,
                'merge_height': float(merge_height),
                'total_size': total_size,
                'leaf_cluster_ids': [int(c) for c in leaf_cluster_ids],
                'synergy': dominant_synergy,
                'top_pokemons': top_pokemons
            }
            branch_profiles.append(branch_profile)

    # Build leaf to cluster mapping
    # dendro_data['leaves'] gives the order of leaves as displayed
    leaves = [int(x) for x in dendro_data.get('leaves', [])]

    # Map from visual leaf position to cluster_id
    # When truncated, leaves can be cluster indices or (n_samples + merge_index)
    leaf_to_cluster = []
    for leaf in leaves:
        if leaf < n_samples:
            # This is an original sample
            leaf_to_cluster.append(int(cluster_labels[leaf]))
        else:
            # This represents a merged cluster - get any of its original samples
            sample_indices = get_leaves_for_node(leaf, Z, n_samples)
            if sample_indices:
                leaf_to_cluster.append(
                    int(cluster_labels[sample_indices[0]]))

    # Convert linkage matrix to list of node objects
    linkage_matrix = []
    for row in Z:
        linkage_matrix.append({
            'cluster1': int(row[0]),
            'cluster2': int(row[1]),
            'distance': float(row[2]),
            'count': int(row[3])
        })

    # Build the dendrogram document
    dendrogram_doc = {
        'linkage_method': linkage_method,
        'n_clusters': n_clusters,
        'n_samples': n_samples,
        'linkage_matrix': linkage_matrix,
        'cluster_profiles': cluster_profiles,
        'branch_profiles': branch_profiles,
        'leaves': leaves,
        'leaf_to_cluster': leaf_to_cluster,
        'icoord': [[float(x) for x in coord] for coord in icoord_list],
        'dcoord': [[float(x) for x in coord] for coord in dcoord_list],
        'generated_at': datetime.now().isoformat()
    }

    # Export to MongoDB
    uri = os.getenv("MONGO_URI")
    client = MongoClient(uri)
    db = client[db_name]

    collection = db["dendrogram"]

    # Clear existing data and insert new dendrogram
    collection.delete_many({})
    collection.insert_one(dendrogram_doc)

    print(f"{datetime.now().time()} exported dendrogram ({n_clusters} clusters, {len(branch_profiles)} branches, {linkage_method}) to MongoDB")
    client.close()

    return dendrogram_doc
