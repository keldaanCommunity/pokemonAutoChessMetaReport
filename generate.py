#!/usr/bin/python3

import os
import math
import random
import pandas as pd
import itertools as itools
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from warnings import simplefilter
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN


class ColorGenerator:
    def __init__(self):
        self.__phi = (1 + math.sqrt(5)) / 2
        self.__x = random.random()

    def __hsl2hex(self, h, s, l):
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
        self.__x = (self.__x + self.__phi) % 1
        h = self.__x * 360
        s = (60 + 40 * random.random()) / 100
        l = (50 + s*0.2 * (random.random()*2-1)) / 100
        return self.__hsl2hex(h, s, l)


load_dotenv()
# ignore sklearn warnings
simplefilter(action='ignore', category=FutureWarning)


# load json data
request_pokemons = requests.get('https://pokemon-auto-chess.com/pokemons')
LIST_POKEMON = [p for p in request_pokemons.json().values()]

types_pokemons = requests.get('https://pokemon-auto-chess.com/types')
TYPE_POKEMON = types_pokemons.json()

trigger = requests.get('https://pokemon-auto-chess.com/types-trigger')
TYPE_TRIGGER = trigger.json()

items = requests.get('https://pokemon-auto-chess.com/items')
ITEM = items.json()

# get list of type for each pokemon
POKEMON_TYPE = {}
for pkm in LIST_POKEMON:
    type_list = []
    for t in TYPE_POKEMON:
        type_name = t.lower()
        type_pokemons = TYPE_POKEMON[t]
        if (pkm in type_pokemons):
            type_list.append(type_name)
    POKEMON_TYPE[pkm] = type_list

LIST_TYPE = [k.lower() for k in TYPE_POKEMON.keys()]


def load_data_mongodb(time_limit):
    uri = os.environ.get("MONGO_URI")
    client = MongoClient(uri)
    db = client.test
    collection = db['detailledstatisticv2']
    cursor = collection.find({"time": {"$gt": time_limit}}, limit=50000)
    result = list(cursor)
    client.close()
    return result


def create_item_data(json_data):
    item_stats = {}
    for item in ITEM:
        item_stats[item] = {"pokemons": {},
                            "rank": 0, "count": 1, "name": item}

    for match in json_data:
        for pokemon in match["pokemons"]:
            if pokemon != "SHEDNINJA":
                for item in pokemon["items"]:
                    if item != "ORAN_BERRY" and item != "WATER_INCENSE" and item != "BRIGHT_POWDER" and item != "ZOOM_LENS" and item != "FOCUS_BAND" and item != "ICY_ROCK" and item != "RUNE_PROTECT":
                        item_stats[item]["count"] += 1
                        item_stats[item]["rank"] += match["rank"] * 8 / match["nbplayers"]
                        if (pokemon["name"] in item_stats[item]["pokemons"]):
                            item_stats[item]["pokemons"][pokemon["name"]] += 1

                        else:
                            item_stats[item]["pokemons"][pokemon["name"]] = 1

    for item in item_stats:
        item_stats[item]["rank"] = round(
            item_stats[item]["rank"] / item_stats[item]["count"], 2)
        item_stats[item]["pokemons"] = dict(
            sorted(item_stats[item]["pokemons"].items(), key=lambda x: x[1], reverse=True))
        item_stats[item]["pokemons"] = list(item_stats[item]["pokemons"])[:3]

    return item_stats.values()


def create_pokemon_data(json_data):
    pokemon_stats = {}
    for pokemon in LIST_POKEMON:
        pokemon_stats[pokemon] = {"items": {},
                                  "rank": 0, "count": 0, "name": pokemon, "item_count": 0}

    for match in json_data:
        for pokemon in match["pokemons"]:
            if pokemon["name"] != "SHEDNINJA":
                pokemon_stats[pokemon["name"]]["rank"] += match["rank"] * 8 / match["nbplayers"]
                pokemon_stats[pokemon["name"]]["item_count"] += len(pokemon["items"])
                pokemon_stats[pokemon["name"]]["count"] += 1
                for item in pokemon["items"]:
                    if (item in pokemon_stats[pokemon["name"]]["items"]):
                        pokemon_stats[pokemon["name"]]["items"][item] += 1
                    else:
                        pokemon_stats[pokemon["name"]]["items"][item] = 1

    for pokemon in pokemon_stats:
        if (pokemon_stats[pokemon]["count"] == 0):
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

    return pokemon_stats.values()

def create_pokemon_data_elo_threshold(json_data):
    elo_threshold_stats = {
        "BEGINNER": {},
        "POKEBALL": {},
        "GREATBALL": {},
        "ULTRABALL": {},
        "MASTERBALL": {} 
    }

    for threshold in ["BEGINNER", "POKEBALL", "GREATBALL", "ULTRABALL", "MASTERBALL"]:
        elo_threshold =  1400 if threshold == "MASTERBALL" else 1250 if threshold == "ULTRABALL" else 1100 if threshold == "GREATBALL" else 900 if threshold == "POKEBALL" else 0
        pokemon_stats = {}
        for pokemon in LIST_POKEMON:
            pokemon_stats[pokemon] = {"items": {}, "rank": 0, "count": 0, "name": pokemon, "item_count": 0}

        for match in json_data:
            for pokemon in match["pokemons"]:
                if pokemon["elo"] >= elo_threshold:
                    pokemon_stats[pokemon["name"]]["rank"] += match["rank"]
                    pokemon_stats[pokemon["name"]]["item_count"] += len(pokemon["items"])
                    pokemon_stats[pokemon["name"]]["count"] += 1
                    for item in pokemon["items"]:
                        if (item in pokemon_stats[pokemon["name"]]["items"]):
                            pokemon_stats[pokemon["name"]]["items"][item] += 1
                        else:
                            pokemon_stats[pokemon["name"]]["items"][item] = 1

        for pokemon in pokemon_stats:
            if (pokemon_stats[pokemon]["count"] == 0):
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

        elo_threshold_stats[threshold] = pokemon_stats
    return elo_threshold_stats.values()


def create_dataframe(json_data):
    list_match = []
    for i in range(len(json_data)):
        data = json_data[i]
        match = {}
        match["rank"] = data["rank"]
        match["nbplayers"] = data["nbplayers"] or 8 # nbplayers has been added later so need fallback value
        # for each pkm in the team
        pokemons = data["pokemons"]
        for j in range(len(pokemons)):
            pkm_name = pokemons[j]["name"]
            # increase number of pkm
            if (pkm_name != "SHEDNINJA"):
                if (pkm_name in match):
                    match[pkm_name] += 1
                else:
                    match[pkm_name] = 1
                    # increase number of pkm types
                    pkm_types = POKEMON_TYPE[pkm_name]
                    for type_name in pkm_types:
                        if (type_name in match):
                            match[type_name] += 1
                        else:
                            match[type_name] = 1
        list_match.append(match)

    dataframe = pd.DataFrame(list_match)
    dataframe.fillna(0, inplace=True)
    return dataframe


def apply_tsne(df, perplexity, n_iter=4000, plot=False):
    # apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                method="barnes_hut", init="pca", learning_rate="auto")
    df_result = pd.DataFrame(tsne.fit_transform(df), columns=["x", "y"])
    if (plot):
        # plot scatter point
        plt.scatter(df_result["x"], df_result["y"], color="black", alpha=0.4)
        plt.show()
    return df_result


def apply_clustering(df, epsilon, min_samples, plot=False):
    # apply DBSCAN on dataframe copy
    df_result = df.copy()
    cluster = DBSCAN(eps=epsilon, min_samples=min_samples).fit(df_result)
    cluster_id = [str(l) for l in cluster.labels_]
    df_result.insert(0, "cluster_id", cluster_id)
    if (plot):
        # plot scatter all points
        plt.scatter(df_result["x"], df_result["y"], color="black", alpha=.1)
        colors = ColorGenerator()
        list_cluster_id = set(list_cluster_id)
        if ('-1' in list_cluster_id):
            list_cluster_id.remove('-1')
        for cluster_id in list_cluster_id:
            df_partial = df_result[df_result["cluster_id"] == cluster_id]
            # plot scatter points of cluster
            plt.scatter(df_partial["x"], df_partial["y"],
                        alpha=.33, c=colors.next())
        plt.show()
    return df_result


def get_meta_report(df):
    n_row_total = df.shape[0]
    list_cluster_id = df["cluster_id"].unique().tolist()
    if '-1' in list_cluster_id:
        list_cluster_id.remove('-1')
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
        s_median_type = df_sub_cluster[[
            c for c in df_sub_cluster.columns if c in LIST_TYPE]].median()
        median_types = s_median_type[s_median_type > 1].to_dict()
        if not median_types:
            print(f"\tskip undefined cluster {cluster_id} with size {n_row}")
            continue
        meta_report["types"] = median_types
        s_mean_pkm = df_sub_cluster[[
            c for c in df_sub_cluster.columns if c in LIST_POKEMON]].mean()
        mean_pkm = s_mean_pkm[s_mean_pkm > .333].to_dict()
        meta_report["pokemons"] = {k: round(mean_pkm[k], 5) for k in mean_pkm}
        list_team = []
        for _, row in df_sub_cluster.iterrows():
            team_data = {}
            team_data["cluster_id"] = cluster_id
            team_data["rank"] = row["rank"]
            team_data["x"] = row["x"]
            team_data["y"] = row["y"]
            list_pkmn = row[[i for i in row.index if i in LIST_POKEMON]]
            team_data["pokemons"] = list_pkmn[list_pkmn != 0].to_dict()
            list_team.append(team_data)
        meta_report["teams"] = list_team
        meta_report["x"] = np.mean([k["x"] for k in list_team])
        meta_report["y"] = np.mean([k["y"] for k in list_team])
        list_meta_report.append(meta_report)
    return list_meta_report


def export_data_mongodb(list_data, db_name, collection_name):
    uri = os.getenv("MONGO_URI")
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[collection_name]
    collection.delete_many({})
    collection.insert_many(list_data)
    client.close()


def plot_cluster_parameters(df, list_sample, list_epsilon):
    # compute graph size
    n_sample = len(list_sample)
    n_epsilon = len(list_epsilon)
    n_plot = n_sample * n_epsilon
    _, ax = plt.subplots(n_sample, n_epsilon, figsize=[10, 10])
    for idx, (spl, eps) in enumerate(itools.product(list_sample, list_epsilon)):
        print(
            f"{datetime.now().time()} subplot {idx+1}/{n_plot} epsilon={eps} samples={spl} ...")
        # apply DBSCAN on dataframe copy
        df_cluster = df.copy()
        cluster = DBSCAN(eps=eps, min_samples=spl).fit(df_cluster)
        df_cluster.insert(0, "cluster_id", cluster.labels_)
        # plot all points in black
        sub_plt = ax[math.floor(idx / n_epsilon)][idx % n_epsilon]
        sub_plt.set_title(f"epsilon={eps} min_samples={spl}")
        sub_plt.scatter(df_cluster["x"],
                        df_cluster["y"], color="black", alpha=.1)
        colors = ColorGenerator()
        list_cluster_id = list(set(cluster.labels_))
        if (-1 in list_cluster_id):
            list_cluster_id.remove(-1)
        # for each cluster
        for cluster_id in list_cluster_id:
            df_sub_cluster = df_cluster[df_cluster["cluster_id"] == cluster_id]
            # plot scatter points of cluster
            sub_plt.scatter(
                df_sub_cluster["x"], df_sub_cluster["y"], alpha=.33, c=colors.next())
    plt.show()


def plot_tsne_parameters(df, list_perplexity):
    # compute graph size
    n_perplexity = len(list_perplexity)
    multi_axes = n_perplexity > 3
    n_rows = 2 if multi_axes else 1
    n_cols = math.floor(n_perplexity/2) if multi_axes else n_perplexity
    _, ax = plt.subplots(n_rows, n_cols, figsize=[10, 10])
    for idx, ppx in enumerate(list_perplexity):
        print(
            f"{datetime.now().time()} subplot {idx+1}/{n_perplexity} perplexity={ppx} ...")
        # apply t-SNE
        tsne = TSNE(n_components=2, perplexity=ppx, method="barnes_hut",
                    init="pca", n_iter=4000, learning_rate="auto")
        df_tsne = pd.DataFrame(tsne.fit_transform(df))
        # plot
        sub_plt = ax[math.floor(idx/n_cols)][idx %
                                             n_cols] if multi_axes else ax[idx]
        sub_plt.set_title(f"perplexity={ppx}")
        sub_plt.scatter(df_tsne[0], df_tsne[1], color="black", alpha=.33)
    plt.show()


def main():
    print(f"{datetime.now().time()} load data from MongoDB")
    time_now = math.floor(datetime.now().timestamp() * 1000)
    time_limit = time_now - 15 * (24 * 60 * 60 * 1000)
    lastRealease = math.floor(datetime(2022, 7, 7, 12, 0).timestamp() * 1000)
    limit = max(lastRealease, time_limit)
    json_data = load_data_mongodb(limit)

    print(f"{datetime.now().time()} creating item data...")
    items = create_item_data(json_data)
    export_data_mongodb(items, "test", "items-statistic")

    print(f"{datetime.now().time()} creating pokemon data...")
    pokemons = create_pokemon_data(json_data)
    export_data_mongodb(pokemons, "test", "pokemons-statistic")

    print(f"{datetime.now().time()} creating pokemon data...")
    pokemons = create_pokemon_data_elo_threshold(json_data)
    export_data_mongodb(pokemons, "test", "pokemons-statistic-v2")

    print(f"{datetime.now().time()} creating dataframe...")
    df_match = create_dataframe(json_data)

    print(f"{datetime.now().time()} applying t-SNE...")
    df_filtered = df_match[LIST_TYPE]
    df_tsne = apply_tsne(df_filtered, 20)
    #plot_tsne_parameters(df_filtered, [20,30,40,50])

    print(f"{datetime.now().time()} applying DBSCAN...")
    df_cluster = apply_clustering(df_tsne, 3.5, 20)
    #plot_cluster_parameters(df_tsne, [10,15,20], [3,3.5,4])

    print(f"{datetime.now().time()} create meta report...")
    df_concat = pd.concat([df_match, df_cluster], axis=1)
    report = get_meta_report(df_concat)

    print(f"{datetime.now().time()} write output file...")
    export_data_mongodb(report, "test", "meta")


if __name__ == "__main__":
    main()
