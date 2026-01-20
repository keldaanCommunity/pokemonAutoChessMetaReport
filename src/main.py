#!/usr/bin/python3

"""Main entry point for Pokemon Auto Chess Meta Report analysis"""

import os
import math
import pandas as pd
from datetime import datetime

from .utils import DB_NAME
from .data_loader import (
    load_data_mongodb,
    create_dataframe,
    create_item_data_elo_threshold,
    create_pokemon_data_elo_threshold,
    create_region_data,
)
from .dimensionality import (
    apply_tsne,
    get_adaptive_perplexity,
    plot_tsne_parameters,
    plot_tsne_init_comparison,
    plot_tsne_metric_comparison,
    plot_tsne_learning_rate_comparison,
    plot_tsne_early_exaggeration_comparison,
    apply_umap,
    plot_umap_parameters_grid,
)
from .clustering import (
    apply_clustering,
    get_adaptive_min_samples,
    plot_cluster_parameters,
    apply_kmeans_clustering,
    get_adaptive_k_clusters,
    plot_kmeans_parameters,
)
from .reporting import (
    get_meta_report,
    create_metadata,
    export_data_mongodb,
    export_meta_report_text,
    export_meta_report_json,
    visualize_meta_report,
    export_meta_report_mongodb,
)


def run_analysis(json_data, elo_threshold=None):
    """
    Run the complete meta analysis pipeline for 1100+ ELO tier.

    Args:
        json_data (list): Match documents from MongoDB
        elo_threshold (int): Minimum ELO to include (default: None, uses 1100+)

    Returns:
        str: Path to results directory
    """
    # Filter data by ELO if threshold provided
    if elo_threshold is not None:
        json_data_filtered = [
            m for m in json_data if m.get("elo", 0) >= elo_threshold]
        print(
            f"{datetime.now().time()} filtered to {len(json_data_filtered)} matches with elo >= {elo_threshold}"
        )
    else:
        json_data_filtered = json_data

    if len(json_data_filtered) < 10:
        print(
            f"{datetime.now().time()} skipping 1100+ tier: insufficient data ({len(json_data_filtered)} matches)"
        )
        return None

    print(f"{datetime.now().time()} creating dataframe for 1100+ tier...")
    df_match = create_dataframe(json_data_filtered)

    # Create results directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    tsne_dir = os.path.join(results_dir, "tsne")
    umap_dir = os.path.join(results_dir, "umap")
    dbscan_dir = os.path.join(results_dir, "dbscan")
    kmeans_dir = os.path.join(results_dir, "kmeans")
    os.makedirs(tsne_dir, exist_ok=True)
    os.makedirs(umap_dir, exist_ok=True)
    os.makedirs(dbscan_dir, exist_ok=True)
    os.makedirs(kmeans_dir, exist_ok=True)
    print(f"{datetime.now().time()} created results directory: {results_dir}")

    # Testing flags
    SKIP_TSNE_COMPARISON = (
        os.environ.get("SKIP_TSNE_COMPARISON", "true").lower() == "true"
    )
    SKIP_UMAP_COMPARISON = (
        os.environ.get("SKIP_UMAP_COMPARISON", "true").lower() == "true"
    )
    SKIP_DBSCAN_COMPARISON = (
        os.environ.get("SKIP_DBSCAN_COMPARISON", "true").lower() == "true"
    )
    SKIP_KMEANS_COMPARISON = (
        os.environ.get("SKIP_KMEANS_COMPARISON", "true").lower() == "true"
    )
    SAVE_PLOTS = os.environ.get("SAVE_PLOTS", "false").lower() == "true"

    print(f"{datetime.now().time()} applying t-SNE...")
    # Select only numeric columns for t-SNE (exclude non-numeric columns like pokemons and items, and metadata like elo)
    numeric_cols = [
        col
        for col in df_match.columns
        if col not in ["rank", "nbplayers", "pokemons", "items", "elo"]
        and pd.api.types.is_numeric_dtype(df_match[col])
    ]

    if not numeric_cols:
        print(
            f"{datetime.now().time()} warning: no synergy columns found. Available columns: {list(df_match.columns)}"
        )
        print(f"{datetime.now().time()} skipping t-SNE analysis")
        return None

    print(
        f"{datetime.now().time()} using {len(numeric_cols)} synergy columns for t-SNE: {numeric_cols[:5]}..."
    )
    df_filtered = df_match[numeric_cols]

    perplexity = 55
    df_tsne = apply_tsne(df_filtered, perplexity,
                         save=SAVE_PLOTS, output_dir=tsne_dir)

    if not SKIP_TSNE_COMPARISON:
        print(f"{datetime.now().time()} comparing t-SNE perplexity values...")
        perplexity_nums = [30, 35, 40, 45, 50, 55, 60, 70]
        print(f"{datetime.now().time()} testing perplexity values: {perplexity_nums}")
        plot_tsne_parameters(df_filtered, perplexity_nums, output_dir=tsne_dir)

        # print(f"{datetime.now().time()} comparing t-SNE initialization methods...")
        # plot_tsne_init_comparison(
        #     df_filtered, ["pca", "random"], output_dir=tsne_dir)

        # print(f"{datetime.now().time()} comparing t-SNE distance metrics...")
        # plot_tsne_metric_comparison(
        #     df_filtered, ["euclidean", "cosine", "manhattan"], output_dir=tsne_dir
        # )

        # print(f"{datetime.now().time()} comparing t-SNE learning rates...")
        # plot_tsne_learning_rate_comparison(
        #     df_filtered,
        #     ["auto", 200, 500, 800],
        #     adaptive_perplexity,
        #     output_dir=tsne_dir,
        # )

        # print(f"{datetime.now().time()} comparing t-SNE early exaggeration values...")
        # plot_tsne_early_exaggeration_comparison(
        #     df_filtered, [8, 12, 16, 20], adaptive_perplexity, output_dir=tsne_dir
        # )

    # min_dist = 0.01
    # n_neighbors = 15
    # print(f"{datetime.now().time()} applying UMAP...")
    # df_umap = apply_umap(
    #     df_filtered, n_neighbors, min_dist, save=SAVE_PLOTS, output_dir=umap_dir
    # )

    if not SKIP_UMAP_COMPARISON:
        print(f"{datetime.now().time()} grid search for UMAP parameters...")
        n_neighbors_values = [5, 10, 15, 30, 60, 100, 150, 200]
        min_dist_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        print(
            f"{datetime.now().time()} testing n_neighbors values: {n_neighbors_values}"
        )
        print(f"{datetime.now().time()} testing min_dist values: {min_dist_values}")
        plot_umap_parameters_grid(
            df_filtered, n_neighbors_values, min_dist_values, output_dir=umap_dir
        )

    epsilon = 3
    min_samples = 20
    print(f"{datetime.now().time()} applying DBSCAN...")
    df_cluster = apply_clustering(
        df_tsne, epsilon, min_samples, save=SAVE_PLOTS, output_dir=dbscan_dir
    )

    if not SKIP_DBSCAN_COMPARISON:
        print(f"{datetime.now().time()} grid search for DBSCAN parameters...")
        min_samples_values = [15, 20, 30, 35]
        epsilon_values = [2.25, 2.5, 2.75, 3, 3.25, 3.5]
        print(
            f"{datetime.now().time()} testing min_samples values: {min_samples_values}"
        )
        plot_cluster_parameters(
            df_tsne, min_samples_values, epsilon_values, output_dir=dbscan_dir
        )

    # print(f"{datetime.now().time()} applying K-Means...")
    # adaptive_k = get_adaptive_k_clusters(n_samples, method="permissive")
    # df_kmeans = apply_kmeans_clustering(df_tsne, adaptive_k, save=SAVE_PLOTS, output_dir=kmeans_dir)

    if not SKIP_KMEANS_COMPARISON:
        print(f"{datetime.now().time()} grid search for K-Means parameters...")
        k_values = [
            get_adaptive_k_clusters("permissive"),
            get_adaptive_k_clusters("aggressive"),
            get_adaptive_k_clusters("balanced"),
            get_adaptive_k_clusters("conservative"),
        ]
        print(f"{datetime.now().time()} testing k values: {k_values}")
        plot_kmeans_parameters(df_tsne, k_values, output_dir=kmeans_dir)

    print(f"{datetime.now().time()} create meta report...")
    # Build the final dataframe with all required columns
    df_concat = df_match.copy()
    df_concat["x"] = df_tsne["x"]
    df_concat["y"] = df_tsne["y"]
    df_concat["cluster_id"] = df_cluster["cluster_id"]
    report = get_meta_report(df_concat)

    # Export reports: debug mode uses text/json/visualization files, prod uses MongoDB
    debug_mode = os.environ.get("DEBUG", "false").lower() == "true"

    if debug_mode:
        print(f"{datetime.now().time()} generating meta report outputs...")
        export_meta_report_text(report, results_dir)
        export_meta_report_json(report, results_dir)
        visualize_meta_report(report, results_dir)
    else:
        print(f"{datetime.now().time()} exporting meta report to MongoDB...")
        export_meta_report_mongodb(report, DB_NAME)

    print(
        f"{datetime.now().time()} analysis complete for 1100+ tier! Results saved in: {results_dir}"
    )
    return results_dir


def main():
    """Main entry point for Pokemon Auto Chess Meta Report analysis"""
    print(f"{datetime.now().time()} load data from MongoDB")
    time_now = math.floor(datetime.now().timestamp() * 1000)
    time_limit = time_now - 55 * (24 * 60 * 60 * 1000)

    debug_limit = os.environ.get("DEBUG_LIMIT")
    debug_limit = int(debug_limit) if debug_limit else None
    json_data = load_data_mongodb(time_limit, limit=debug_limit)
    print(f"{datetime.now().time()} loaded {len(json_data)} documents")

    print(f"{datetime.now().time()} creating metadata...")
    metadata = create_metadata(json_data, time_limit)
    export_data_mongodb(metadata, DB_NAME, "metadata")

    print(f"{datetime.now().time()} creating item data with threshold...")
    items = create_item_data_elo_threshold(json_data)
    export_data_mongodb(items, DB_NAME, "items-statistic-v2")

    print(f"{datetime.now().time()} creating pokemon data with threshold...")
    pokemons = create_pokemon_data_elo_threshold(json_data)
    export_data_mongodb(pokemons, DB_NAME, "pokemons-statistic-v2")

    # print(f"{datetime.now().time()} creating region data...")
    # regions = create_region_data(json_data)
    # export_data_mongodb(regions, DB_NAME, "regions-statistic")

    # Define ELO tier to analyze (1100+ only)
    elo_threshold = 1100

    try:
        run_analysis(json_data, elo_threshold)
    except Exception as e:
        print(f"{datetime.now().time()} error during analysis: {e}")


if __name__ == "__main__":
    main()
