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
    plot_tsne_parameters,
    plot_tsne_init_comparison,
    plot_tsne_metric_comparison,
    plot_tsne_learning_rate_comparison,
    plot_tsne_early_exaggeration_comparison,
    apply_umap,
    plot_umap_parameters_grid,
    apply_pca,
    apply_phate,
    plot_phate_parameters_grid,
    plot_phate_mds_comparison,
    apply_isomap,
    plot_isomap_parameters,
    apply_spectral_embedding,
    plot_spectral_embedding_parameters_grid,
    apply_pacmap,
    plot_pacmap_parameters_grid,
    plot_pacmap_n_neighbors_grid,
    apply_trimap,
    plot_trimap_parameters_grid,
    plot_trimap_n_random_grid,
    apply_nmf,
    plot_nmf_parameters_grid,
    plot_nmf_init_comparison,
    plot_nmf_solver_comparison,
    plot_nmf_regularization_grid,
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


# Dimensionality reduction method registry
DIMENSIONALITY_METHODS = {
    "tsne": {
        "apply": apply_tsne,
        "default_params": {"perplexity": 55},
        "comparison_enabled": "SKIP_TSNE_COMPARISON",
        "compare_func": plot_tsne_parameters,
    },
    "umap": {
        "apply": apply_umap,
        "default_params": {"n_neighbors": 15, "min_dist": 0.1},
        "comparison_enabled": "SKIP_UMAP_COMPARISON",
        "compare_func": plot_umap_parameters_grid,
    },
    "pca": {
        "apply": apply_pca,
        "default_params": {},
        "comparison_enabled": "SKIP_PCA_COMPARISON",
        "compare_func": None,
    },
    "phate": {
        "apply": apply_phate,
        "default_params": {"k_neighbors": 40, "decay": 20},
        "comparison_enabled": "SKIP_PHATE_COMPARISON",
        "compare_func": plot_phate_parameters_grid,
    },
    "isomap": {
        "apply": apply_isomap,
        "default_params": {"n_neighbors": 15},
        "comparison_enabled": "SKIP_ISOMAP_COMPARISON",
        "compare_func": plot_isomap_parameters,
    },
    "spectral": {
        "apply": apply_spectral_embedding,
        "default_params": {"n_neighbors": 10, "affinity": "nearest_neighbors"},
        "comparison_enabled": "SKIP_SPECTRAL_COMPARISON",
        "compare_func": plot_spectral_embedding_parameters_grid,
    },
    "pacmap": {
        "apply": apply_pacmap,
        "default_params": {"n_neighbors": 20, "MN_ratio": 0.5, "FP_ratio": 7.0},
        "comparison_enabled": "SKIP_PACMAP_COMPARISON",
        "compare_func": plot_pacmap_parameters_grid,
    },
    "trimap": {
        "apply": apply_trimap,
        "default_params": {"n_inliers": 15, "n_outliers": 10, "n_random": 5},
        "comparison_enabled": "SKIP_TRIMAP_COMPARISON",
        "compare_func": plot_trimap_parameters_grid,
    },
    "nmf": {
        "apply": apply_nmf,
        "default_params": {"n_components": 2, "init": "random"},
        "comparison_enabled": "SKIP_NMF_COMPARISON",
        "compare_func": plot_nmf_parameters_grid,
    },
}

# Clustering method registry
CLUSTERING_METHODS = {
    "dbscan": {
        "apply": apply_clustering,
        # tsne-adapted parameters {"epsilon": 3, "min_samples": 20}
        # pacmap adapted parameters {"epsilon": 0.7, "min_samples": 8}
        "default_params": {"epsilon": 3, "min_samples": 20},
        "comparison_enabled": "SKIP_DBSCAN_COMPARISON",
        "compare_func": plot_cluster_parameters,
    },
    "kmeans": {
        "apply": apply_kmeans_clustering,
        "default_params": {"adaptive_k": "permissive"},
        "comparison_enabled": "SKIP_KMEANS_COMPARISON",
        "compare_func": plot_kmeans_parameters,
    },
}


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
    os.makedirs(results_dir, exist_ok=True)
    print(f"{datetime.now().time()} created results directory: {results_dir}")

    # Define subdirectory paths (will be created only when needed)
    method_dirs = {
        "tsne": os.path.join(results_dir, "tsne"),
        "umap": os.path.join(results_dir, "umap"),
        "pca": os.path.join(results_dir, "pca"),
        "phate": os.path.join(results_dir, "phate"),
        "isomap": os.path.join(results_dir, "isomap"),
        "spectral": os.path.join(results_dir, "spectral"),
        "pacmap": os.path.join(results_dir, "pacmap"),
        "trimap": os.path.join(results_dir, "trimap"),
        "nmf": os.path.join(results_dir, "nmf"),
        "dbscan": os.path.join(results_dir, "dbscan"),
        "kmeans": os.path.join(results_dir, "kmeans"),
    }

    # Parse environment variables
    debug_mode = os.environ.get("DEBUG", "false").lower() == "true"
    SAVE_PLOTS = os.environ.get("SAVE_PLOTS", "false").lower() == "true"

    # Get selected methods from environment variables (default to tsne + dbscan)
    dim_reduction_method = os.environ.get(
        "DIMENSIONALITY_METHOD", "tsne").lower()
    clustering_method = os.environ.get("CLUSTERING_METHOD", "dbscan").lower()

    # Validate method selections
    if dim_reduction_method not in DIMENSIONALITY_METHODS:
        print(f"{datetime.now().time()} warning: unknown dimensionality method '{dim_reduction_method}', using 'tsne'")
        dim_reduction_method = "tsne"

    if clustering_method not in CLUSTERING_METHODS:
        print(f"{datetime.now().time()} warning: unknown clustering method '{clustering_method}', using 'dbscan'")
        clustering_method = "dbscan"

    print(f"{datetime.now().time()} selected methods: dimensionality={dim_reduction_method}, clustering={clustering_method}")

    # Select only numeric columns (exclude non-numeric columns like pokemons and items, and metadata like elo)
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
        print(f"{datetime.now().time()} skipping analysis")
        return None

    print(
        f"{datetime.now().time()} using {len(numeric_cols)} synergy columns: {numeric_cols[:5]}..."
    )
    df_filtered = df_match[numeric_cols]

    # Apply dimensionality reduction method
    print(f"{datetime.now().time()} applying {dim_reduction_method}...")
    method_info = DIMENSIONALITY_METHODS[dim_reduction_method]
    method_dir = method_dirs[dim_reduction_method]
    skip_comparison_flag = method_info["comparison_enabled"]
    skip_comparison = os.environ.get(
        skip_comparison_flag, "true").lower() == "true"

    if SAVE_PLOTS or not skip_comparison:
        os.makedirs(method_dir, exist_ok=True)

    # Apply the main dimensionality reduction
    df_reduced = method_info["apply"](
        df_filtered, save=SAVE_PLOTS, output_dir=method_dir, **method_info["default_params"]
    )

    # Run parameter comparisons if enabled and comparison function exists
    if not skip_comparison and method_info["compare_func"] is not None:
        print(
            f"{datetime.now().time()} running {dim_reduction_method} parameter comparison...")

        if dim_reduction_method == "tsne":
            perplexity_nums = [40, 45, 50, 55, 60, 65]
            method_info["compare_func"](
                df_filtered, perplexity_nums, output_dir=method_dir)
            # Run additional t-SNE comparisons
            plot_tsne_init_comparison(
                df_filtered, ["pca", "random"], output_dir=method_dir)
            plot_tsne_metric_comparison(
                df_filtered, ["euclidean", "cosine", "manhattan"], output_dir=method_dir)
            # plot_tsne_learning_rate_comparison(df_filtered, [
            #                                   "auto", 200, 500, 800], method_info["default_params"]["perplexity"], output_dir=method_dir)
            # plot_tsne_early_exaggeration_comparison(df_filtered, [
            #                                        8, 12, 16, 20], method_info["default_params"]["perplexity"], output_dir=method_dir)

        elif dim_reduction_method == "umap":
            n_neighbors_values = [5, 10, 15, 30, 60, 100, 150, 200]
            min_dist_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
            method_info["compare_func"](
                df_filtered, n_neighbors_values, min_dist_values, output_dir=method_dir)

        elif dim_reduction_method == "phate":
            k_neighbors_values = [15, 20, 30, 40, 50]
            decay_values = [5, 10, 15, 20, 25]
            method_info["compare_func"](
                df_filtered, k_neighbors_values, decay_values, output_dir=method_dir)
            plot_phate_mds_comparison(df_filtered, knn=method_info["default_params"]["k_neighbors"],
                                      decay=method_info["default_params"]["decay"], output_dir=method_dir)

        elif dim_reduction_method == "isomap":
            n_neighbors_values = [5, 10, 15, 20, 25]
            method_info["compare_func"](
                df_filtered, n_neighbors_values, output_dir=method_dir)

        elif dim_reduction_method == "spectral":
            n_neighbors_values = [5, 10, 15]
            affinity_values = ['nearest_neighbors', 'rbf']
            method_info["compare_func"](
                df_filtered, n_neighbors_values, affinity_values, output_dir=method_dir)

        elif dim_reduction_method == "pacmap":
            n_neighbors_values = [20, 25, 30, 35]
            MN_ratio_values = [0.3, 0.5, 0.7, 0.9]
            FP_ratio_values = [5.0, 6.0, 7.0, 8.0]
            plot_pacmap_n_neighbors_grid(
                df_filtered, n_neighbors_values, MN_ratio_values, FP_ratio_values, output_dir=method_dir)

        elif dim_reduction_method == "trimap":
            n_random_values = [2, 3, 4]
            n_inliers_values = [8, 12, 16]
            n_outliers_values = [2, 4, 6]
            plot_trimap_n_random_grid(
                df_filtered, n_random_values, n_inliers_values, n_outliers_values, output_dir=method_dir)

        elif dim_reduction_method == "nmf":
            init_values = ['random', 'nndsvd', 'nndsvda', 'nndsvdar']
            solver_values = ['cd', 'mu']
            alpha_W_values = [0.0, 0.1, 0.5]
            beta_loss_values = ['frobenius',
                                'kullback-leibler', 'itakura-saito']

            # Plot comprehensive 4x2 grid: init methods × solvers
            print(
                f"{datetime.now().time()} NMF: generating 4x2 parameters grid (init methods × solvers)...")
            plot_nmf_parameters_grid(df_filtered, init_values=init_values,
                                     solver_values=solver_values, output_dir=method_dir)

            # Plot 3x3 grid: regularization × loss functions
            print(
                f"{datetime.now().time()} NMF: generating 3x3 regularization grid (alpha_W × beta_loss)...")
            plot_nmf_regularization_grid(df_filtered, init_method='nndsvda',
                                         alpha_W_values=alpha_W_values,
                                         beta_loss_values=beta_loss_values, output_dir=method_dir)

            # Plot 1x4 init comparison
            print(f"{datetime.now().time()} NMF: generating init comparison plot...")
            plot_nmf_init_comparison(
                df_filtered, init_values=init_values, output_dir=method_dir)

            # Plot 1x2 solver comparison
            print(
                f"{datetime.now().time()} NMF: generating solver comparison plot...")
            plot_nmf_solver_comparison(
                df_filtered, solver_values=solver_values, output_dir=method_dir)

    # Apply clustering method
    clustering_info = CLUSTERING_METHODS[clustering_method]
    clustering_dir = method_dirs[clustering_method]
    skip_clustering_comparison_flag = clustering_info["comparison_enabled"]
    skip_clustering_comparison = os.environ.get(
        skip_clustering_comparison_flag, "true").lower() == "true"

    if SAVE_PLOTS or not skip_clustering_comparison:
        os.makedirs(clustering_dir, exist_ok=True)

    print(f"{datetime.now().time()} applying {clustering_method}...")

    if clustering_method == "dbscan":
        df_cluster = clustering_info["apply"](
            df_reduced,
            epsilon=clustering_info["default_params"]["epsilon"],
            min_samples=clustering_info["default_params"]["min_samples"],
            save=SAVE_PLOTS,
            output_dir=clustering_dir
        )
    elif clustering_method == "kmeans":
        adaptive_k = get_adaptive_k_clusters(
            method=clustering_info["default_params"]["adaptive_k"])
        df_cluster = clustering_info["apply"](
            df_reduced,
            adaptive_k,
            save=SAVE_PLOTS,
            output_dir=clustering_dir
        )

    # Run parameter comparisons if enabled
    if not skip_clustering_comparison and clustering_info["compare_func"] is not None:
        print(
            f"{datetime.now().time()} running {clustering_method} parameter comparison...")

        if clustering_method == "dbscan":
            min_samples_values = [10, 15, 20, 25]
            epsilon_values = [1.0, 3.0, 5.0]
            clustering_info["compare_func"](
                df_reduced, min_samples_values, epsilon_values, output_dir=clustering_dir)

        elif clustering_method == "kmeans":
            k_values = [
                get_adaptive_k_clusters("permissive"),
                get_adaptive_k_clusters("aggressive"),
                get_adaptive_k_clusters("balanced"),
                get_adaptive_k_clusters("conservative"),
            ]
            clustering_info["compare_func"](
                df_reduced, k_values, output_dir=clustering_dir)

    print(f"{datetime.now().time()} creating meta report...")
    # Build the final dataframe with all required columns
    df_concat = df_match.copy()
    df_concat["x"] = df_reduced["x"]
    df_concat["y"] = df_reduced["y"]
    df_concat["cluster_id"] = df_cluster["cluster_id"]
    report = get_meta_report(df_concat)

    # Export reports: debug mode uses text/json/visualization files, prod uses MongoDB
    if debug_mode:
        print(f"{datetime.now().time()} generating meta report outputs...")
        export_meta_report_text(report, results_dir)
        export_meta_report_json(report, results_dir)
        visualize_meta_report(report, results_dir)

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
    time_limit = time_now - 15 * (24 * 60 * 60 * 1000)

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
