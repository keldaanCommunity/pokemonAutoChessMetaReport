"""UMAP dimensionality reduction methods"""

import umap
import itertools as itools
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')


def apply_umap(df, n_neighbors=15, min_dist=0.1, plot=False, save=False, output_dir=None):
    """
    Apply UMAP dimensionality reduction to reduce high-dimensional data to 2D.

    UMAP is often faster than t-SNE and preserves more global structure.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features to reduce
        n_neighbors (int): Size of local neighborhood for UMAP (default: 15)
        min_dist (float): Minimum distance between points in embedding (default: 0.1)
        plot (bool): Whether to display the plot interactively (default: False)
        save (bool): Whether to save the plot to file (default: False)
        output_dir (str): Directory to save the plot if save=True (default: None)

    Returns:
        pd.DataFrame: DataFrame with 'x' and 'y' columns containing 2D UMAP coordinates
    """
    print(f"{datetime.now().time()} UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                        metric="cosine", init="random", verbose=False)
    df_result = pd.DataFrame(reducer.fit_transform(df), columns=["x", "y"])

    if plot or save:
        plt.figure(figsize=(10, 8))
        plt.scatter(df_result["x"], df_result["y"],
                    color="black", alpha=0.4, s=20)
        plt.title(
            f"UMAP Visualization (n_neighbors={n_neighbors}, min_dist={min_dist})")
        plt.xlabel("UMAP Component 1")
        plt.ylabel("UMAP Component 2")
        plt.grid(True, alpha=0.3)
        if save and output_dir:
            filepath = os.path.join(
                output_dir, f"umap_neighbors_{n_neighbors}_mindist_{min_dist}.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"{datetime.now().time()} saved plot to {filepath}")
        if plot:
            plt.show()
        plt.close()
    return df_result


def plot_umap_parameters_grid(df, list_n_neighbors, list_min_dist, output_dir=None):
    """
    Create a grid of subplots showing UMAP results with different n_neighbors and min_dist combinations.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        list_n_neighbors (list): List of n_neighbors values to test
        list_min_dist (list): List of min_dist values to test
        output_dir (str): Directory to save the grid plot if provided (default: None)

    Returns:
        None: Displays or saves the grid of UMAP visualizations
    """
    n_neighbors_count = len(list_n_neighbors)
    min_dist_count = len(list_min_dist)
    n_plots = n_neighbors_count * min_dist_count

    fig, ax = plt.subplots(min_dist_count, n_neighbors_count, figsize=[18, 14])

    for idx, (min_dist, n_neigh) in enumerate(itools.product(list_min_dist, list_n_neighbors)):
        print(f"{datetime.now().time()} subplot {idx+1}/{n_plots} n_neighbors={n_neigh}, min_dist={min_dist} ...")

        reducer = umap.UMAP(n_components=2, n_neighbors=n_neigh, min_dist=min_dist,
                            metric="cosine", init="random", n_jobs=1, random_state=42, verbose=False)
        df_umap = pd.DataFrame(reducer.fit_transform(df), columns=[0, 1])

        row = list_min_dist.index(min_dist)
        col = list_n_neighbors.index(n_neigh)
        sub_plt = ax[row][col]
        sub_plt.set_title(f"n_neigh={n_neigh}, min_dist={min_dist}")
        sub_plt.scatter(df_umap[0], df_umap[1], color="black", alpha=.2, s=5)
        sub_plt.grid(True, alpha=0.3)
        sub_plt.axis('off')

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "umap_parameters_grid.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved UMAP parameters grid to {filepath}")
        plt.close()
    else:
        plt.show()
