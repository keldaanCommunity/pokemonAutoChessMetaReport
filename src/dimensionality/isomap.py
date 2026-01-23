"""Isomap dimensionality reduction methods"""

from sklearn.manifold import Isomap
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')


def apply_isomap(df, n_neighbors=5, plot=False, save=False, output_dir=None):
    """
    Apply Isomap (Isometric Mapping) dimensionality reduction to reduce high-dimensional data to 2D.

    Isomap is a non-linear dimensionality reduction method that preserves geodesic distances
    along the manifold. It works by constructing a k-nearest neighbor graph and computing
    shortest paths on that graph.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features to reduce
        n_neighbors (int): Number of nearest neighbors for graph construction (default: 5)
        plot (bool): Whether to display the plot interactively (default: False)
        save (bool): Whether to save the plot to file (default: False)
        output_dir (str): Directory to save the plot if save=True (default: None)

    Returns:
        pd.DataFrame: DataFrame with 'x' and 'y' columns containing 2D Isomap coordinates
    """
    print(f"{datetime.now().time()} applying Isomap with n_neighbors={n_neighbors}...")
    isomap = Isomap(n_components=2, n_neighbors=n_neighbors, n_jobs=-1)
    df_result = pd.DataFrame(isomap.fit_transform(df), columns=["x", "y"])

    if plot or save:
        plt.figure(figsize=(10, 8))
        plt.scatter(df_result["x"], df_result["y"],
                    color="black", alpha=0.4, s=20)
        plt.title(f"Isomap Visualization (n_neighbors={n_neighbors})")
        plt.xlabel("Isomap Component 1")
        plt.ylabel("Isomap Component 2")
        plt.grid(True, alpha=0.3)
        if save and output_dir:
            filepath = os.path.join(
                output_dir, f"isomap_neighbors_{n_neighbors}.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"{datetime.now().time()} saved Isomap plot to {filepath}")
        if plot:
            plt.show()
        plt.close()
    return df_result


def plot_isomap_parameters(df, n_neighbors_values, output_dir=None):
    """
    Grid search for Isomap n_neighbors parameter.

    Creates a row of subplots showing Isomap projections for different n_neighbors values.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        n_neighbors_values (list): List of n_neighbors values to test
        output_dir (str): Directory to save the grid plot if provided (default: None)
    """
    n_values = len(n_neighbors_values)
    fig, ax_flat = plt.subplots(1, n_values, figsize=(5*n_values, 4))
    if n_values == 1:
        ax_flat = [ax_flat]

    print(f"{datetime.now().time()} testing {n_values} Isomap n_neighbors values...")

    for idx, n_neighbors in enumerate(n_neighbors_values):
        print(f"{datetime.now().time()} Isomap: n_neighbors={n_neighbors}...")
        try:
            isomap = Isomap(n_components=2, n_neighbors=n_neighbors, n_jobs=-1)
            df_isomap = pd.DataFrame(
                isomap.fit_transform(df), columns=[0, 1])

            ax = ax_flat[idx]
            ax.set_title(f"n_neighbors={n_neighbors}")
            ax.scatter(df_isomap[0], df_isomap[1],
                       color="black", alpha=0.33, s=3)
            ax.grid(True, alpha=0.3)
        except Exception as e:
            print(
                f"{datetime.now().time()} Error with n_neighbors={n_neighbors}: {e}")
            ax = ax_flat[idx]
            ax.text(
                0.5, 0.5, f"Error:\n{str(e)[:30]}", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "isomap_parameters_grid.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved Isomap parameters grid to {filepath}")
        plt.close()
    else:
        plt.show()
