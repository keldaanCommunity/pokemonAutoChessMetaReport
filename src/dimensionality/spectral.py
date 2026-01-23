"""Spectral Embedding dimensionality reduction methods"""

from sklearn.manifold import SpectralEmbedding
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')


def apply_spectral_embedding(df, n_neighbors=10, affinity='nearest_neighbors', plot=False, save=False, output_dir=None):
    """
    Apply Spectral Embedding (Laplacian Eigenmaps) to reduce high-dimensional data to 2D.

    Spectral Embedding is a non-linear manifold learning method that preserves local structure
    by using the eigenvectors of the graph Laplacian. No external dependencies required.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features to reduce
        n_neighbors (int): Number of nearest neighbors (default: 10, range: 5-20)
        affinity (str): Method for affinity matrix ('nearest_neighbors' or 'rbf', default: 'nearest_neighbors')
        plot (bool): Whether to display the plot interactively (default: False)
        save (bool): Whether to save the plot to file (default: False)
        output_dir (str): Directory to save the plot if save=True (default: None)

    Returns:
        pd.DataFrame: DataFrame with 'x' and 'y' columns containing 2D Spectral Embedding coordinates
    """
    print(f"{datetime.now().time()} Spectral Embedding with n_neighbors={n_neighbors}, affinity={affinity}...")
    se_model = SpectralEmbedding(
        n_components=2,
        n_neighbors=n_neighbors,
        affinity=affinity,
        n_jobs=-1,
        random_state=42
    )
    df_result = pd.DataFrame(
        se_model.fit_transform(df), columns=["x", "y"])

    if plot or save:
        plt.figure(figsize=(10, 8))
        plt.scatter(df_result["x"], df_result["y"],
                    color="black", alpha=0.4, s=20)
        plt.title(
            f"Spectral Embedding Visualization (n_neighbors={n_neighbors}, affinity={affinity})")
        plt.xlabel("Spectral Component 1")
        plt.ylabel("Spectral Component 2")
        plt.grid(True, alpha=0.3)
        if save and output_dir:
            filepath = os.path.join(
                output_dir, f"spectral_nn{n_neighbors}_{affinity}.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"{datetime.now().time()} saved plot to {filepath}")
        if plot:
            plt.show()
        plt.close()
    return df_result


def plot_spectral_embedding_parameters_grid(df, n_neighbors_values, affinity_values, output_dir=None):
    """
    Grid search for Spectral Embedding parameters: n_neighbors and affinity.

    Creates a grid of subplots showing Spectral Embedding projections for different parameter combinations.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        n_neighbors_values (list): List of n_neighbors values to test (e.g., [5, 10, 15])
        affinity_values (list): List of affinity methods to test (e.g., ['nearest_neighbors', 'rbf'])
        output_dir (str): Directory to save the grid plot if provided (default: None)
    """
    total_combinations = len(n_neighbors_values) * len(affinity_values)
    print(f"{datetime.now().time()} testing {total_combinations} Spectral Embedding parameter combinations...")

    fig, ax_flat = plt.subplots(len(affinity_values), len(n_neighbors_values),
                                figsize=(4*len(n_neighbors_values), 4*len(affinity_values)))
    if total_combinations == 1:
        ax_flat = [[ax_flat]]
    elif len(affinity_values) == 1:
        ax_flat = [ax_flat]

    plot_idx = 0
    for idx_aff, affinity in enumerate(affinity_values):
        for idx_nn, n_neighbors in enumerate(n_neighbors_values):
            print(f"{datetime.now().time()} Spectral {plot_idx+1}/{total_combinations}: n_neighbors={n_neighbors}, affinity={affinity}...")

            try:
                se_model = SpectralEmbedding(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    affinity=affinity,
                    n_jobs=-1,
                    random_state=42
                )
                df_se = pd.DataFrame(
                    se_model.fit_transform(df), columns=[0, 1])

                if len(affinity_values) > 1:
                    ax = ax_flat[idx_aff][idx_nn]
                else:
                    ax = ax_flat[0][idx_nn]
                ax.set_title(f"nn={n_neighbors}, aff={affinity}")
                ax.scatter(df_se[0], df_se[1],
                           color="black", alpha=0.33, s=3)
                ax.grid(True, alpha=0.3)
            except Exception as e:
                print(
                    f"{datetime.now().time()} Error with n_neighbors={n_neighbors}, affinity={affinity}: {e}")
                if len(affinity_values) > 1:
                    ax = ax_flat[idx_aff][idx_nn]
                else:
                    ax = ax_flat[0][idx_nn]
                ax.text(
                    0.5, 0.5, f"Error:\n{str(e)[:30]}", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])

            plot_idx += 1

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(
            output_dir, "spectral_embedding_parameters_grid.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(
            f"{datetime.now().time()} saved Spectral Embedding parameters grid to {filepath}")
        plt.close()
    else:
        plt.show()
