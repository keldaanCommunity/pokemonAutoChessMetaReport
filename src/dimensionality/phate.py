"""PHATE dimensionality reduction methods"""

import phate
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')


def apply_phate(df, k_neighbors=15, decay=15, plot=False, save=False, output_dir=None):
    """
    Apply PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding) 
    to reduce high-dimensional data to 2D.

    PHATE balances local and global structure preservation, making it excellent for 
    visualizing data with natural progressions or evolving meta-states (like game strategies).

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features to reduce
        k_neighbors (int): Number of nearest neighbors (knn) for k-NN graph construction (default: 15)
        decay (int): Decay parameter for the heat kernel diffusion (default: 15, higher = smoother transitions)
        plot (bool): Whether to display the plot interactively (default: False)
        save (bool): Whether to save the plot to file (default: False)
        output_dir (str): Directory to save the plot if save=True (default: None)

    Returns:
        pd.DataFrame: DataFrame with 'x' and 'y' columns containing 2D PHATE coordinates
    """
    print(f"{datetime.now().time()} applying PHATE with knn={k_neighbors}, decay={decay}...")
    phate_op = phate.PHATE(
        n_components=2, knn=k_neighbors, decay=decay, n_jobs=-1)
    df_result = pd.DataFrame(phate_op.fit_transform(df), columns=["x", "y"])

    if plot or save:
        plt.figure(figsize=(10, 8))
        plt.scatter(df_result["x"], df_result["y"],
                    color="black", alpha=0.4, s=20)
        plt.title(
            f"PHATE Visualization (knn={k_neighbors}, decay={decay})")
        plt.xlabel("PHATE Component 1")
        plt.ylabel("PHATE Component 2")
        plt.grid(True, alpha=0.3)
        if save and output_dir:
            filepath = os.path.join(
                output_dir, f"phate_k{k_neighbors}_d{decay}.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"{datetime.now().time()} saved PHATE plot to {filepath}")
        if plot:
            plt.show()
        plt.close()
    return df_result


def plot_phate_parameters_grid(df, k_neighbors_values, decay_values, output_dir=None):
    """
    Grid search for PHATE parameters k_neighbors and decay.

    Creates a grid of subplots showing PHATE projections for different parameter combinations.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        k_neighbors_values (list): List of k_neighbors values to test
        decay_values (list): List of decay values to test
        output_dir (str): Directory to save the grid plot if provided (default: None)
    """
    n_k = len(k_neighbors_values)
    n_decay = len(decay_values)

    fig, ax_flat = plt.subplots(n_decay, n_k, figsize=(4*n_k, 4*n_decay))
    if n_decay == 1 or n_k == 1:
        ax_flat = ax_flat.reshape(
            n_decay, n_k) if n_decay > 1 else ax_flat.reshape(1, -1)

    total_combinations = n_k * n_decay
    print(f"{datetime.now().time()} testing {total_combinations} PHATE parameter combinations...")

    for idx_decay, decay in enumerate(decay_values):
        for idx_k, k in enumerate(k_neighbors_values):
            print(f"{datetime.now().time()} PHATE: knn={k}, decay={decay}...")
            try:
                phate_op = phate.PHATE(
                    n_components=2, knn=k, decay=decay, n_jobs=-1)
                df_phate = pd.DataFrame(
                    phate_op.fit_transform(df), columns=[0, 1])

                ax = ax_flat[idx_decay, idx_k]
                ax.set_title(f"k={k}, decay={decay}")
                ax.scatter(df_phate[0], df_phate[1],
                           color="black", alpha=0.33, s=3)
                ax.grid(True, alpha=0.3)
            except Exception as e:
                print(
                    f"{datetime.now().time()} Error with k={k}, decay={decay}: {e}")
                ax = ax_flat[idx_decay, idx_k]
                ax.text(
                    0.5, 0.5, f"Error:\n{str(e)[:30]}", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "phate_parameters_grid.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved PHATE parameters grid to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_phate_mds_comparison(df, knn=15, decay=15, output_dir=None):
    """
    Compare different MDS methods for PHATE.

    Creates a grid of 3 subplots showing PHATE projections with different MDS algorithms.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        knn (int): Number of nearest neighbors (default: 15)
        decay (int): Decay parameter for the heat kernel (default: 15)
        output_dir (str): Directory to save the plot if provided (default: None)
    """
    mds_methods = ['classic', 'metric', 'nonmetric']
    fig, ax_flat = plt.subplots(1, 3, figsize=(15, 4))

    total_methods = len(mds_methods)
    print(f"{datetime.now().time()} testing {total_methods} PHATE MDS methods with knn={knn}, decay={decay}...")

    for idx, mds_method in enumerate(mds_methods):
        print(f"{datetime.now().time()} PHATE: mds={mds_method}...")
        try:
            phate_op = phate.PHATE(
                n_components=2, knn=knn, decay=decay, mds=mds_method, n_jobs=-1)
            df_phate = pd.DataFrame(
                phate_op.fit_transform(df), columns=[0, 1])

            ax = ax_flat[idx]
            ax.set_title(f"mds={mds_method}")
            ax.scatter(df_phate[0], df_phate[1],
                       color="black", alpha=0.33, s=3)
            ax.grid(True, alpha=0.3)
        except Exception as e:
            print(
                f"{datetime.now().time()} Error with mds={mds_method}: {e}")
            ax = ax_flat[idx]
            ax.text(
                0.5, 0.5, f"Error:\n{str(e)[:30]}", ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(
            output_dir, f"phate_mds_comparison_k{knn}_d{decay}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved PHATE MDS comparison to {filepath}")
        plt.close()
    else:
        plt.show()
