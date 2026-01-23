"""PaCMAP dimensionality reduction methods"""

import pacmap
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')


def apply_pacmap(df, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, plot=False, save=False, output_dir=None):
    """
    Apply PaCMAP (Pairwise Controlled Manifold Approximation Projection) 
    to reduce high-dimensional data to 2D.

    PaCMAP is fast and produces high-quality embeddings by carefully controlling 
    the balance between local and global structure preservation.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features to reduce
        n_neighbors (int): Number of nearest neighbors for local structure (default: 10, range: 5-20)
        MN_ratio (float): Ratio of mid-near pairs to nearest neighbor pairs for global structure (default: 0.5, range: 0.1-1.0)
        FP_ratio (float): Ratio of further pairs to nearest neighbor pairs (default: 2.0, range: 1.0-5.0)
        plot (bool): Whether to display the plot interactively (default: False)
        save (bool): Whether to save the plot to file (default: False)
        output_dir (str): Directory to save the plot if save=True (default: None)

    Returns:
        pd.DataFrame: DataFrame with 'x' and 'y' columns containing 2D PaCMAP coordinates
    """
    print(f"{datetime.now().time()} PaCMAP with n_neighbors={n_neighbors}, MN_ratio={MN_ratio}, FP_ratio={FP_ratio}...")
    pac = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors,
                        MN_ratio=MN_ratio, FP_ratio=FP_ratio)
    df_result = pd.DataFrame(pac.fit_transform(df), columns=["x", "y"])

    if plot or save:
        plt.figure(figsize=(10, 8))
        plt.scatter(df_result["x"], df_result["y"],
                    color="black", alpha=0.4, s=20)
        plt.title(
            f"PaCMAP Visualization (n_neighbors={n_neighbors}, MN_ratio={MN_ratio}, FP_ratio={FP_ratio})")
        plt.xlabel("PaCMAP Component 1")
        plt.ylabel("PaCMAP Component 2")
        plt.grid(True, alpha=0.3)
        if save and output_dir:
            filepath = os.path.join(
                output_dir, f"pacmap_nn{n_neighbors}_mnr{MN_ratio}_fpr{FP_ratio}.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"{datetime.now().time()} saved plot to {filepath}")
        if plot:
            plt.show()
        plt.close()
    return df_result


def plot_pacmap_parameters_grid(df, MN_ratio_values, FP_ratio_values, n_neighbors=10, output_dir=None):
    """
    Grid search for PaCMAP parameters: MN_ratio and FP_ratio with fixed n_neighbors.

    Creates a grid of subplots showing PaCMAP projections for different parameter combinations.
    MN_ratio controls the balance between local (neighbors) and global (mid-near) structure preservation.
    FP_ratio controls further pairs for both local and global structure.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        MN_ratio_values (list): List of MN_ratio values to test (e.g., [0.3, 0.5, 0.7])
        FP_ratio_values (list): List of FP_ratio values to test (e.g., [1.0, 2.0, 3.0])
        n_neighbors (int): Fixed n_neighbors value (default: 10)
        output_dir (str): Directory to save the grid plot if provided (default: None)
    """
    total_combinations = len(MN_ratio_values) * len(FP_ratio_values)
    print(f"{datetime.now().time()} testing {total_combinations} PaCMAP parameter combinations with n_neighbors={n_neighbors}...")

    fig, ax_flat = plt.subplots(len(FP_ratio_values), len(MN_ratio_values),
                                figsize=(4*len(MN_ratio_values), 4*len(FP_ratio_values)))
    if total_combinations == 1:
        ax_flat = [[ax_flat]]
    elif len(FP_ratio_values) == 1:
        ax_flat = [ax_flat]

    plot_idx = 0
    for idx_fp, FP_ratio in enumerate(FP_ratio_values):
        for idx_mn, MN_ratio in enumerate(MN_ratio_values):
            print(f"{datetime.now().time()} PaCMAP {plot_idx+1}/{total_combinations}: n_neighbors={n_neighbors}, MN_ratio={MN_ratio}, FP_ratio={FP_ratio}...")

            try:
                pac = pacmap.PaCMAP(
                    n_components=2, n_neighbors=n_neighbors, MN_ratio=MN_ratio, FP_ratio=FP_ratio)
                df_pac = pd.DataFrame(pac.fit_transform(df), columns=[0, 1])

                if len(FP_ratio_values) > 1:
                    ax = ax_flat[idx_fp][idx_mn]
                else:
                    ax = ax_flat[0][idx_mn]
                ax.set_title(f"MN={MN_ratio}, FP={FP_ratio}")
                ax.scatter(df_pac[0], df_pac[1],
                           color="black", alpha=0.33, s=3)
                ax.grid(True, alpha=0.3)
            except Exception as e:
                print(
                    f"{datetime.now().time()} Error with MN_ratio={MN_ratio}, FP_ratio={FP_ratio}: {e}")
                if len(FP_ratio_values) > 1:
                    ax = ax_flat[idx_fp][idx_mn]
                else:
                    ax = ax_flat[0][idx_mn]
                ax.text(
                    0.5, 0.5, f"Error:\n{str(e)[:30]}", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])

            plot_idx += 1

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "pacmap_parameters_grid.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved PaCMAP parameters grid to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_pacmap_n_neighbors_grid(df, n_neighbors_values, MN_ratio_values, FP_ratio_values, output_dir=None):
    """
    Grid search for PaCMAP parameters: vary n_neighbors with MN_ratio and FP_ratio grids.

    Creates separate plots for each n_neighbors value, with each plot showing a grid of 
    MN_ratio vs FP_ratio combinations.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        n_neighbors_values (list): List of n_neighbors values to test (e.g., [5, 10, 15, 20])
        MN_ratio_values (list): List of MN_ratio values to test (e.g., [0.3, 0.5, 0.7])
        FP_ratio_values (list): List of FP_ratio values to test (e.g., [1.0, 2.0, 3.0])
        output_dir (str): Directory to save the plots (default: None)
    """
    total_combinations = len(n_neighbors_values) * \
        len(MN_ratio_values) * len(FP_ratio_values)
    print(f"{datetime.now().time()} testing {total_combinations} PaCMAP parameter combinations across {len(n_neighbors_values)} n_neighbors values...")

    for n_neighbors in n_neighbors_values:
        print(
            f"{datetime.now().time()} creating PaCMAP grid for n_neighbors={n_neighbors}...")

        fig, ax_flat = plt.subplots(len(FP_ratio_values), len(MN_ratio_values),
                                    figsize=(4*len(MN_ratio_values), 4*len(FP_ratio_values)))

        total_grid_combinations = len(MN_ratio_values) * len(FP_ratio_values)
        if total_grid_combinations == 1:
            ax_flat = [[ax_flat]]
        elif len(FP_ratio_values) == 1:
            ax_flat = [ax_flat]

        plot_idx = 0
        for idx_fp, FP_ratio in enumerate(FP_ratio_values):
            for idx_mn, MN_ratio in enumerate(MN_ratio_values):
                print(f"{datetime.now().time()} PaCMAP n_neighbors={n_neighbors} {plot_idx+1}/{total_grid_combinations}: MN_ratio={MN_ratio}, FP_ratio={FP_ratio}...")

                try:
                    pac = pacmap.PaCMAP(
                        n_components=2, n_neighbors=n_neighbors, MN_ratio=MN_ratio, FP_ratio=FP_ratio)
                    df_pac = pd.DataFrame(
                        pac.fit_transform(df), columns=[0, 1])

                    if len(FP_ratio_values) > 1:
                        ax = ax_flat[idx_fp][idx_mn]
                    else:
                        ax = ax_flat[0][idx_mn]
                    ax.set_title(f"MN={MN_ratio}, FP={FP_ratio}")
                    ax.scatter(df_pac[0], df_pac[1],
                               color="black", alpha=0.33, s=3)
                    ax.grid(True, alpha=0.3)
                except Exception as e:
                    print(
                        f"{datetime.now().time()} Error with n_neighbors={n_neighbors}, MN_ratio={MN_ratio}, FP_ratio={FP_ratio}: {e}")
                    if len(FP_ratio_values) > 1:
                        ax = ax_flat[idx_fp][idx_mn]
                    else:
                        ax = ax_flat[0][idx_mn]
                    ax.text(
                        0.5, 0.5, f"Error:\n{str(e)[:30]}", ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])

                plot_idx += 1

        plt.suptitle(
            f"PaCMAP Parameters Grid (n_neighbors={n_neighbors})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        if output_dir:
            filepath = os.path.join(
                output_dir, f"pacmap_nn{n_neighbors}_parameters_grid.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(
                f"{datetime.now().time()} saved PaCMAP n_neighbors={n_neighbors} grid to {filepath}")
            plt.close()
        else:
            plt.show()
