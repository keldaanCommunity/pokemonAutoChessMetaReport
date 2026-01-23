"""TriMap dimensionality reduction methods"""

import trimap
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')


def apply_trimap(df, n_inliers=15, n_outliers=10, n_random=5, plot=False, save=False, output_dir=None):
    """
    Apply TriMap (Triplet-based Manifold Approximation Projection) 
    to reduce high-dimensional data to 2D.

    TriMap is a fast and reliable algorithm for 2D/3D visualization of high-dimensional data.
    It uses triplet constraints to balance local and global structure preservation.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features to reduce
        n_inliers (int): Number of inliers (local neighbors) per point (default: 15, range: 5-30)
            Controls local structure preservation - more inliers = more detail
        n_outliers (int): Number of outliers (global neighbors) per point (default: 10, range: 5-20)
            Controls global structure preservation - more outliers = more global context
        n_random (int): Number of random neighbors per point (default: 5, range: 2-10)
            Controls random repulsion for escaping local minima
        plot (bool): Whether to display the plot interactively (default: False)
        save (bool): Whether to save the plot to file (default: False)
        output_dir (str): Directory to save the plot if save=True (default: None)

    Returns:
        pd.DataFrame: DataFrame with 'x' and 'y' columns containing 2D TriMap coordinates
    """
    print(f"{datetime.now().time()} TriMap with n_inliers={n_inliers}, n_outliers={n_outliers}, n_random={n_random}...")
    tri = trimap.TRIMAP(n_inliers=n_inliers, n_outliers=n_outliers,
                        n_random=n_random, verbose=False)
    df_result = pd.DataFrame(tri.fit_transform(df.values), columns=["x", "y"])

    if plot or save:
        plt.figure(figsize=(10, 8))
        plt.scatter(df_result["x"], df_result["y"],
                    color="black", alpha=0.4, s=20)
        plt.title(
            f"TriMap Visualization (n_inliers={n_inliers}, n_outliers={n_outliers}, n_random={n_random})")
        plt.xlabel("TriMap Component 1")
        plt.ylabel("TriMap Component 2")
        plt.grid(True, alpha=0.3)
        if save and output_dir:
            filepath = os.path.join(
                output_dir, f"trimap_ni{n_inliers}_no{n_outliers}_nr{n_random}.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"{datetime.now().time()} saved plot to {filepath}")
        if plot:
            plt.show()
        plt.close()
    return df_result


def plot_trimap_parameters_grid(df, n_inliers_values, n_outliers_values, n_random=5, output_dir=None):
    """
    Grid search for TriMap parameters: n_inliers and n_outliers with fixed n_random.

    Creates a grid of subplots showing TriMap projections for different parameter combinations.
    n_inliers controls the number of local neighbors (local structure preservation).
    n_outliers controls the number of global neighbors (global structure preservation).

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        n_inliers_values (list): List of n_inliers values to test (e.g., [10, 15, 20])
        n_outliers_values (list): List of n_outliers values to test (e.g., [5, 10, 15])
        n_random (int): Fixed n_random value (default: 5)
        output_dir (str): Directory to save the grid plot if provided (default: None)
    """
    total_combinations = len(n_inliers_values) * len(n_outliers_values)
    print(f"{datetime.now().time()} testing {total_combinations} TriMap parameter combinations with n_random={n_random}...")

    fig, ax_flat = plt.subplots(len(n_outliers_values), len(n_inliers_values),
                                figsize=(4*len(n_inliers_values), 4*len(n_outliers_values)))
    if total_combinations == 1:
        ax_flat = [[ax_flat]]
    elif len(n_outliers_values) == 1:
        ax_flat = [ax_flat]

    plot_idx = 0
    for idx_out, n_outliers in enumerate(n_outliers_values):
        for idx_in, n_inliers in enumerate(n_inliers_values):
            print(f"{datetime.now().time()} TriMap {plot_idx+1}/{total_combinations}: n_inliers={n_inliers}, n_outliers={n_outliers}, n_random={n_random}...")

            try:
                tri = trimap.TRIMAP(n_inliers=n_inliers, n_outliers=n_outliers,
                                    n_random=n_random, verbose=False)
                df_tri = pd.DataFrame(
                    tri.fit_transform(df.values), columns=[0, 1])

                if len(n_outliers_values) > 1:
                    ax = ax_flat[idx_out][idx_in]
                else:
                    ax = ax_flat[0][idx_in]
                ax.set_title(f"IN={n_inliers}, OUT={n_outliers}")
                ax.scatter(df_tri[0], df_tri[1],
                           color="black", alpha=0.33, s=3)
                ax.grid(True, alpha=0.3)
            except Exception as e:
                print(
                    f"{datetime.now().time()} Error with n_inliers={n_inliers}, n_outliers={n_outliers}: {e}")
                if len(n_outliers_values) > 1:
                    ax = ax_flat[idx_out][idx_in]
                else:
                    ax = ax_flat[0][idx_in]
                ax.text(
                    0.5, 0.5, f"Error:\n{str(e)[:30]}", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])

            plot_idx += 1

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "trimap_parameters_grid.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved TriMap parameters grid to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_trimap_n_random_grid(df, n_random_values, n_inliers_values, n_outliers_values, output_dir=None):
    """
    Grid search for TriMap parameters: vary n_random with n_inliers and n_outliers grids.

    Creates separate plots for each n_random value, with each plot showing a grid of 
    n_inliers vs n_outliers combinations.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        n_random_values (list): List of n_random values to test (e.g., [2, 5, 10])
        n_inliers_values (list): List of n_inliers values to test (e.g., [10, 15, 20])
        n_outliers_values (list): List of n_outliers values to test (e.g., [5, 10, 15])
        output_dir (str): Directory to save the plots (default: None)
    """
    total_combinations = len(n_random_values) * \
        len(n_inliers_values) * len(n_outliers_values)
    print(f"{datetime.now().time()} testing {total_combinations} TriMap parameter combinations across {len(n_random_values)} n_random values...")

    for n_random in n_random_values:
        print(
            f"{datetime.now().time()} creating TriMap grid for n_random={n_random}...")

        fig, ax_flat = plt.subplots(len(n_outliers_values), len(n_inliers_values),
                                    figsize=(4*len(n_inliers_values), 4*len(n_outliers_values)))

        total_grid_combinations = len(
            n_inliers_values) * len(n_outliers_values)
        if total_grid_combinations == 1:
            ax_flat = [[ax_flat]]
        elif len(n_outliers_values) == 1:
            ax_flat = [ax_flat]

        plot_idx = 0
        for idx_out, n_outliers in enumerate(n_outliers_values):
            for idx_in, n_inliers in enumerate(n_inliers_values):
                print(f"{datetime.now().time()} TriMap n_random={n_random} {plot_idx+1}/{total_grid_combinations}: n_inliers={n_inliers}, n_outliers={n_outliers}...")

                try:
                    tri = trimap.TRIMAP(n_inliers=n_inliers, n_outliers=n_outliers,
                                        n_random=n_random, verbose=False)
                    df_tri = pd.DataFrame(
                        tri.fit_transform(df.values), columns=[0, 1])

                    if len(n_outliers_values) > 1:
                        ax = ax_flat[idx_out][idx_in]
                    else:
                        ax = ax_flat[0][idx_in]
                    ax.set_title(f"IN={n_inliers}, OUT={n_outliers}")
                    ax.scatter(df_tri[0], df_tri[1],
                               color="black", alpha=0.33, s=3)
                    ax.grid(True, alpha=0.3)
                except Exception as e:
                    print(
                        f"{datetime.now().time()} Error with n_random={n_random}, n_inliers={n_inliers}, n_outliers={n_outliers}: {e}")
                    if len(n_outliers_values) > 1:
                        ax = ax_flat[idx_out][idx_in]
                    else:
                        ax = ax_flat[0][idx_in]
                    ax.text(
                        0.5, 0.5, f"Error:\n{str(e)[:30]}", ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])

                plot_idx += 1

        plt.suptitle(
            f"TriMap Parameters Grid (n_random={n_random})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        if output_dir:
            filepath = os.path.join(
                output_dir, f"trimap_nr{n_random}_parameters_grid.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(
                f"{datetime.now().time()} saved TriMap n_random={n_random} grid to {filepath}")
            plt.close()
        else:
            plt.show()
