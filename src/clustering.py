"""DBSCAN and K-Means clustering and related functions"""

from .utils import ColorGenerator
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, KMeans
from datetime import datetime
import matplotlib.pyplot as plt
import os
import math
import itertools as itools
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid tkinter issues


def apply_clustering(df, epsilon, min_samples, plot=False, save=False, output_dir=None):
    """
    Apply DBSCAN clustering to 2D data and calculate clustering metrics.

    Args:
        df (pd.DataFrame): Input DataFrame with 'x' and 'y' columns for 2D points
        epsilon (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other
        min_samples (int): The number of samples in a neighborhood for a point to be considered a core point
        plot (bool): Whether to display the plot interactively (default: False)
        save (bool): Whether to save the plot to file (default: False)
        output_dir (str): Directory to save the plot if save=True (default: None)

    Returns:
        pd.DataFrame: DataFrame with 'cluster_id' column added containing cluster assignments
    """
    df_result = df.copy()
    cluster = DBSCAN(eps=epsilon, min_samples=min_samples).fit(df_result)
    cluster_id = [str(l) for l in cluster.labels_]
    df_result.insert(0, "cluster_id", cluster_id)

    # Calculate clustering metrics
    n_clusters = len(set(cluster.labels_)) - \
        (1 if -1 in cluster.labels_ else 0)
    n_noise = list(cluster.labels_).count(-1)
    if n_clusters > 1 and n_noise < len(cluster.labels_):
        mask = cluster.labels_ != -1
        if mask.sum() > 0:
            silhouette = silhouette_score(
                df_result[mask][["x", "y"]], cluster.labels_[mask])
            print(f"{datetime.now().time()} DBSCAN eps={epsilon}, min_samples={min_samples}: "
                  f"{n_clusters} clusters, {n_noise} noise points, silhouette={silhouette:.3f}")

    if plot or save:
        plt.figure(figsize=(10, 8))
        plt.scatter(df_result["x"], df_result["y"],
                    color="black", alpha=.1, s=20)
        colors = ColorGenerator()
        list_cluster_id = sorted([int(x)
                                 for x in set(cluster_id) if x != '-1'])

        for cid in list_cluster_id:
            df_partial = df_result[df_result["cluster_id"] == str(cid)]
            plt.scatter(df_partial["x"], df_partial["y"],
                        alpha=.5, c=colors.next(), label=f"Cluster {cid}", s=30)

        # Plot noise points
        df_noise = df_result[df_result["cluster_id"] == '-1']
        if len(df_noise) > 0:
            plt.scatter(df_noise["x"], df_noise["y"], color="red",
                        alpha=0.3, marker='x', label="Noise", s=50)

        plt.title(
            f"DBSCAN Clustering (eps={epsilon}, min_samples={min_samples})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('off')

        if save and output_dir:
            filepath = os.path.join(
                output_dir, f"dbscan_eps_{epsilon}_samples_{min_samples}.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"{datetime.now().time()} saved plot to {filepath}")
        if plot:
            plt.show()
        plt.close()

    return df_result


def get_adaptive_min_samples(n_samples, method="balanced"):
    """
    Calculate adaptive min_samples for DBSCAN based on dataset size.

    Different strategies for finding clusters:
    - 'permissive': Very lenient, finds many clusters
    - 'aggressive': Lenient, good for clear patterns
    - 'balanced': Moderate, balanced sensitivity
    - 'conservative': Strict, only obvious clusters

    Args:
        n_samples: number of samples in dataset
        method: 'permissive', 'aggressive', 'balanced' (default), or 'conservative'

    Returns:
        Recommended min_samples value
    """
    log_n = math.log(n_samples) if n_samples > 1 else 1

    if method == "permissive":
        # Very lenient - only 2-3 minimum
        return max(2, int(2 + log_n * 0.5))
    elif method == "aggressive":
        # Lenient - encourages cluster formation
        return max(3, int(5 + log_n))
    elif method == "conservative":
        # Strict - only strong clusters
        return max(5, int(10 + log_n * 1.5))
    else:  # balanced (default)
        # Middle ground
        if n_samples < 1000:
            return max(3, int(4 + log_n))
        elif n_samples < 10000:
            return max(5, int(8 + log_n * 1.5))
        else:
            return max(10, int(10 + log_n * 2))


def plot_cluster_parameters(df, list_sample, list_epsilon, output_dir=None):
    """
    Create a grid of subplots showing DBSCAN clustering results with different parameter combinations.

    Args:
        df (pd.DataFrame): Input DataFrame with 'x' and 'y' columns for 2D points
        list_sample (list): List of min_samples values to test
        list_epsilon (list): List of epsilon values to test
        output_dir (str): Directory to save the grid plot if provided (default: None)

    Returns:
        None: Displays or saves the grid of clustering visualizations
    """
    n_sample = len(list_sample)
    n_epsilon = len(list_epsilon)
    n_plot = n_sample * n_epsilon
    _, ax = plt.subplots(n_sample, n_epsilon, figsize=[14, 12])

    for idx, (spl, eps) in enumerate(itools.product(list_sample, list_epsilon)):
        print(
            f"{datetime.now().time()} subplot {idx+1}/{n_plot} epsilon={eps} samples={spl} ...")

        df_cluster = df.copy()
        cluster = DBSCAN(eps=eps, min_samples=spl).fit(df_cluster)
        df_cluster.insert(0, "cluster_id", cluster.labels_)

        sub_plt = ax[math.floor(idx / n_epsilon)][idx % n_epsilon]
        sub_plt.set_title(f"epsilon={eps} min_samples={spl}")
        sub_plt.scatter(df_cluster["x"], df_cluster["y"],
                        color="black", alpha=.1, s=10)

        colors = ColorGenerator()
        list_cluster_id = list(set(cluster.labels_))
        if -1 in list_cluster_id:
            list_cluster_id.remove(-1)

        for cluster_id in list_cluster_id:
            df_sub_cluster = df_cluster[df_cluster["cluster_id"] == cluster_id]
            sub_plt.scatter(df_sub_cluster["x"], df_sub_cluster["y"],
                            alpha=.33, c=colors.next(), s=15)
        sub_plt.grid(True, alpha=0.3)
        sub_plt.axis('off')

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "dbscan_parameters_grid.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved DBSCAN grid to {filepath}")
        plt.close()
    else:
        plt.show()


def apply_kmeans_clustering(df, n_clusters, plot=False, save=False, output_dir=None):
    """
    Apply K-Means clustering to 2D data and calculate clustering metrics.

    Args:
        df (pd.DataFrame): Input DataFrame with 'x' and 'y' columns for 2D points
        n_clusters (int): Number of clusters to find
        plot (bool): Whether to display the plot interactively (default: False)
        save (bool): Whether to save the plot to file (default: False)
        output_dir (str): Directory to save the plot if save=True (default: None)

    Returns:
        pd.DataFrame: DataFrame with 'cluster_id' column added containing cluster assignments
    """
    df_result = df.copy()
    kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(
        df_result[["x", "y"]])
    cluster_id = [str(l) for l in kmeans.labels_]
    df_result.insert(0, "cluster_id", cluster_id)

    # Calculate clustering metrics
    silhouette = silhouette_score(df_result[["x", "y"]], kmeans.labels_)
    inertia = kmeans.inertia_
    print(f"{datetime.now().time()} K-Means k={n_clusters}: "
          f"silhouette={silhouette:.3f}, inertia={inertia:.2f}")

    if plot or save:
        plt.figure(figsize=(10, 8))
        plt.scatter(df_result["x"], df_result["y"],
                    color="black", alpha=.1, s=20)
        colors = ColorGenerator()

        for cid in sorted(set(cluster_id)):
            df_partial = df_result[df_result["cluster_id"] == cid]
            plt.scatter(df_partial["x"], df_partial["y"],
                        alpha=.5, c=colors.next(), label=f"Cluster {cid}", s=30)

        plt.title(f"K-Means Clustering (k={n_clusters})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('off')

        if save and output_dir:
            filepath = os.path.join(output_dir, f"kmeans_k_{n_clusters}.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"{datetime.now().time()} saved plot to {filepath}")
        if plot:
            plt.show()
        plt.close()

    return df_result


def get_adaptive_k_clusters(method="balanced"):
    """
    Calculate number of clusters for K-Means in the range 20-80.

    Different strategies for choosing k:
    - 'permissive': 80 clusters (max)
    - 'aggressive': 60 clusters
    - 'balanced': 40 clusters (default)
    - 'conservative': 20 clusters (min)

    Args:
        n_samples: number of samples in dataset (unused, kept for compatibility)
        method: 'permissive', 'aggressive', 'balanced' (default), or 'conservative'

    Returns:
        Recommended k value: 20, 40, 60, or 80
    """
    if method == "permissive":
        return 80
    elif method == "aggressive":
        return 60
    elif method == "conservative":
        return 20
    else:  # balanced (default)
        return 40


def plot_kmeans_parameters(df, list_k, output_dir=None):
    """
    Create a grid of subplots showing K-Means clustering results with different k values.

    Args:
        df (pd.DataFrame): Input DataFrame with 'x' and 'y' columns for 2D points
        list_k (list): List of k (number of clusters) values to test
        output_dir (str): Directory to save the grid plot if provided (default: None)

    Returns:
        None: Displays or saves the grid of clustering visualizations
    """
    n_k = len(list_k)
    n_cols = min(4, n_k)
    n_rows = math.ceil(n_k / n_cols)
    _, ax = plt.subplots(n_rows, n_cols, figsize=[16, 4*n_rows])

    # Flatten ax array if it's not already flat
    if n_rows == 1 and n_cols == 1:
        ax = [[ax]]
    elif n_rows == 1 or n_cols == 1:
        ax = [[a] for a in ax] if n_rows > 1 else [ax]

    for idx, k in enumerate(list_k):
        print(f"{datetime.now().time()} subplot {idx+1}/{n_k} k={k} ...")

        df_cluster = df.copy()
        kmeans = KMeans(n_clusters=k, random_state=42,
                        n_init=10).fit(df_cluster[["x", "y"]])
        df_cluster.insert(0, "cluster_id", kmeans.labels_)

        row = idx // n_cols
        col = idx % n_cols
        sub_plt = ax[row][col]
        sub_plt.set_title(f"k={k} (inertia={kmeans.inertia_:.0f})")
        sub_plt.scatter(df_cluster["x"], df_cluster["y"],
                        color="black", alpha=.1, s=10)

        colors = ColorGenerator()
        for cluster_id in sorted(set(kmeans.labels_)):
            df_sub_cluster = df_cluster[df_cluster["cluster_id"] == cluster_id]
            sub_plt.scatter(df_sub_cluster["x"], df_sub_cluster["y"],
                            alpha=.33, c=colors.next(), s=15)
        sub_plt.grid(True, alpha=0.3)
        sub_plt.axis('off')

    # Hide unused subplots
    for idx in range(n_k, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax[row][col].set_visible(False)

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "kmeans_parameters_grid.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved K-Means grid to {filepath}")
        plt.close()
    else:
        plt.show()
