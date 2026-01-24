"""DBSCAN, K-Means, and Hierarchical clustering and related functions"""

from .utils import ColorGenerator
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
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


def apply_hierarchical_clustering(df, n_clusters, linkage_method='ward', plot=False, save=False, output_dir=None):
    """
    Apply Hierarchical (Agglomerative) clustering to 2D data and calculate clustering metrics.

    Args:
        df (pd.DataFrame): Input DataFrame with 'x' and 'y' columns for 2D points
        n_clusters (int): Number of clusters to find
        linkage_method (str): Linkage criterion ('ward', 'complete', 'average', 'single')
        plot (bool): Whether to display the plot interactively (default: False)
        save (bool): Whether to save the plot to file (default: False)
        output_dir (str): Directory to save the plot if save=True (default: None)

    Returns:
        pd.DataFrame: DataFrame with 'cluster_id' column added containing cluster assignments
    """
    print(f"{datetime.now().time()} Hierarchical Clustering with n_clusters={n_clusters}, linkage={linkage_method}...")

    df_result = df.copy()
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method
    ).fit(df_result[["x", "y"]])
    cluster_id = [str(l) for l in hierarchical.labels_]
    df_result.insert(0, "cluster_id", cluster_id)

    # Calculate clustering metrics
    silhouette = silhouette_score(df_result[["x", "y"]], hierarchical.labels_)
    print(f"{datetime.now().time()} Hierarchical n_clusters={n_clusters}, linkage={linkage_method}: "
          f"silhouette={silhouette:.3f}")

    if plot or save:
        plt.figure(figsize=(10, 8))
        plt.scatter(df_result["x"], df_result["y"],
                    color="black", alpha=.1, s=20)
        colors = ColorGenerator()

        for cid in sorted(set(cluster_id)):
            df_partial = df_result[df_result["cluster_id"] == cid]
            plt.scatter(df_partial["x"], df_partial["y"],
                        alpha=.5, c=colors.next(), label=f"Cluster {cid}", s=30)

        plt.title(
            f"Hierarchical Clustering (n={n_clusters}, linkage={linkage_method})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('off')

        if save and output_dir:
            filepath = os.path.join(
                output_dir, f"hierarchical_n_{n_clusters}_{linkage_method}.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"{datetime.now().time()} saved plot to {filepath}")
        if plot:
            plt.show()
        plt.close()

    return df_result


def plot_hierarchical_parameters(df, list_n_clusters, list_linkage, output_dir=None):
    """
    Create a grid of subplots showing Hierarchical clustering results with different parameter combinations.

    Args:
        df (pd.DataFrame): Input DataFrame with 'x' and 'y' columns for 2D points
        list_n_clusters (list): List of n_clusters values to test
        list_linkage (list): List of linkage methods to test ('ward', 'complete', 'average', 'single')
        output_dir (str): Directory to save the grid plot if provided (default: None)

    Returns:
        None: Displays or saves the grid of clustering visualizations
    """
    n_clusters_count = len(list_n_clusters)
    n_linkage = len(list_linkage)
    n_plot = n_clusters_count * n_linkage
    _, ax = plt.subplots(n_clusters_count, n_linkage,
                         figsize=[16, 4*n_clusters_count])

    # Handle single row/col case
    if n_clusters_count == 1 and n_linkage == 1:
        ax = [[ax]]
    elif n_clusters_count == 1:
        ax = [ax]
    elif n_linkage == 1:
        ax = [[a] for a in ax]

    for idx, (n_clust, link) in enumerate(itools.product(list_n_clusters, list_linkage)):
        print(
            f"{datetime.now().time()} subplot {idx+1}/{n_plot} n_clusters={n_clust} linkage={link} ...")

        df_cluster = df.copy()
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clust,
            linkage=link
        ).fit(df_cluster[["x", "y"]])
        df_cluster.insert(0, "cluster_id", hierarchical.labels_)

        row = idx // n_linkage
        col = idx % n_linkage
        sub_plt = ax[row][col]

        # Calculate silhouette for title
        silhouette = silhouette_score(
            df_cluster[["x", "y"]], hierarchical.labels_)
        sub_plt.set_title(f"n={n_clust}, {link} (sil={silhouette:.2f})")
        sub_plt.scatter(df_cluster["x"], df_cluster["y"],
                        color="black", alpha=.1, s=10)

        colors = ColorGenerator()
        for cluster_id in sorted(set(hierarchical.labels_)):
            df_sub_cluster = df_cluster[df_cluster["cluster_id"] == cluster_id]
            sub_plt.scatter(df_sub_cluster["x"], df_sub_cluster["y"],
                            alpha=.33, c=colors.next(), s=15)
        sub_plt.grid(True, alpha=0.3)
        sub_plt.axis('off')

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "hierarchical_parameters_grid.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved Hierarchical grid to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_dendrogram(df, linkage_method='ward', truncate_level=5, output_dir=None):
    """
    Create a dendrogram visualization showing hierarchical cluster relationships.

    This is the key visualization for interpretability - shows how clusters merge
    at different distance thresholds.

    Args:
        df (pd.DataFrame): Input DataFrame with 'x' and 'y' columns for 2D points
        linkage_method (str): Linkage criterion ('ward', 'complete', 'average', 'single')
        truncate_level (int): Level at which to truncate the dendrogram (default: 5)
        output_dir (str): Directory to save the plot if provided (default: None)

    Returns:
        None: Displays or saves the dendrogram visualization
    """
    print(f"{datetime.now().time()} Creating dendrogram with linkage={linkage_method}...")

    # Compute linkage matrix
    Z = linkage(df[["x", "y"]].values, method=linkage_method)

    fig, ax = plt.subplots(1, 2, figsize=[16, 6])

    # Full dendrogram (truncated for readability)
    ax[0].set_title(f"Dendrogram (truncated, linkage={linkage_method})")
    dendrogram(Z, ax=ax[0], truncate_mode='level', p=truncate_level,
               leaf_rotation=90, leaf_font_size=8, show_leaf_counts=True)
    ax[0].set_xlabel("Sample index or (cluster size)")
    ax[0].set_ylabel("Distance")

    # Distance distribution - helps choose number of clusters
    ax[1].set_title("Merge distances (for choosing n_clusters)")
    distances = Z[:, 2]
    ax[1].plot(range(1, len(distances) + 1),
               distances[::-1], 'b-', linewidth=1)
    ax[1].scatter(range(1, min(21, len(distances) + 1)), distances[-20:][::-1],
                  c='red', s=50, zorder=5)
    ax[1].set_xlabel("Number of clusters")
    ax[1].set_ylabel("Distance")
    ax[1].set_xlim(0, 100)
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, f"dendrogram_{linkage_method}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved dendrogram to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_annotated_dendrogram(df_2d, df_synergies, n_clusters=100, linkage_method='ward', output_dir=None):
    """
    Create a dendrogram with branch annotations showing top synergies for each cluster.

    This shows what characterizes each branch by labeling clusters with their
    dominant synergies.

    Args:
        df_2d (pd.DataFrame): DataFrame with 'x' and 'y' columns (reduced coordinates)
        df_synergies (pd.DataFrame): Original synergy features DataFrame
        n_clusters (int): Number of leaf clusters to show (default: 100)
        linkage_method (str): Linkage criterion ('ward', 'complete', 'average', 'single')
        output_dir (str): Directory to save the plot if provided (default: None)

    Returns:
        dict: Cluster profiles with top synergies for each cluster
    """
    import numpy as np
    from scipy.cluster.hierarchy import fcluster

    print(f"{datetime.now().time()} Creating annotated dendrogram with {n_clusters} clusters...")

    # Compute linkage matrix on 2D coordinates
    Z = linkage(df_2d[["x", "y"]].values, method=linkage_method)

    # Get cluster assignments for n_clusters
    cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')

    # Calculate global means for comparison
    global_means = df_synergies.mean()

    # Build cluster profiles
    cluster_profiles = {}
    for cluster_id in range(1, n_clusters + 1):
        mask = cluster_labels == cluster_id
        cluster_size = mask.sum()

        if cluster_size > 0:
            # Get mean synergies for this cluster
            cluster_means = df_synergies[mask].mean()

            # Calculate how much each synergy is above global average
            diff_from_global = cluster_means - global_means

            # Get top 3 synergies (most above average)
            top_synergies = diff_from_global.nlargest(3)

            cluster_profiles[cluster_id] = {
                'size': cluster_size,
                'top_synergies': [(syn, f"+{val:.1f}") for syn, val in top_synergies.items() if val > 0],
                'mean_synergies': cluster_means.to_dict()
            }

    # Create figure with dendrogram
    fig, ax = plt.subplots(1, 1, figsize=[20, 12])

    # Plot dendrogram truncated to n_clusters leaves
    dendro = dendrogram(
        Z,
        ax=ax,
        truncate_mode='lastp',
        p=n_clusters,
        leaf_rotation=90,
        leaf_font_size=6,
        show_leaf_counts=True,
        show_contracted=True
    )

    ax.set_title(
        f"Annotated Dendrogram ({n_clusters} clusters, linkage={linkage_method})")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Distance")

    # Add legend with cluster profile summary
    # Show top 10 most distinctive clusters in a text box
    sorted_clusters = sorted(
        cluster_profiles.items(),
        key=lambda x: x[1]['size'],
        reverse=True
    )[:15]

    legend_text = "Top 15 largest clusters:\n"
    for cid, profile in sorted_clusters:
        top_syn = profile['top_synergies'][:2]
        if top_syn:
            syn_str = ", ".join([f"{s[0]}" for s in top_syn])
            legend_text += f"C{cid} (n={profile['size']}): {syn_str}\n"

    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(
            output_dir, f"dendrogram_annotated_{linkage_method}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved annotated dendrogram to {filepath}")
        plt.close()

        # Also export cluster profiles to JSON
        import json
        profiles_filepath = os.path.join(
            output_dir, f"cluster_profiles_{linkage_method}.json")
        # Convert for JSON serialization
        export_profiles = {}
        for cid, profile in cluster_profiles.items():
            export_profiles[str(cid)] = {
                'size': int(profile['size']),
                'top_synergies': profile['top_synergies']
            }
        with open(profiles_filepath, 'w') as f:
            json.dump(export_profiles, f, indent=2)
        print(f"{datetime.now().time()} saved cluster profiles to {profiles_filepath}")
    else:
        plt.show()

    return cluster_profiles


def get_cluster_synergy_summary(df_synergies, cluster_labels, top_n=3):
    """
    Generate a summary of top synergies for each cluster.

    Args:
        df_synergies (pd.DataFrame): Original synergy features DataFrame
        cluster_labels (array-like): Cluster assignment for each row
        top_n (int): Number of top synergies to return per cluster

    Returns:
        dict: {cluster_id: [(synergy_name, avg_value, diff_from_global), ...]}
    """
    global_means = df_synergies.mean()
    summaries = {}

    for cluster_id in set(cluster_labels):
        if cluster_id == -1:  # Skip noise for DBSCAN
            continue
        mask = cluster_labels == cluster_id if hasattr(
            cluster_labels, '__iter__') else False
        if isinstance(cluster_labels, (list, pd.Series)):
            mask = [l == cluster_id for l in cluster_labels]

        cluster_data = df_synergies[mask] if any(mask) else pd.DataFrame()

        if len(cluster_data) > 0:
            cluster_means = cluster_data.mean()
            diff = cluster_means - global_means
            top = diff.nlargest(top_n)
            summaries[cluster_id] = [
                (syn, cluster_means[syn], diff[syn])
                for syn in top.index
            ]

    return summaries
