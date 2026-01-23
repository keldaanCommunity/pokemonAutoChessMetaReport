"""t-SNE dimensionality reduction methods"""

from sklearn.manifold import TSNE
from datetime import datetime
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid tkinter issues


def apply_tsne(df, perplexity, n_iter=None, plot=False, save=False, output_dir=None):
    """
    Apply t-SNE dimensionality reduction to reduce high-dimensional data to 2D with adaptive iteration tuning.

    Uses adaptive iteration count based on dataset size to balance quality and speed:
    - Small datasets (<1000): 4000 iterations for fine-tuning
    - Medium (1000-50000): 1500 iterations
    - Large (>50000): 1000 iterations

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features to reduce
        perplexity (float): Perplexity parameter for t-SNE; typically between 5 and 50
        n_iter (int): Number of iterations for optimization; None for adaptive (default: None)
        plot (bool): Whether to display the plot interactively (default: False)
        save (bool): Whether to save the plot to file (default: False)
        output_dir (str): Directory to save the plot if save=True (default: None)

    Returns:
        pd.DataFrame: DataFrame with 'x' and 'y' columns containing 2D t-SNE coordinates
    """
    # Adaptive iteration count based on dataset size
    if n_iter is None:
        n_samples = len(df)
        if n_samples < 1000:
            n_iter = 4000
        elif n_samples < 50000:
            n_iter = 1500
        else:
            n_iter = 1000

    print(f"{datetime.now().time()} t-SNE with perplexity={perplexity}, n_iter={n_iter}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter,
                method="barnes_hut", init="pca", learning_rate="auto",
                metric="euclidean")
    df_result = pd.DataFrame(tsne.fit_transform(df), columns=["x", "y"])

    if plot or save:
        plt.figure(figsize=(10, 8))
        plt.scatter(df_result["x"], df_result["y"],
                    color="black", alpha=0.4, s=20)
        plt.title(f"t-SNE Visualization (perplexity={perplexity})")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.grid(True, alpha=0.3)
        if save and output_dir:
            filepath = os.path.join(
                output_dir, f"tsne_perplexity_{perplexity}.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"{datetime.now().time()} saved plot to {filepath}")
        if plot:
            plt.show()
        plt.close()
    return df_result


def get_perplexity_range(n_samples, n_values=8):
    """
    Generate a range of perplexity values for comparison based on dataset size.

    Creates logarithmically-spaced perplexity values between 5 and a maximum determined 
    by the dataset size, ensuring valid range constraints.

    Args:
        n_samples (int): Number of samples in the dataset
        n_values (int): Number of perplexity values to generate (default: 8)

    Returns:
        list: List of integer perplexity values suitable for t-SNE testing
    """
    max_perplexity = min(n_samples // 3, 500)
    min_perplexity = 5

    perplexities = np.logspace(
        np.log10(min_perplexity),
        np.log10(max_perplexity),
        n_values
    )

    return [int(p) for p in perplexities]


def plot_tsne_parameters(df, list_perplexity, output_dir=None):
    """
    Create a grid of subplots showing t-SNE results with different perplexity values.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        list_perplexity (list): List of perplexity values to test
        output_dir (str): Directory to save the grid plot if provided (default: None)

    Returns:
        None: Displays or saves the grid of t-SNE visualizations
    """
    n_perplexity = len(list_perplexity)
    multi_axes = n_perplexity > 3
    n_rows = 2 if multi_axes else 1
    n_cols = math.ceil(n_perplexity / n_rows) if multi_axes else n_perplexity
    fig, ax = plt.subplots(n_rows, n_cols, figsize=[16, 10])

    if multi_axes:
        ax_flat = ax.flatten()
    else:
        ax_flat = [ax] if n_perplexity == 1 else ax

    for idx, ppx in enumerate(list_perplexity):
        print(
            f"{datetime.now().time()} subplot {idx+1}/{n_perplexity} perplexity={ppx} ...")
        tsne = TSNE(n_components=2, perplexity=ppx, method="barnes_hut",
                    init="pca", max_iter=1500, learning_rate="auto", random_state=42)
        df_tsne = pd.DataFrame(tsne.fit_transform(df), columns=[0, 1])

        sub_plt = ax_flat[idx]
        sub_plt.set_title(f"perplexity={ppx}")
        sub_plt.scatter(df_tsne[0], df_tsne[1], color="black", alpha=.33, s=5)
        sub_plt.grid(True, alpha=0.3)

    for idx in range(len(list_perplexity), len(ax_flat)):
        ax_flat[idx].axis('off')

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "tsne_perplexity_comparison.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved t-SNE comparison to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_tsne_init_comparison(df, list_inits, output_dir=None):
    """
    Compare different t-SNE initialization methods side by side.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        list_inits (list): List of initialization methods to compare (e.g., ['pca', 'random'])
        output_dir (str): Directory to save the comparison plot if provided (default: None)

    Returns:
        None: Displays or saves the comparison visualization
    """
    n_inits = len(list_inits)
    fig, ax = plt.subplots(1, n_inits, figsize=[16, 4])
    ax_flat = [ax] if n_inits == 1 else ax

    for idx, init_method in enumerate(list_inits):
        print(f"{datetime.now().time()} t-SNE with init={init_method}...")
        tsne = TSNE(n_components=2, perplexity=50, method="barnes_hut",
                    init=init_method, max_iter=4000, learning_rate="auto", random_state=42)
        df_tsne = pd.DataFrame(tsne.fit_transform(df), columns=[0, 1])

        ax_flat[idx].set_title(f"init={init_method}")
        ax_flat[idx].scatter(df_tsne[0], df_tsne[1],
                             color="black", alpha=.33, s=10)
        ax_flat[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "tsne_init_comparison.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved init comparison to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_tsne_metric_comparison(df, list_metrics, output_dir=None):
    """
    Compare different distance metrics for t-SNE side by side.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        list_metrics (list): List of distance metrics to compare (e.g., ['euclidean', 'cosine', 'manhattan'])
        output_dir (str): Directory to save the comparison plot if provided (default: None)

    Returns:
        None: Displays or saves the comparison visualization
    """
    n_metrics = len(list_metrics)
    fig, ax = plt.subplots(1, n_metrics, figsize=[16, 4])
    ax_flat = [ax] if n_metrics == 1 else ax

    for idx, metric in enumerate(list_metrics):
        print(f"{datetime.now().time()} t-SNE with metric={metric}...")
        try:
            tsne = TSNE(n_components=2, perplexity=50, method="barnes_hut",
                        init="pca", max_iter=4000, learning_rate="auto",
                        metric=metric, random_state=42)
            df_tsne = pd.DataFrame(tsne.fit_transform(df), columns=[0, 1])

            ax_flat[idx].set_title(f"metric={metric}")
            ax_flat[idx].scatter(df_tsne[0], df_tsne[1],
                                 color="black", alpha=.33, s=10)
            ax_flat[idx].grid(True, alpha=0.3)
        except Exception as e:
            print(f"{datetime.now().time()} warning: {metric} failed - {str(e)}")
            ax_flat[idx].text(
                0.5, 0.5, f"Failed: {metric}", ha='center', va='center')

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "tsne_metric_comparison.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved metric comparison to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_tsne_learning_rate_comparison(df, list_learning_rates, adaptive_perplexity, output_dir=None):
    """
    Compare different learning rates for t-SNE side by side.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        list_learning_rates (list): List of learning rates to compare (e.g., ['auto', 200, 500, 800])
        adaptive_perplexity (float): Perplexity value to use for all comparisons
        output_dir (str): Directory to save the comparison plot if provided (default: None)

    Returns:
        None: Displays or saves the comparison visualization
    """
    n_rates = len(list_learning_rates)
    fig, ax = plt.subplots(1, n_rates, figsize=[16, 4])
    ax_flat = [ax] if n_rates == 1 else ax

    for idx, lr in enumerate(list_learning_rates):
        print(f"{datetime.now().time()} t-SNE with learning_rate={lr}...")
        tsne = TSNE(n_components=2, perplexity=adaptive_perplexity, method="barnes_hut",
                    init="pca", max_iter=4000, learning_rate=lr, random_state=42)
        df_tsne = pd.DataFrame(tsne.fit_transform(df), columns=[0, 1])

        ax_flat[idx].set_title(f"learning_rate={lr}")
        ax_flat[idx].scatter(df_tsne[0], df_tsne[1],
                             color="black", alpha=.33, s=10)
        ax_flat[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(
            output_dir, "tsne_learning_rate_comparison.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved learning_rate comparison to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_tsne_early_exaggeration_comparison(df, list_early_exag, adaptive_perplexity, output_dir=None):
    """
    Compare different early exaggeration values for t-SNE side by side.

    Early exaggeration amplifies distances during the first phase of optimization,
    helping t-SNE preserve global structure better.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        list_early_exag (list): List of early exaggeration values to compare (e.g., [8, 12, 16, 20])
        adaptive_perplexity (float): Perplexity value to use for all comparisons
        output_dir (str): Directory to save the comparison plot if provided (default: None)

    Returns:
        None: Displays or saves the comparison visualization
    """
    n_exag = len(list_early_exag)
    fig, ax = plt.subplots(1, n_exag, figsize=[16, 4])
    ax_flat = [ax] if n_exag == 1 else ax

    for idx, early_exag in enumerate(list_early_exag):
        print(f"{datetime.now().time()} t-SNE with early_exaggeration={early_exag}...")
        tsne = TSNE(n_components=2, perplexity=adaptive_perplexity, method="barnes_hut",
                    init="pca", max_iter=4000, learning_rate="auto",
                    early_exaggeration=early_exag, random_state=42)
        df_tsne = pd.DataFrame(tsne.fit_transform(df), columns=[0, 1])

        ax_flat[idx].set_title(f"early_exaggeration={early_exag}")
        ax_flat[idx].scatter(df_tsne[0], df_tsne[1],
                             color="black", alpha=.33, s=10)
        ax_flat[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(
            output_dir, "tsne_early_exaggeration_comparison.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(
            f"{datetime.now().time()} saved early_exaggeration comparison to {filepath}")
        plt.close()
    else:
        plt.show()
