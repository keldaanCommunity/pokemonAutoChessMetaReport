"""PCA dimensionality reduction methods"""

from sklearn.decomposition import PCA
from datetime import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')


def apply_pca(df, plot=False, save=False, output_dir=None):
    """
    Apply PCA dimensionality reduction to reduce high-dimensional data to 2D.

    PCA is a fast, linear method that preserves global structure well.
    It doesn't have parameters to tune like t-SNE or UMAP.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features to reduce
        plot (bool): Whether to display the plot interactively (default: False)
        save (bool): Whether to save the plot to file (default: False)
        output_dir (str): Directory to save the plot if save=True (default: None)

    Returns:
        pd.DataFrame: DataFrame with 'x' and 'y' columns containing 2D PCA coordinates
    """
    print(f"{datetime.now().time()} applying PCA...")
    pca = PCA(n_components=2)
    df_result = pd.DataFrame(pca.fit_transform(df), columns=["x", "y"])

    # Print explained variance
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.sum(explained_var)
    print(f"{datetime.now().time()} PCA explained variance ratio: {explained_var}")
    print(f"{datetime.now().time()} PCA cumulative explained variance: {cumulative_var:.4f}")

    if plot or save:
        plt.figure(figsize=(10, 8))
        plt.scatter(df_result["x"], df_result["y"],
                    color="black", alpha=0.4, s=20)
        plt.title(
            f"PCA Visualization (Explained Variance: {cumulative_var:.4f})")
        plt.xlabel(f"PC1 ({explained_var[0]:.4f})")
        plt.ylabel(f"PC2 ({explained_var[1]:.4f})")
        plt.grid(True, alpha=0.3)
        if save and output_dir:
            filepath = os.path.join(output_dir, "pca_projection.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"{datetime.now().time()} saved PCA plot to {filepath}")
        if plot:
            plt.show()
        plt.close()
    return df_result
