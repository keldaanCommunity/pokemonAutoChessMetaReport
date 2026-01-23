"""NMF (Non-Negative Matrix Factorization) dimensionality reduction methods"""

from datetime import datetime
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import matplotlib
matplotlib.use('Agg')


def apply_nmf(df, n_components=2, init='random', solver='cd', beta_loss='frobenius',
              alpha_W=0.0, alpha_H='same', l1_ratio=0.0, random_state=0,
              plot=False, save=False, output_dir=None):
    """
    Apply NMF (Non-Negative Matrix Factorization) to reduce high-dimensional data to 2D.

    NMF decomposes non-negative data into non-negative components, making it ideal for
    count-based data like synergy scores. It finds latent factors that explain the data.

    Args:
        df (pd.DataFrame): Input DataFrame with non-negative features to reduce
        n_components (int): Number of components to extract (default: 2, for 2D visualization)
        init (str): Type of factor matrix initialization 
            ('random', 'nndsvd', 'nndsvda', 'nndsvdar') (default: 'random')
        solver (str): Numerical solver ('cd' for Coordinate Descent, 'mu' for Multiplicative Update)
            (default: 'cd', generally faster)
        beta_loss (str): Divergence measure ('frobenius', 'kullback-leibler', 'itakura-saito')
            (default: 'frobenius', fastest)
        alpha_W (float): Regularization for W matrix (default: 0.0, no regularization)
        alpha_H (float or 'same'): Regularization for H matrix (default: 'same' as alpha_W)
        l1_ratio (float): Mix between L1 (1.0) and L2 (0.0) regularization (default: 0.0, pure L2)
        random_state (int): Random seed for reproducibility (default: 0)
        plot (bool): Whether to display the plot interactively (default: False)
        save (bool): Whether to save the plot to file (default: False)
        output_dir (str): Directory to save the plot if save=True (default: None)

    Returns:
        pd.DataFrame: DataFrame with 'x' and 'y' columns containing 2D NMF coordinates
    """
    print(f"{datetime.now().time()} NMF with n_components={n_components}, init={init}, solver={solver}, alpha_W={alpha_W}...")

    # Ensure all values are non-negative (NMF requirement)
    df_values = df.values
    df_min = np.min(df_values)
    if df_min < 0:
        print(
            f"{datetime.now().time()} warning: NMF requires non-negative data, shifting by {-df_min}")
        df_values = df_values - df_min

    nmf = NMF(n_components=n_components, init=init, solver=solver, beta_loss=beta_loss,
              alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio,
              random_state=random_state, max_iter=500)
    components = nmf.fit_transform(df_values)
    df_result = pd.DataFrame(components, columns=[
                             f"component_{i}" for i in range(n_components)])

    # Rename to x, y for consistency
    if n_components >= 2:
        df_result = df_result.rename(columns={
            f"component_0": "x",
            f"component_1": "y"
        })

    if plot or save:
        plt.figure(figsize=(10, 8))
        plt.scatter(df_result["x"], df_result["y"],
                    color="black", alpha=0.4, s=20)
        plt.title(
            f"NMF (init={init}, solver={solver}, alpha_W={alpha_W})")
        plt.xlabel("NMF Component 1")
        plt.ylabel("NMF Component 2")
        plt.grid(True, alpha=0.3)
        if save and output_dir:
            filepath = os.path.join(
                output_dir, f"nmf_nc{n_components}_init{init}_solver{solver}_alphaW{alpha_W}.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"{datetime.now().time()} saved plot to {filepath}")
        if plot:
            plt.show()
        plt.close()

    return df_result


def plot_nmf_parameters_grid(df, init_values=['random', 'nndsvd', 'nndsvda', 'nndsvdar'],
                             solver_values=['cd', 'mu'], output_dir=None):
    """
    Create a 4x2 grid showing NMF results with different initialization methods and solvers.

    Combines initialization methods (rows) with solvers (columns) in a single comprehensive grid.

    Args:
        df (pd.DataFrame): Input DataFrame with non-negative features
        init_values (list): List of initialization methods to test (default: ['random', 'nndsvd', 'nndsvda', 'nndsvdar'])
        solver_values (list): List of solvers to compare (default: ['cd', 'mu'])
        output_dir (str): Directory to save the grid plot if provided (default: None)

    Returns:
        None: Displays or saves the grid of NMF visualizations
    """
    n_inits = len(init_values)
    n_solvers = len(solver_values)
    fig, ax = plt.subplots(n_inits, n_solvers, figsize=[14, 16])

    if n_inits == 1 and n_solvers == 1:
        ax_flat = [[ax]]
    elif n_inits == 1:
        ax_flat = [ax]
    elif n_solvers == 1:
        ax_flat = [[a] for a in ax]
    else:
        ax_flat = ax

    # Ensure all values are non-negative
    df_values = df.values
    df_min = np.min(df_values)
    if df_min < 0:
        df_values = df_values - df_min

    total_plots = n_inits * n_solvers
    plot_count = 0
    for i_idx, init_method in enumerate(init_values):
        for s_idx, solver in enumerate(solver_values):
            plot_count += 1
            print(
                f"{datetime.now().time()} subplot {plot_count}/{total_plots}: init={init_method}, solver={solver}...")
            try:
                nmf = NMF(n_components=2, init=init_method, solver=solver,
                          alpha_W=0.0, alpha_H='same', random_state=0, max_iter=500)
                components = nmf.fit_transform(df_values)

                if n_inits > 1 and n_solvers > 1:
                    sub_plt = ax_flat[i_idx, s_idx]
                elif n_inits > 1:
                    sub_plt = ax_flat[i_idx][0]
                elif n_solvers > 1:
                    sub_plt = ax_flat[0][s_idx]
                else:
                    sub_plt = ax_flat[0][0]

                sub_plt.set_title(
                    f"init={init_method}\nsolver={solver}", fontsize=9)
                sub_plt.scatter(
                    components[:, 0], components[:, 1], color="black", alpha=.33, s=3)
                sub_plt.grid(True, alpha=0.3)
                sub_plt.set_xlabel("Component 1", fontsize=8)
                sub_plt.set_ylabel("Component 2", fontsize=8)
            except Exception as e:
                print(
                    f"{datetime.now().time()} warning: init={init_method}, solver={solver} failed: {e}")
                if n_inits > 1 and n_solvers > 1:
                    sub_plt = ax_flat[i_idx, s_idx]
                elif n_inits > 1:
                    sub_plt = ax_flat[i_idx][0]
                elif n_solvers > 1:
                    sub_plt = ax_flat[0][s_idx]
                else:
                    sub_plt = ax_flat[0][0]
                sub_plt.text(
                    0.5, 0.5, f"Error:\n{str(e)[:25]}", ha='center', va='center', fontsize=8)
                sub_plt.set_xticks([])
                sub_plt.set_yticks([])

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "nmf_parameters_grid.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved NMF parameters grid to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_nmf_init_comparison(df, init_values=['random', 'nndsvd', 'nndsvda', 'nndsvdar'],
                             output_dir=None):
    """
    Compare different NMF initialization methods side by side in a 1x4 grid.

    Args:
        df (pd.DataFrame): Input DataFrame with non-negative features
        init_values (list): List of initialization methods to compare
        output_dir (str): Directory to save the comparison plot if provided (default: None)

    Returns:
        None: Displays or saves the comparison visualization
    """
    n_inits = len(init_values)
    fig, ax = plt.subplots(1, n_inits, figsize=[16, 4])
    ax_flat = [ax] if n_inits == 1 else ax

    # Ensure all values are non-negative
    df_values = df.values
    df_min = np.min(df_values)
    if df_min < 0:
        df_values = df_values - df_min

    for idx, init_method in enumerate(init_values):
        print(
            f"{datetime.now().time()} NMF init comparison {idx+1}/{n_inits}: init={init_method}...")
        try:
            nmf = NMF(n_components=2, init=init_method, solver='cd',
                      alpha_W=0.0, alpha_H='same', random_state=0, max_iter=500)
            components = nmf.fit_transform(df_values)

            ax_flat[idx].set_title(f"init={init_method}")
            ax_flat[idx].scatter(components[:, 0], components[:, 1],
                                 color="black", alpha=.33, s=10)
            ax_flat[idx].grid(True, alpha=0.3)
        except Exception as e:
            print(f"{datetime.now().time()} warning: init={init_method} failed: {e}")
            ax_flat[idx].text(
                0.5, 0.5, f"Error:\n{str(e)[:30]}", ha='center', va='center')
            ax_flat[idx].set_xticks([])
            ax_flat[idx].set_yticks([])

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "nmf_init_comparison.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved NMF init comparison to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_nmf_regularization_grid(df, init_method='nndsvda',
                                 alpha_W_values=[0.0, 0.1, 0.5],
                                 beta_loss_values=[
                                     'frobenius', 'kullback-leibler', 'itakura-saito'],
                                 output_dir=None):
    """
    Create a 3x3 grid testing different regularization and loss parameters.

    Combines alpha_W regularization (rows) with beta_loss divergence measures (columns).
    Automatically selects appropriate solver for each beta_loss (cd for frobenius, mu for others).

    Args:
        df (pd.DataFrame): Input DataFrame with non-negative features
        init_method (str): Initialization method to use (default: 'nndsvda')
        alpha_W_values (list): Regularization values to test (default: [0.0, 0.1, 0.5])
        beta_loss_values (list): Loss functions to test (default: ['frobenius', 'kullback-leibler', 'itakura-saito'])
        output_dir (str): Directory to save the grid plot if provided (default: None)

    Returns:
        None: Displays or saves the grid of NMF visualizations
    """
    n_alphas = len(alpha_W_values)
    n_losses = len(beta_loss_values)
    fig, ax = plt.subplots(n_alphas, n_losses, figsize=[14, 12])

    if n_alphas == 1 and n_losses == 1:
        ax_flat = [[ax]]
    elif n_alphas == 1:
        ax_flat = [ax]
    elif n_losses == 1:
        ax_flat = [[a] for a in ax]
    else:
        ax_flat = ax

    # Ensure all values are non-negative
    df_values = df.values
    df_min = np.min(df_values)
    if df_min < 0:
        df_values = df_values - df_min

    total_plots = n_alphas * n_losses
    plot_count = 0
    for a_idx, alpha_w in enumerate(alpha_W_values):
        for l_idx, beta_loss in enumerate(beta_loss_values):
            plot_count += 1
            print(
                f"{datetime.now().time()} subplot {plot_count}/{total_plots}: alpha_W={alpha_w}, beta_loss={beta_loss}...")
            try:
                # Select appropriate solver for beta_loss: cd for frobenius, mu for others
                solver = 'cd' if beta_loss == 'frobenius' else 'mu'

                nmf = NMF(n_components=2, init=init_method, solver=solver,
                          alpha_W=alpha_w, alpha_H='same', beta_loss=beta_loss,
                          random_state=0, max_iter=500)
                components = nmf.fit_transform(df_values)

                if n_alphas > 1 and n_losses > 1:
                    sub_plt = ax_flat[a_idx, l_idx]
                elif n_alphas > 1:
                    sub_plt = ax_flat[a_idx][0]
                elif n_losses > 1:
                    sub_plt = ax_flat[0][l_idx]
                else:
                    sub_plt = ax_flat[0][0]

                sub_plt.set_title(
                    f"alpha_W={alpha_w}\nbeta={beta_loss[:6]}", fontsize=9)
                sub_plt.scatter(
                    components[:, 0], components[:, 1], color="black", alpha=.33, s=3)
                sub_plt.grid(True, alpha=0.3)
                sub_plt.set_xlabel("Component 1", fontsize=8)
                sub_plt.set_ylabel("Component 2", fontsize=8)
            except Exception as e:
                print(
                    f"{datetime.now().time()} warning: alpha_W={alpha_w}, beta_loss={beta_loss} failed: {e}")
                if n_alphas > 1 and n_losses > 1:
                    sub_plt = ax_flat[a_idx, l_idx]
                elif n_alphas > 1:
                    sub_plt = ax_flat[a_idx][0]
                elif n_losses > 1:
                    sub_plt = ax_flat[0][l_idx]
                else:
                    sub_plt = ax_flat[0][0]
                sub_plt.text(
                    0.5, 0.5, f"Error:\n{str(e)[:25]}", ha='center', va='center', fontsize=8)
                sub_plt.set_xticks([])
                sub_plt.set_yticks([])

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "nmf_regularization_grid.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved NMF regularization grid to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_nmf_solver_comparison(df, solver_values=['cd', 'mu'], output_dir=None):
    """
    Compare different NMF solvers side by side in a 1x2 grid.

    Args:
        df (pd.DataFrame): Input DataFrame with non-negative features
        solver_values (list): List of solvers to compare (default: ['cd', 'mu'])
        output_dir (str): Directory to save the comparison plot if provided (default: None)

    Returns:
        None: Displays or saves the comparison visualization
    """
    n_solvers = len(solver_values)
    fig, ax = plt.subplots(1, n_solvers, figsize=[12, 4])
    ax_flat = [ax] if n_solvers == 1 else ax

    # Ensure all values are non-negative
    df_values = df.values
    df_min = np.min(df_values)
    if df_min < 0:
        df_values = df_values - df_min

    for idx, solver in enumerate(solver_values):
        print(
            f"{datetime.now().time()} NMF solver comparison {idx+1}/{n_solvers}: solver={solver}...")
        try:
            nmf = NMF(n_components=2, init='nndsvda', solver=solver,
                      alpha_W=0.0, alpha_H='same', random_state=0, max_iter=500)
            components = nmf.fit_transform(df_values)

            ax_flat[idx].set_title(f"solver={solver}")
            ax_flat[idx].scatter(components[:, 0], components[:, 1],
                                 color="black", alpha=.33, s=10)
            ax_flat[idx].grid(True, alpha=0.3)
        except Exception as e:
            print(f"{datetime.now().time()} warning: solver={solver} failed: {e}")
            ax_flat[idx].text(
                0.5, 0.5, f"Error:\n{str(e)[:30]}", ha='center', va='center')
            ax_flat[idx].set_xticks([])
            ax_flat[idx].set_yticks([])

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "nmf_solver_comparison.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"{datetime.now().time()} saved NMF solver comparison to {filepath}")
        plt.close()
    else:
        plt.show()
