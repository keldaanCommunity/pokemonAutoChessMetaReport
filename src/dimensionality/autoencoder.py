"""Autoencoder dimensionality reduction methods using PyTorch"""

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')


class Autoencoder(nn.Module):
    """
    Simple feedforward autoencoder for dimensionality reduction.

    Architecture: input -> encoder -> latent (2D) -> decoder -> output
    """

    def __init__(self, input_dim, hidden_dims=[64, 32], latent_dim=2):
        """
        Args:
            input_dim (int): Number of input features
            hidden_dims (list): List of hidden layer dimensions for encoder
            latent_dim (int): Dimension of latent space (default: 2 for visualization)
        """
        super(Autoencoder, self).__init__()

        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder layers (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def encode(self, x):
        return self.encoder(x)


def apply_autoencoder(df, hidden_dims=[64, 32], latent_dim=2, epochs=100,
                      learning_rate=0.001, batch_size=256,
                      plot=False, save=False, output_dir=None):
    """
    Apply Autoencoder dimensionality reduction to reduce high-dimensional data to 2D.

    Autoencoders learn a compressed representation by training a neural network
    to reconstruct the input through a bottleneck layer. The bottleneck (latent space)
    captures the most important features of the data.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features to reduce
        hidden_dims (list): Hidden layer dimensions for encoder (default: [64, 32])
        latent_dim (int): Dimension of latent space (default: 2 for visualization)
        epochs (int): Number of training epochs (default: 100)
        learning_rate (float): Learning rate for Adam optimizer (default: 0.001)
        batch_size (int): Batch size for training (default: 256)
        plot (bool): Whether to display the plot interactively (default: False)
        save (bool): Whether to save the plot to file (default: False)
        output_dir (str): Directory to save the plot if save=True (default: None)

    Returns:
        pd.DataFrame: DataFrame with 'x' and 'y' columns containing 2D autoencoder coordinates
    """
    print(f"{datetime.now().time()} Autoencoder with hidden_dims={hidden_dims}, latent_dim={latent_dim}, epochs={epochs}...")

    # Prepare data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.FloatTensor(df.values.copy())

    # Normalize data for better training
    data_mean = data.mean(dim=0)
    data_std = data.std(dim=0) + 1e-8
    data_normalized = (data - data_mean) / data_std

    dataset = TensorDataset(data_normalized)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create and train model
    input_dim = df.shape[1]
    model = Autoencoder(input_dim, hidden_dims, latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            print(
                f"{datetime.now().time()} Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # Get latent representations
    model.eval()
    with torch.no_grad():
        latent = model.encode(data_normalized.to(device)).cpu().numpy()

    df_result = pd.DataFrame(latent[:, :2], columns=["x", "y"])

    if plot or save:
        plt.figure(figsize=(10, 8))
        plt.scatter(df_result["x"], df_result["y"],
                    color="black", alpha=0.4, s=20)
        plt.title(f"Autoencoder (hidden={hidden_dims}, epochs={epochs})")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.grid(True, alpha=0.3)
        if save and output_dir:
            filepath = os.path.join(
                output_dir, f"autoencoder_h{'_'.join(map(str, hidden_dims))}_e{epochs}.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"{datetime.now().time()} saved plot to {filepath}")
        if plot:
            plt.show()
        plt.close()

    return df_result


def plot_autoencoder_parameters_grid(df, hidden_dims_values=[[32], [64, 32], [128, 64, 32]],
                                     epochs_values=[50, 100, 200], output_dir=None):
    """
    Create a grid of subplots showing Autoencoder results with different architectures and epochs.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        hidden_dims_values (list): List of hidden dimension configurations to test
        epochs_values (list): List of epoch values to test
        output_dir (str): Directory to save the grid plot if provided (default: None)

    Returns:
        None: Displays or saves the grid of Autoencoder visualizations
    """
    n_archs = len(hidden_dims_values)
    n_epochs = len(epochs_values)
    total_plots = n_archs * n_epochs

    fig, ax = plt.subplots(n_archs, n_epochs, figsize=[14, 12])

    if n_archs == 1 and n_epochs == 1:
        ax_flat = [[ax]]
    elif n_archs == 1:
        ax_flat = [ax]
    elif n_epochs == 1:
        ax_flat = [[a] for a in ax]
    else:
        ax_flat = ax

    # Prepare data once
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.FloatTensor(df.values)
    data_mean = data.mean(dim=0)
    data_std = data.std(dim=0) + 1e-8
    data_normalized = (data - data_mean) / data_std
    input_dim = df.shape[1]

    print(f"{datetime.now().time()} testing {total_plots} Autoencoder configurations...")

    plot_count = 0
    for a_idx, hidden_dims in enumerate(hidden_dims_values):
        for e_idx, epochs in enumerate(epochs_values):
            plot_count += 1
            arch_str = "x".join(map(str, hidden_dims))
            print(
                f"{datetime.now().time()} subplot {plot_count}/{total_plots}: arch={arch_str}, epochs={epochs}...")

            try:
                # Train model
                model = Autoencoder(input_dim, hidden_dims,
                                    latent_dim=2).to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                dataset = TensorDataset(data_normalized)
                dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

                model.train()
                for epoch in range(epochs):
                    for batch in dataloader:
                        x = batch[0].to(device)
                        optimizer.zero_grad()
                        reconstructed, _ = model(x)
                        loss = criterion(reconstructed, x)
                        loss.backward()
                        optimizer.step()

                # Get latent representations
                model.eval()
                with torch.no_grad():
                    latent = model.encode(
                        data_normalized.to(device)).cpu().numpy()

                if n_archs > 1 and n_epochs > 1:
                    sub_plt = ax_flat[a_idx, e_idx]
                elif n_archs > 1:
                    sub_plt = ax_flat[a_idx][0]
                elif n_epochs > 1:
                    sub_plt = ax_flat[0][e_idx]
                else:
                    sub_plt = ax_flat[0][0]

                sub_plt.set_title(
                    f"arch={arch_str}\nepochs={epochs}", fontsize=9)
                sub_plt.scatter(latent[:, 0], latent[:, 1],
                                color="black", alpha=.33, s=3)
                sub_plt.grid(True, alpha=0.3)

            except Exception as e:
                print(
                    f"{datetime.now().time()} warning: arch={arch_str}, epochs={epochs} failed: {e}")
                if n_archs > 1 and n_epochs > 1:
                    sub_plt = ax_flat[a_idx, e_idx]
                elif n_archs > 1:
                    sub_plt = ax_flat[a_idx][0]
                elif n_epochs > 1:
                    sub_plt = ax_flat[0][e_idx]
                else:
                    sub_plt = ax_flat[0][0]
                sub_plt.text(
                    0.5, 0.5, f"Error:\n{str(e)[:25]}", ha='center', va='center', fontsize=8)
                sub_plt.set_xticks([])
                sub_plt.set_yticks([])

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "autoencoder_parameters_grid.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(
            f"{datetime.now().time()} saved Autoencoder parameters grid to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_autoencoder_learning_rate_comparison(df, lr_values=[0.005, 0.01, 0.02],
                                              hidden_dims=[128, 64, 32], epochs=200, output_dir=None):
    """
    Compare different learning rates for Autoencoder.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        lr_values (list): List of learning rates to compare (default: [0.005, 0.01, 0.02])
        hidden_dims (list): Hidden layer dimensions (default: [128, 64, 32])
        epochs (int): Number of training epochs (default: 200)
        output_dir (str): Directory to save the comparison plot if provided (default: None)

    Returns:
        None: Displays or saves the comparison visualization
    """
    n_lrs = len(lr_values)
    fig, ax = plt.subplots(1, n_lrs, figsize=[16, 4])
    ax_flat = [ax] if n_lrs == 1 else ax

    # Prepare data once
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.FloatTensor(df.values)
    data_mean = data.mean(dim=0)
    data_std = data.std(dim=0) + 1e-8
    data_normalized = (data - data_mean) / data_std
    input_dim = df.shape[1]

    print(f"{datetime.now().time()} comparing {n_lrs} Autoencoder learning rates...")

    for idx, lr in enumerate(lr_values):
        print(
            f"{datetime.now().time()} Autoencoder learning rate comparison {idx+1}/{n_lrs}: lr={lr}...")
        try:
            model = Autoencoder(input_dim, hidden_dims,
                                latent_dim=2).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            dataset = TensorDataset(data_normalized)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

            model.train()
            for epoch in range(epochs):
                for batch in dataloader:
                    x = batch[0].to(device)
                    optimizer.zero_grad()
                    reconstructed, _ = model(x)
                    loss = criterion(reconstructed, x)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                latent = model.encode(data_normalized.to(device)).cpu().numpy()

            ax_flat[idx].set_title(f"lr={lr}")
            ax_flat[idx].scatter(latent[:, 0], latent[:, 1],
                                 color="black", alpha=.33, s=10)
            ax_flat[idx].grid(True, alpha=0.3)

        except Exception as e:
            print(f"{datetime.now().time()} warning: lr={lr} failed: {e}")
            ax_flat[idx].text(
                0.5, 0.5, f"Error:\n{str(e)[:30]}", ha='center', va='center')
            ax_flat[idx].set_xticks([])
            ax_flat[idx].set_yticks([])

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(output_dir, "autoencoder_lr_comparison.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(
            f"{datetime.now().time()} saved Autoencoder learning rate comparison to {filepath}")
        plt.close()
    else:
        plt.show()


def plot_autoencoder_architecture_comparison(df, arch_values=[[128, 64, 32], [256, 128, 64], [128, 96, 64, 32], [256, 128, 64, 32]],
                                             epochs=200, output_dir=None):
    """
    Compare different Autoencoder architectures side by side.

    Args:
        df (pd.DataFrame): Input DataFrame with high-dimensional features
        arch_values (list): List of hidden dimension configurations to compare
        epochs (int): Number of training epochs (default: 200)
        output_dir (str): Directory to save the comparison plot if provided (default: None)

    Returns:
        None: Displays or saves the comparison visualization
    """
    n_archs = len(arch_values)
    multi_axes = n_archs > 3
    n_rows = 2 if multi_axes else 1
    n_cols = math.ceil(n_archs / n_rows) if multi_axes else n_archs
    fig, ax = plt.subplots(n_rows, n_cols, figsize=[
                           16, 10 if multi_axes else 4])

    if multi_axes:
        ax_flat = ax.flatten()
    else:
        ax_flat = [ax] if n_archs == 1 else ax

    # Prepare data once
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.FloatTensor(df.values)
    data_mean = data.mean(dim=0)
    data_std = data.std(dim=0) + 1e-8
    data_normalized = (data - data_mean) / data_std
    input_dim = df.shape[1]

    print(f"{datetime.now().time()} comparing {n_archs} Autoencoder architectures...")

    for idx, hidden_dims in enumerate(arch_values):
        arch_str = "x".join(map(str, hidden_dims))
        print(
            f"{datetime.now().time()} Autoencoder arch comparison {idx+1}/{n_archs}: {arch_str}...")
        try:
            model = Autoencoder(input_dim, hidden_dims,
                                latent_dim=2).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            dataset = TensorDataset(data_normalized)
            dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

            model.train()
            for epoch in range(epochs):
                for batch in dataloader:
                    x = batch[0].to(device)
                    optimizer.zero_grad()
                    reconstructed, _ = model(x)
                    loss = criterion(reconstructed, x)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                latent = model.encode(data_normalized.to(device)).cpu().numpy()

            ax_flat[idx].set_title(f"arch={arch_str}")
            ax_flat[idx].scatter(latent[:, 0], latent[:, 1],
                                 color="black", alpha=.33, s=5)
            ax_flat[idx].grid(True, alpha=0.3)

        except Exception as e:
            print(f"{datetime.now().time()} warning: arch={arch_str} failed: {e}")
            ax_flat[idx].text(
                0.5, 0.5, f"Error:\n{str(e)[:30]}", ha='center', va='center')
            ax_flat[idx].set_xticks([])
            ax_flat[idx].set_yticks([])

    # Hide unused subplots
    for idx in range(n_archs, len(ax_flat)):
        ax_flat[idx].axis('off')

    plt.tight_layout()
    if output_dir:
        filepath = os.path.join(
            output_dir, "autoencoder_architecture_comparison.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(
            f"{datetime.now().time()} saved Autoencoder architecture comparison to {filepath}")
        plt.close()
    else:
        plt.show()
