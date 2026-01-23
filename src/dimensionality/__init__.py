"""Dimensionality reduction methods for Pokemon Auto Chess Meta Report"""

from .tsne import (
    apply_tsne,
    get_perplexity_range,
    plot_tsne_parameters,
    plot_tsne_init_comparison,
    plot_tsne_metric_comparison,
    plot_tsne_learning_rate_comparison,
    plot_tsne_early_exaggeration_comparison,
)

from .umap import apply_umap, plot_umap_parameters_grid

from .pca import apply_pca

from .phate import (
    apply_phate,
    plot_phate_parameters_grid,
    plot_phate_mds_comparison,
)

from .isomap import apply_isomap, plot_isomap_parameters

from .spectral import (
    apply_spectral_embedding,
    plot_spectral_embedding_parameters_grid,
)

from .pacmap import apply_pacmap, plot_pacmap_parameters_grid, plot_pacmap_n_neighbors_grid

__all__ = [
    # t-SNE
    "apply_tsne",
    "get_perplexity_range",
    "plot_tsne_parameters",
    "plot_tsne_init_comparison",
    "plot_tsne_metric_comparison",
    "plot_tsne_learning_rate_comparison",
    "plot_tsne_early_exaggeration_comparison",
    # UMAP
    "apply_umap",
    "plot_umap_parameters_grid",
    # PCA
    "apply_pca",
    # PHATE
    "apply_phate",
    "plot_phate_parameters_grid",
    "plot_phate_mds_comparison",
    # Isomap
    "apply_isomap",
    "plot_isomap_parameters",
    # Spectral
    "apply_spectral_embedding",
    "plot_spectral_embedding_parameters_grid",
    # PaCMAP
    "apply_pacmap",
    "plot_pacmap_parameters_grid",
    "plot_pacmap_n_neighbors_grid",
]
