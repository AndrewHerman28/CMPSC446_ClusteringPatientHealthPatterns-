import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.clustering_accuracy import parse_labels
from src.paths import FIGURES, PCA_BASE_CSV, PCA_RESULTS_CSV, PCA_VARIANCE_CSV
from src.visualization.plot_utils import (
    _panel_title, _set_window_title, axis_label, build_palette, plot_cluster_panels,
    select_one_per_k,
)

DEFAULT_OUTPUT_DIR = FIGURES / 'pca_clusters'
K_RANGE = (2, 6)


def _figure_path(dimensionality, target_algo, output_base_dir):
    """<output_base_dir>/2D or /3D / <algo>_clusters.png"""
    sub_dir = '2D' if str(dimensionality).endswith('2D') else '3D'
    clean_algo_name = (
        target_algo.lower()
        .replace('pca_', '')
        .replace(' ', '_')
        .replace('-', '_')
        .replace('+', '')
    )
    return output_base_dir / sub_dir / f'{clean_algo_name}_clusters.png'


def _select(pca_results_df, dimensionality, target_algo):
    rows = pca_results_df[
        (pca_results_df['Dimensionality'] == dimensionality)
        & (pca_results_df['Algorithm'] == target_algo)
        ]
    if target_algo.endswith('DBSCAN') and not rows.empty:
        # DBSCAN has several settings per cluster count: keep k in range and
        # show the best run (by ARI) for each k.
        rows = rows[rows['k'].between(*K_RANGE)]
        if not rows.empty:
            rows = select_one_per_k(rows, metric='ARI')
    return rows.sort_values(by='k').reset_index(drop=True)


def _suptitle(dimensionality, target_algo):
    title = f'Clusters on {dimensionality} using {target_algo} (k in [2, 6])'
    if target_algo.endswith('DBSCAN'):
        title += '\nbest eps / min_samples per k'
    return title


def plot_pca_clusters_from_results(
        pca_results_df,
        pca_base,
        dimensionality='PCA_2D',
        target_algo='PCA_k-Means++',
        draw_boundaries=True,
        output_base_dir=DEFAULT_OUTPUT_DIR,
        save_fig=True,
        show=True,
        explained=None,
):
    """2D plots (PC1 vs PC2): one subplot per k (2 to 6) for one algorithm."""
    rows = _select(pca_results_df, dimensionality, target_algo)
    if rows.empty:
        print(f'No records found for {dimensionality} using {target_algo}.')
        return

    plot_cluster_panels(
        rows,
        pca_base,
        suptitle=_suptitle(dimensionality, target_algo),
        window_title=f'PCA Visualizer: {dimensionality} - {target_algo}',
        save_path=_figure_path(dimensionality, target_algo, output_base_dir) if save_fig else None,
        draw_boundaries=draw_boundaries,
        show=show,
        explained=explained,
    )


def plot_pca_clusters_3d(
        pca_results_df,
        pca_base,
        target_algo='PCA_k-Means++',
        dimensionality='PCA_3D',
        output_base_dir=DEFAULT_OUTPUT_DIR,
        save_fig=True,
        show=True,
        explained=None,
        elev=22,
        azim=-55,
):
    """True 3D scatter (PC1, PC2, PC3): one subplot per k (2 to 6), 3 per row."""
    if 'PC3' not in pca_base.columns:
        raise ValueError(
            "pca_base_coords.csv has no PC3 column. Re-run python main.py "
            "to regenerate it."
        )

    rows = _select(pca_results_df, dimensionality, target_algo)
    if rows.empty:
        print(f'No records found for {dimensionality} using {target_algo}.')
        return

    n_plots = len(rows)
    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig = plt.figure(figsize=(7 * n_cols, 6.5 * n_rows))
    _set_window_title(fig, f'PCA Visualizer: {dimensionality} - {target_algo}')

    for idx, row in rows.iterrows():
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        labels = parse_labels(row['Labels'])
        if len(labels) != len(pca_base):
            raise ValueError(
                f'{len(labels)} labels but {len(pca_base)} PCA points. '
                'Re-run python main.py so all outputs come from the same data.'
            )
        palette = build_palette(labels)

        for cluster_id, color in palette.items():
            mask = labels == cluster_id
            ax.scatter(
                pca_base.loc[mask, 'PC1'],
                pca_base.loc[mask, 'PC2'],
                pca_base.loc[mask, 'PC3'],
                color=color,
                s=24,
                alpha=0.8,
                edgecolors='none',
                label='Noise' if cluster_id == -1 else f'Cluster {cluster_id}',
            )

        ax.set_xlabel(axis_label('PC1', explained), labelpad=8)
        ax.set_ylabel(axis_label('PC2', explained), labelpad=8)
        ax.set_zlabel(axis_label('PC3', explained), labelpad=8)
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc='upper left', fontsize=8, frameon=True)
        ax.set_title(_panel_title(row), fontsize=11, fontweight='bold')

    fig.suptitle(
        _suptitle(dimensionality, target_algo),
        fontsize=14,
        fontweight='bold',
    )
    fig.tight_layout()

    if save_fig:
        save_path = _figure_path(dimensionality, target_algo, output_base_dir)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved figure to: {save_path}')

    if show:
        plt.show()
    plt.close(fig)


def main(show=True):
    pca_results = pd.read_csv(PCA_RESULTS_CSV)
    pca_base_coords = pd.read_csv(PCA_BASE_CSV)

    explained = None
    if PCA_VARIANCE_CSV.exists():
        var_df = pd.read_csv(PCA_VARIANCE_CSV)
        explained = dict(zip(var_df['Component'], var_df['ExplainedVariance']))

    # Iterate dynamically through all projections and algorithms present in results
    for dim in pca_results['Dimensionality'].unique():
        for algo in pca_results['Algorithm'].unique():
            if dim == 'PCA_3D':
                plot_pca_clusters_3d(
                    pca_results, pca_base_coords, target_algo=algo,
                    dimensionality=dim, show=show, explained=explained,
                )
            else:
                plot_pca_clusters_from_results(
                    pca_results, pca_base_coords, dimensionality=dim,
                    target_algo=algo, show=show, explained=explained,
                )


if __name__ == '__main__':
    main()