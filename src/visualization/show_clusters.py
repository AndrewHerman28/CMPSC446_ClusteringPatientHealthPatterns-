from pathlib import Path

import pandas as pd

from src.evaluation.clustering_accuracy import ALGORITHM_FILES
from src.paths import FIGURES, PCA_BASE_CSV
from src.visualization.plot_utils import plot_cluster_panels, select_one_per_k

ALGO_FIG_DIR = FIGURES / 'algorithm_clusters'
K_RANGE = (2, 6)


def plot_clusters_from_results(
        results_df,
        algo_title,
        pca_base,
        draw_boundaries=True,
        output_dir=ALGO_FIG_DIR,
        save_fig=True,
        show=True,
):
    """Plots full-multivariate clusters for k in [2, 6] on the PC1/PC2 plane.

    Algorithms with one run per k are plotted as they are; for DBSCAN (several
    settings per cluster count) the best run by ARI is shown for each k.
    """
    df_multi = results_df[
        (results_df['Subset'] == 'full_multivariate')
        & results_df['k'].between(*K_RANGE)
    ]
    if df_multi.empty:
        print(f'No full-multivariate runs with k in {K_RANGE} for {algo_title}; skipping.')
        return
    rows = select_one_per_k(df_multi, metric='ARI')

    suptitle = f'{algo_title}: Full Multivariate Clusters across k in [{K_RANGE[0]}, {K_RANGE[1]}]'
    if 'Eps' in rows.columns:
        suptitle += ' (best eps / min_samples per k)'

    filename = f"{algo_title.lower().replace(' ', '_').replace('-', '_')}_clusters.png"
    plot_cluster_panels(
        rows,
        pca_base,
        suptitle=suptitle,
        window_title=f'Cluster Visualizer: {algo_title}',
        save_path=Path(output_dir) / filename if save_fig else None,
        draw_boundaries=draw_boundaries,
        show=show,
    )


def main(show=True):
    pca_base_coords = pd.read_csv(PCA_BASE_CSV)

    for algo, path in ALGORITHM_FILES.items():
        if not path.exists():
            print(f'Skipping {algo}: {path.name} not found.')
            continue
        plot_clusters_from_results(pd.read_csv(path), algo, pca_base_coords, show=show)


if __name__ == '__main__':
    main()
