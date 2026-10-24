import os
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import seaborn as sns


def draw_cluster_boundary(ax, points, color, alpha_fill=0.15, alpha_edge=0.8):
    """Draws a semi-transparent polygon outline enclosing the outer edge points of a cluster."""
    if len(points) < 3:
        return

    try:
        hull = ConvexHull(points)
        hull_pts = points[hull.vertices]

        poly = Polygon(
            hull_pts,
            closed=True,
            facecolor=color,
            edgecolor=color,
            alpha=alpha_fill,
            linewidth=1.8,
            linestyle='--',
            zorder=1,
        )
        ax.add_patch(poly)

        ax.plot(
            np.append(hull_pts[:, 0], hull_pts[0, 0]),
            np.append(hull_pts[:, 1], hull_pts[0, 1]),
            color=color,
            linewidth=1.8,
            alpha=alpha_edge,
            zorder=2,
        )
    except Exception:
        pass


def plot_clusters_from_results(
        results_df,
        algo_title,
        pca_base,
        draw_boundaries=True,
        output_dir='../../figures/clusters/',
        save_fig=True,
):
    """Takes an algorithm results dataframe, plots the 5 subplots (k=2 to k=6),

    saves the figure to disk, and displays it.
    """
    df_multi = (
        results_df[results_df['Subset'] == 'full_multivariate']
        .sort_values(by='k')
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5), sharex=True, sharey=True)
    fig.canvas.manager.set_window_title(f'Cluster Visualizer: {algo_title}')

    palette = sns.color_palette('tab10', 10)

    for idx, row in df_multi.iterrows():
        ax = axes[idx]

        raw = row['Labels']
        if isinstance(raw, str):
            labels_array = np.array(re.findall(r'-?\d+', raw), dtype=int)
        else:
            labels_array = np.asarray(raw, dtype=int)

        plot_df = pca_base.copy()
        plot_df['Cluster'] = pd.Categorical(labels_array)

        if draw_boundaries:
            unique_clusters = np.unique(labels_array)
            for cluster_id in unique_clusters:
                if cluster_id == -1:
                    continue

                cluster_pts = plot_df.loc[
                    plot_df['Cluster'] == cluster_id, ['PC1', 'PC2']
                ].values
                cluster_color = palette[cluster_id % len(palette)]
                draw_cluster_boundary(ax, cluster_pts, cluster_color)

        sns.scatterplot(
            data=plot_df,
            x='PC1',
            y='PC2',
            hue='Cluster',
            palette='tab10',
            alpha=0.85,
            s=45,
            ax=ax,
            legend='full',
            zorder=3,
        )

        ax.set_title(
            f"k = {row['k']}\nSil: {row['Silhouette']} | ARI: {row['ARI']}",
            fontsize=11,
            fontweight='bold',
        )
        ax.set_xlabel('PC1')
        if idx == 0:
            ax.set_ylabel('PC2')
        else:
            ax.set_ylabel('')

    plt.suptitle(
        f'{algo_title}: Full Multivariate Clusters across k in [2, 6]',
        fontsize=14,
        fontweight='bold',
        y=1.04,
    )
    plt.tight_layout()

    # Save before plt.show()
    if save_fig:
        os.makedirs(output_dir, exist_ok=True)
        filename = (
            f"{algo_title.lower().replace(' ', '_').replace('-', '_')}_clusters.png"
        )
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved figure to: {save_path}')

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    Kmeans_results = pd.read_csv('../../data/processed/individual_models/kmeans_results.csv')
    Agglomerative_results = pd.read_csv('../../data/processed/individual_models/agglomerative_results.csv')
    GMM_results = pd.read_csv('../../data/processed/individual_models/gmm_results.csv')
    BIRCH_results = pd.read_csv('../../data/processed/individual_models/birch_results.csv')

    pca_base_coords = pd.read_csv('../../data/processed/pca_base_coords.csv')

    plot_clusters_from_results(Kmeans_results, 'KMeans', pca_base_coords, save_fig=True)
    plot_clusters_from_results(Agglomerative_results, 'Agglomerative', pca_base_coords, save_fig=True)
    plot_clusters_from_results(GMM_results, 'GMM', pca_base_coords, save_fig=True)
    plot_clusters_from_results(BIRCH_results, 'BIRCH', pca_base_coords, save_fig=True)
