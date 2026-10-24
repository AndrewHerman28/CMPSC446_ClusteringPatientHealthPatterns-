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


def plot_pca_clusters_from_results(
        pca_results_df,
        pca_base,
        dimensionality='PCA_2D',
        target_algo='k-Means++',
        draw_boundaries=True,
        output_base_dir='../../figures/pca_clusters',
        save_fig=True,
):
    """Takes PCA evaluation results, plots subplots (k=2 to k=6) for a specified algorithm,

    saves the figure to the corresponding 2D or 3D subdirectory, and displays it.
    """
    df_filtered = pca_results_df[
        (pca_results_df['Dimensionality'] == dimensionality)
        & (pca_results_df['Algorithm'] == target_algo)
        ].sort_values(by='k').reset_index(drop=True)

    if df_filtered.empty:
        print(f'No records found for {dimensionality} using {target_algo}.')
        return

    n_plots = len(df_filtered)
    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(5 * n_plots, 5.5),
        sharex=True,
        sharey=True,
    )
    fig.canvas.manager.set_window_title(
        f'PCA Visualizer: {dimensionality} - {target_algo}'
    )

    if n_plots == 1:
        axes = [axes]

    palette = sns.color_palette('tab10', 10)

    for idx, row in df_filtered.iterrows():
        ax = axes[idx]

        # Robust label parsing
        raw = row['Labels']
        if isinstance(raw, str):
            labels_array = np.array(re.findall(r'-?\d+', raw), dtype=int)
        else:
            labels_array = np.asarray(raw, dtype=int)

        plot_df = pca_base.copy()
        plot_df['Cluster'] = pd.Categorical(labels_array)

        # 1. Draw outer boundary polygon for each cluster
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

        # 2. Scatter plot points on top
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
        f'Clusters on {dimensionality} using {target_algo} (k in [2, 6])',
        fontsize=14,
        fontweight='bold',
        y=1.04,
    )
    plt.tight_layout()

    # Route directory to figures/pca_clusters/2D or figures/pca_clusters/3D
    if save_fig:
        sub_dir = '2D' if '2' in str(dimensionality) else '3D'
        target_dir = os.path.join(output_base_dir, sub_dir)
        os.makedirs(target_dir, exist_ok=True)

        clean_algo_name = (
            target_algo.lower()
            .replace('pca_', '')
            .replace(' ', '_')
            .replace('-', '_')
            .replace('+', '')
        )
        filename = f'{clean_algo_name}_clusters.png'
        save_path = os.path.join(target_dir, filename)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved figure to: {save_path}')

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    PCA_results = pd.read_csv('../../data/processed/pca_results.csv')
    pca_base_coords = pd.read_csv('../../data/processed/pca_base_coords.csv')

    # Iterate dynamically through all projections and algorithms present in results
    for dim in PCA_results['Dimensionality'].unique():
        for algo in PCA_results['Algorithm'].unique():
            plot_pca_clusters_from_results(
                PCA_results,
                pca_base_coords,
                dimensionality=dim,
                target_algo=algo,
                draw_boundaries=True,
                output_base_dir='../../figures/clusters/pca_clusters',
                save_fig=True,
            )
