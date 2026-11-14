import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from src.evaluation.clustering_accuracy import parse_labels

NOISE_COLOR = '#9e9e9e'  # DBSCAN noise (label -1)


def build_palette(labels):
    """Explicit {cluster_id: color} so scatter points and hull outlines always match."""
    base = sns.color_palette('tab10', 10)
    return {
        int(c): (NOISE_COLOR if c == -1 else base[int(c) % len(base)])
        for c in np.unique(labels)
    }


def draw_cluster_boundary(ax, points, color, alpha_fill=0.15, alpha_edge=0.8):
    """Draws a semi-transparent convex-hull polygon around a cluster."""
    if len(points) < 3:
        return

    try:
        hull = ConvexHull(points)
        hull_pts = points[hull.vertices]

        ax.add_patch(Polygon(
            hull_pts,
            closed=True,
            facecolor=color,
            edgecolor=color,
            alpha=alpha_fill,
            linewidth=1.8,
            linestyle='--',
            zorder=1,
        ))
        ax.plot(
            np.append(hull_pts[:, 0], hull_pts[0, 0]),
            np.append(hull_pts[:, 1], hull_pts[0, 1]),
            color=color,
            linewidth=1.8,
            alpha=alpha_edge,
            zorder=2,
        )
    except Exception:
        pass  # degenerate (e.g. collinear) clusters have no hull


def select_one_per_k(df, metric='Silhouette'):
    """Keep the best row per k. A no-op for algorithms with one row per k;
    for DBSCAN it picks the best (eps, min_samples) for each cluster count."""
    df = df.reset_index(drop=True)
    best = df.groupby('k')[metric].idxmax()
    return df.loc[best].sort_values('k').reset_index(drop=True)


def axis_label(name, explained=None):
    """'PC1' or 'PC1 (39.7%)' when explained-variance info is available."""
    if explained is not None and name in explained:
        return f'{name} ({explained[name]:.1%})'
    return name


def _set_window_title(fig, title):
    try:
        fig.canvas.manager.set_window_title(title)
    except AttributeError:
        pass  # non-interactive backends (Agg, servers) have no window manager


def _panel_title(row):
    title = f"k = {row['k']}\nSil: {row['Silhouette']} | ARI: {row['ARI']}"
    if 'Eps' in row and pd.notna(row['Eps']):
        title += (
            f"\neps={row['Eps']}, min_samples={int(row['Min_Samples'])}"
            f" | noise={row['Noise_Fraction']:.0%}"
        )
    return title


def plot_cluster_panels(
        rows,
        pca_base,
        suptitle,
        window_title,
        save_path=None,
        draw_boundaries=True,
        show=True,
        explained=None,
):
    """One panel per row of `rows`, drawn on the PC1/PC2 coordinates."""
    rows = rows.reset_index(drop=True)
    n_plots = len(rows)

    fig, axes = plt.subplots(
        1, n_plots, figsize=(5 * n_plots, 5.5),
        sharex=True, sharey=True, squeeze=False,
    )
    axes = axes[0]
    _set_window_title(fig, window_title)

    for idx, row in rows.iterrows():
        ax = axes[idx]
        labels = parse_labels(row['Labels'])
        if len(labels) != len(pca_base):
            raise ValueError(
                f'{len(labels)} labels but {len(pca_base)} PCA points. '
                'Re-run python main.py so all outputs come from the same data.'
            )

        palette = build_palette(labels)
        plot_df = pca_base.copy()
        plot_df['Cluster'] = pd.Categorical(labels)

        if draw_boundaries:
            for cluster_id, color in palette.items():
                if cluster_id == -1:
                    continue  # no outline around noise
                pts = plot_df.loc[plot_df['Cluster'] == cluster_id, ['PC1', 'PC2']].values
                draw_cluster_boundary(ax, pts, color)

        sns.scatterplot(
            data=plot_df, x='PC1', y='PC2', hue='Cluster', palette=palette,
            alpha=0.85, s=45, ax=ax, legend='full', zorder=3,
        )

        ax.set_title(_panel_title(row), fontsize=11, fontweight='bold')
        ax.set_xlabel(axis_label('PC1', explained))
        ax.set_ylabel(axis_label('PC2', explained) if idx == 0 else '')

    plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.04)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved figure to: {save_path}')

    if show:
        plt.show()
    plt.close(fig)