from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.evaluation.clustering_accuracy import ALGORITHM_FILES, load_ground_truth
from src.paths import ACCURACY_FIG
from src.visualization.plot_utils import select_one_per_k

K_RANGE = (2, 6)


def load_multivariate_results():
    """Full-multivariate runs for every algorithm, one row per k.

    For DBSCAN (several settings per cluster count) the best run by ARI is kept.
    """
    frames = []
    for algo, path in ALGORITHM_FILES.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if 'Accuracy' not in df.columns:
            raise ValueError(
                f'{path.name} has no Accuracy column. Run the accuracy report '
                '(src/evaluation/clustering_accuracy.py) first.'
            )
        df = df[(df['Subset'] == 'full_multivariate') & df['k'].between(*K_RANGE)]
        if not df.empty:
            frames.append(select_one_per_k(df, metric='ARI'))
    return pd.concat(frames, ignore_index=True)


def plot_accuracy(multi_evals=None, baseline=None, save_path=ACCURACY_FIG, show=True):
    """Line plot of clustering accuracy vs. k for the full multivariate subset.

    With no arguments it loads everything it needs from the saved CSVs.
    baseline: accuracy of always predicting the majority class (dashed line).
    """
    if multi_evals is None:
        multi_evals = load_multivariate_results()
        if baseline is None:
            baseline = pd.Series(load_ground_truth()).value_counts(normalize=True).max()

    fig = plt.figure(figsize=(9, 5))
    sns.set_theme(style='whitegrid')

    sns.lineplot(
        data=multi_evals,
        x='k',
        y='Accuracy',
        hue='Algorithm',
        marker='o',
        linewidth=2.2,
        palette='tab10',
    )

    if baseline is not None:
        plt.axhline(baseline, color='gray', linestyle='--', linewidth=1.5,
                    label=f'Majority-class baseline ({baseline:.2f})')

    plt.title(
        'Clustering Accuracy vs. Number of Clusters (Full Multivariate)',
        fontsize=13,
        fontweight='bold',
    )
    plt.xlabel('Number of Clusters (k)', fontsize=11)
    plt.ylabel('Clustering Accuracy (ACC)', fontsize=11)
    plt.xticks(range(K_RANGE[0], K_RANGE[1] + 1))
    plt.ylim(min(0.3, multi_evals['Accuracy'].min() - 0.02), 0.8)
    plt.legend(frameon=True, title='Algorithm', loc='center left', bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Saved figure to: {save_path}')
    if show:
        plt.show()
    plt.close(fig)
