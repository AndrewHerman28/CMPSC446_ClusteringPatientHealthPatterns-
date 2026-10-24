import re
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def compute_hungarian_accuracy(y_true, cluster_labels):
    """Computes Clustering Accuracy (ACC) using optimal bipartite matching.

    Handles rectangular cost matrices (k != number of true classes) via majority-vote fallback.
    """
    y_true_arr = np.asarray(y_true)
    labels_arr = np.asarray(cluster_labels)

    # Build contingency matrix (true classes x cluster IDs)
    contingency = pd.crosstab(y_true_arr, labels_arr)
    true_classes = contingency.index.tolist()
    cluster_ids = contingency.columns.tolist()

    cost_matrix = -contingency.values
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Initial optimal 1-to-1 assignments
    mapping = {cluster_ids[c]: true_classes[r] for r, c in zip(row_ind, col_ind)}

    # Fallback for remaining unmapped clusters if k > num_classes
    for c_idx, c_id in enumerate(cluster_ids):
        if c_id not in mapping:
            best_class_idx = np.argmax(contingency.values[:, c_idx])
            mapping[c_id] = true_classes[best_class_idx]

    # Map raw cluster IDs to predicted risk strings
    mapped_preds = np.array([mapping[c] for c in labels_arr])
    acc = accuracy_score(y_true_arr, mapped_preds)

    return round(acc, 4), mapping


def add_accuracy_to_results(df_results, y_true):
    """Iterates through each saved run in a results dataframe and computes Clustering Accuracy."""
    df = df_results.copy()
    acc_scores = []

    for _, row in df.iterrows():
        raw = row['Labels']
        if isinstance(raw, str):
            labels = np.array(re.findall(r'-?\d+', raw), dtype=int)
        else:
            labels = np.asarray(raw, dtype=int)

        acc, _ = compute_hungarian_accuracy(y_true, labels)
        acc_scores.append(acc)

    df['Accuracy'] = acc_scores
    return df


if __name__ == '__main__':
    # 1. Load ground truth
    raw_df = pd.read_csv('../../data/raw/Maternal Health Risk Data Set.csv')
    # Drop duplicates and handle anomalies to match your scaled feature indexing
    raw_df = raw_df.drop_duplicates().reset_index(drop=True)
    raw_df = raw_df[raw_df['HeartRate'] > 10].reset_index(drop=True)
    y_true = raw_df['RiskLevel'].values

    # 2. Paths to all result files
    files = {
        'KMeans': '../../data/processed/individual_models/kmeans_results.csv',
        'Agglomerative': '../../data/processed/individual_models/agglomerative_results.csv',
        'GMM': '../../data/processed/individual_models/gmm_results.csv',
        'BIRCH': '../../data/processed/individual_models/birch_results.csv',
    }

    evaluated_dfs = {}

    for algo, path in files.items():
        res_df = pd.read_csv(path)
        res_df = add_accuracy_to_results(res_df, y_true)

        # Save back updated dataframe with Accuracy column
        res_df.to_csv(path, index=False)
        evaluated_dfs[algo] = res_df
        print(f'Computed Accuracy for {algo} across all {len(res_df)} runs.')

    # 3. Concatenate and display the Top 10 Configurations by Real Accuracy
    all_evals = pd.concat(evaluated_dfs.values(), ignore_index=True)

    top_acc = all_evals.sort_values(by='Accuracy', ascending=False)[
        [
            'Algorithm',
            'Subset',
            'k',
            'Accuracy',
            'ARI',
            'NMI',
            'Silhouette',
            'Davies_Bouldin',
        ]
    ]

    print('\n=== Top 10 Clustering Runs by True Real Accuracy ===')
    print(top_acc.head(10).to_string(index=False))

    plt.figure(figsize=(9, 5))
    sns.set_theme(style='whitegrid')

    # Filter to full multivariate set for primary comparison
    multi_evals = all_evals[all_evals['Subset'] == 'full_multivariate']

    sns.lineplot(
        data=multi_evals,
        x='k',
        y='Accuracy',
        hue='Algorithm',
        marker='o',
        linewidth=2.2,
        palette='tab10',
    )

    plt.title(
        'Clustering Accuracy vs. Number of Clusters (Full Multivariate)',
        fontsize=13,
        fontweight='bold',
    )
    plt.xlabel('Number of Clusters (k)', fontsize=11)
    plt.ylabel('Clustering Accuracy (ACC)', fontsize=11)
    plt.xticks([2, 3, 4, 5, 6])
    plt.ylim(0.3, 0.8)
    plt.legend(frameon=True, title='Algorithm')
    plt.tight_layout()
    # plt.savefig('figures/multivariate_accuracy_comparison.png', dpi=300)
    plt.show()
