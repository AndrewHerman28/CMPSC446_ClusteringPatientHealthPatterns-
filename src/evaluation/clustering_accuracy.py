import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from src.paths import MASTER_CSV, MODELS_DIR, RAW_CSV, Y_TRUE_CSV
from src.preprocessing import clean_data

NOISE_LABEL = -1  # DBSCAN marks noise points with -1

# Result files evaluated by this step (missing files are skipped)
ALGORITHM_FILES = {
    'KMeans': MODELS_DIR / 'kmeans_results.csv',
    'Agglomerative': MODELS_DIR / 'agglomerative_results.csv',
    'GMM': MODELS_DIR / 'gmm_results.csv',
    'BIRCH': MODELS_DIR / 'birch_results.csv',
    'DBSCAN': MODELS_DIR / 'dbscan_results.csv',
}


def parse_labels(raw):
    """Labels are stored as a JSON list string in the CSVs (or already an array)."""
    if isinstance(raw, str):
        return np.array(re.findall(r'-?\d+', raw), dtype=int)
    return np.asarray(raw, dtype=int)


def load_ground_truth():
    """True risk levels, row-aligned with the saved cluster labels."""
    if Y_TRUE_CSV.exists():
        return pd.read_csv(Y_TRUE_CSV)['RiskLevel'].values
    return clean_data(pd.read_csv(RAW_CSV))['RiskLevel'].values


def _contingency_without_noise(y_true_arr, labels_arr):
    contingency = pd.crosstab(y_true_arr, labels_arr)
    if NOISE_LABEL in contingency.columns:
        contingency = contingency.drop(columns=NOISE_LABEL)
    return contingency


def compute_hungarian_accuracy(y_true, cluster_labels):
    """Clustering Accuracy (ACC) using strict one-to-one optimal matching.

    Each cluster is matched to at most one true class (Hungarian algorithm) and
    each class to at most one cluster. Clusters left over when k > number of
    classes stay unmatched, and their points count as errors. DBSCAN noise
    points (label -1) are never matched and also count as errors. This keeps
    the score comparable across different k (it cannot rise just because k grows).
    """
    y_true_arr = np.asarray(y_true)
    labels_arr = np.asarray(cluster_labels)

    contingency = _contingency_without_noise(y_true_arr, labels_arr)
    if contingency.shape[1] == 0:
        return 0.0, {}
    true_classes = contingency.index.tolist()
    cluster_ids = contingency.columns.tolist()

    # Optimal one-to-one assignment (handles rectangular matrices)
    row_ind, col_ind = linear_sum_assignment(-contingency.values)
    mapping = {cluster_ids[c]: true_classes[r] for r, c in zip(row_ind, col_ind)}

    # Unmatched clusters and noise map to None, so they are always counted as wrong
    mapped_preds = np.array([mapping.get(c) for c in labels_arr], dtype=object)
    acc = float(np.mean(mapped_preds == y_true_arr))

    return round(acc, 4), mapping


def compute_purity(y_true, cluster_labels):
    """Cluster purity: each cluster is credited with its most common class.

    Unlike accuracy, purity can only go up as k grows (k = n gives 1.0), so it
    should NOT be used to compare different values of k. Noise points count as
    errors.
    """
    y_true_arr = np.asarray(y_true)
    contingency = _contingency_without_noise(y_true_arr, np.asarray(cluster_labels))
    if contingency.shape[1] == 0:
        return 0.0
    return round(float(contingency.max(axis=0).sum() / len(y_true_arr)), 4)


def add_accuracy_to_results(df_results, y_true):
    """Computes Accuracy and Purity for every saved run in a results dataframe."""
    df = df_results.copy()
    acc_scores = []
    purity_scores = []

    for _, row in df.iterrows():
        labels = parse_labels(row['Labels'])
        if len(labels) != len(y_true):
            raise ValueError(
                f'{len(labels)} labels but {len(y_true)} ground-truth rows. '
                'Re-run python main.py so all outputs come from the same data.'
            )

        acc, _ = compute_hungarian_accuracy(y_true, labels)
        acc_scores.append(acc)
        purity_scores.append(compute_purity(y_true, labels))

    df['Accuracy'] = acc_scores
    df['Purity'] = purity_scores
    return df


def main():
    y_true = load_ground_truth()

    # Accuracy of always guessing the most common class (reference point)
    baseline = pd.Series(y_true).value_counts(normalize=True).max()
    print(f'Majority-class baseline accuracy: {baseline:.4f}')

    evaluated_dfs = []
    for algo, path in ALGORITHM_FILES.items():
        if not path.exists():
            print(f'Skipping {algo}: {path.name} not found.')
            continue
        res_df = add_accuracy_to_results(pd.read_csv(path), y_true)
        res_df.to_csv(path, index=False)  # save back with Accuracy/Purity columns
        evaluated_dfs.append(res_df)
        print(f'Computed Accuracy for {algo} across all {len(res_df)} runs.')

    # Refresh the master table so it includes Accuracy and Purity
    all_evals = pd.concat(evaluated_dfs, ignore_index=True)
    all_evals.drop(columns=['BIC'], errors='ignore').to_csv(MASTER_CSV, index=False)

    top_acc = all_evals.sort_values(by='Accuracy', ascending=False)[
        ['Algorithm', 'Subset', 'k', 'Accuracy', 'Purity', 'ARI', 'NMI', 'Silhouette', 'Davies_Bouldin']
    ]
    print('\n=== Top 10 Clustering Runs by Accuracy ===')
    print(top_acc.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
