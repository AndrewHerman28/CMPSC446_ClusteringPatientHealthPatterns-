import warnings

import pandas as pd
from sklearn.cluster import Birch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


def fit_birch_exact_k(X_sub, k, threshold=0.5, branching_factor=50, min_threshold=1e-3):
    """Fit BIRCH and make sure it returns exactly k clusters.

    BIRCH first builds subclusters controlled by `threshold`, then merges them
    down to k. If the threshold is too large it can find fewer than k
    subclusters, and the result silently has fewer than k clusters. In that
    case the threshold is halved until k clusters are produced.

    Returns (labels, threshold_used), or (None, threshold_used) if k clusters
    could not be reached.
    """
    while threshold >= min_threshold:
        birch = Birch(n_clusters=k, threshold=threshold, branching_factor=branching_factor)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            labels = birch.fit_predict(X_sub)
        if len(set(labels)) == k:
            return labels, threshold
        threshold /= 2
    return None, threshold


def run_BIRCH(subsets, X, y):
    birch_results = []
    k_values = [2, 3, 4, 5, 6]

    for subset_name, cols in subsets.items():
        X_sub = X[cols].values

        for k in k_values:
            labels, used_threshold = fit_birch_exact_k(X_sub, k)

            if labels is None:
                print(f'[BIRCH] Skipping {subset_name}, k={k}: could not reach k clusters.')
                continue
            if used_threshold != 0.5:
                print(f'[BIRCH] {subset_name}, k={k}: threshold lowered 0.5 -> {used_threshold} to reach k clusters.')

            sil = silhouette_score(X_sub, labels)
            dbi = davies_bouldin_score(X_sub, labels)
            chi = calinski_harabasz_score(X_sub, labels)

            ari = adjusted_rand_score(y, labels)
            nmi = normalized_mutual_info_score(y, labels)

            birch_results.append({
                'Algorithm': "BIRCH",
                'Subset': subset_name,
                'Num_Features': len(cols),
                'k': k,
                'Silhouette': round(sil, 4),
                'Davies_Bouldin': round(dbi, 4),
                'Calinski_Harabasz': round(chi, 4),
                'ARI': round(ari, 4),
                'NMI': round(nmi, 4),
                'Labels': labels,
            })

    birch_df = pd.DataFrame(birch_results)
    birch_df = birch_df.sort_values(by=['ARI', 'NMI'], ascending=False)
    return birch_df
