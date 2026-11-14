import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


# Default settings grid (tuned for the standardized original features)
DEFAULT_EPS_VALUES = [0.3, 0.5, 0.75, 1.0, 1.5]
DEFAULT_MIN_SAMPLES_VALUES = [5, 10]


def run_dbscan_grid(X_sub, y, eps_values=None, min_samples_values=None):
    """Runs DBSCAN over an eps / min_samples grid on one feature matrix.

    DBSCAN chooses its own number of clusters, so there is no k to sweep.
    Settings that give fewer than 2 clusters (all noise, or one big cluster)
    are skipped because the metrics are undefined. Silhouette, Davies-Bouldin
    and Calinski-Harabasz ignore noise points; ARI and NMI keep them (as one
    extra "cluster", label -1).

    Returns a list of result dicts (one per usable setting).
    """
    if eps_values is None:
        eps_values = DEFAULT_EPS_VALUES
    if min_samples_values is None:
        min_samples_values = DEFAULT_MIN_SAMPLES_VALUES

    rows = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X_sub)

            mask = labels != -1
            n_clusters = len(set(labels[mask]))
            if n_clusters < 2:
                continue  # all noise or a single cluster: metrics undefined

            rows.append({
                'k': n_clusters,
                'Eps': eps,
                'Min_Samples': min_samples,
                'Noise_Fraction': round(float(np.mean(~mask)), 4),
                'Silhouette': round(silhouette_score(X_sub[mask], labels[mask]), 4),
                'Davies_Bouldin': round(davies_bouldin_score(X_sub[mask], labels[mask]), 4),
                'Calinski_Harabasz': round(calinski_harabasz_score(X_sub[mask], labels[mask]), 4),
                'ARI': round(adjusted_rand_score(y, labels), 4),
                'NMI': round(normalized_mutual_info_score(y, labels), 4),
                'Labels': labels,
            })
    return rows


def run_DBSCAN(subsets, X, y):
    dbscan_results = []

    for subset_name, cols in subsets.items():
        for run in run_dbscan_grid(X[cols].values, y):
            dbscan_results.append({
                'Algorithm': "DBSCAN",
                'Subset': subset_name,
                'Num_Features': len(cols),
                **run,
            })

    dbscan_df = pd.DataFrame(dbscan_results)
    if not dbscan_df.empty:
        dbscan_df = dbscan_df.sort_values(by=['ARI', 'NMI'], ascending=False)
    return dbscan_df