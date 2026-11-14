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


def run_DBSCAN(subsets, X, y):
    dbscan_results = []
    eps_values = [0.3, 0.5, 0.75, 1.0, 1.5]
    min_samples_values = [5, 10]

    for subset_name, cols in subsets.items():
        X_sub = X[cols].values

        for eps in eps_values:
            for min_samples in min_samples_values:
                db = DBSCAN(eps=eps, min_samples=min_samples)
                labels = db.fit_predict(X_sub)

                mask = labels != -1
                n_clusters = len(set(labels[mask]))
                if n_clusters < 2:
                    continue  # all noise or a single cluster: metrics undefined

                sil = silhouette_score(X_sub[mask], labels[mask])
                dbi = davies_bouldin_score(X_sub[mask], labels[mask])
                chi = calinski_harabasz_score(X_sub[mask], labels[mask])

                ari = adjusted_rand_score(y, labels)
                nmi = normalized_mutual_info_score(y, labels)

                dbscan_results.append({
                    'Algorithm': "DBSCAN",
                    'Subset': subset_name,
                    'Num_Features': len(cols),
                    'k': n_clusters,
                    'Eps': eps,
                    'Min_Samples': min_samples,
                    'Noise_Fraction': round(float(np.mean(~mask)), 4),
                    'Silhouette': round(sil, 4),
                    'Davies_Bouldin': round(dbi, 4),
                    'Calinski_Harabasz': round(chi, 4),
                    'ARI': round(ari, 4),
                    'NMI': round(nmi, 4),
                    'Labels': labels,
                })

    dbscan_df = pd.DataFrame(dbscan_results)
    if not dbscan_df.empty:
        dbscan_df = dbscan_df.sort_values(by=['ARI', 'NMI'], ascending=False)
    return dbscan_df
