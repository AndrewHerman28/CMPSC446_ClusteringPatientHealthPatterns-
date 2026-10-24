import pandas as pd
from sklearn.cluster import Birch
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


def run_BIRCH(subsets, X, y):
    birch_results = []
    k_values = [2, 3, 4, 5, 6]

    for subset_name, cols in subsets.items():
        X_sub = X[cols].values

        for k in k_values:
            birch = Birch(n_clusters=k, threshold=0.5, branching_factor=50)
            labels = birch.fit_predict(X_sub)

            n_unique_labels = len(set(labels))
            if n_unique_labels < 2:
                continue

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
    birch_df = birch_df.sort_values(by='ARI', ascending=False)
    return birch_df
