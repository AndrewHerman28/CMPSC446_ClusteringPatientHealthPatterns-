import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


def run_Agglomerative(subsets, X, y):
    agg_results = []
    k_values = [2, 3, 4, 5, 6]

    for subset_name, cols in subsets.items():
        X_sub = X[cols].values

        for k in k_values:
            agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
            labels = agg.fit_predict(X_sub)

            sil = silhouette_score(X_sub, labels)
            dbi = davies_bouldin_score(X_sub, labels)
            chi = calinski_harabasz_score(X_sub, labels)

            ari = adjusted_rand_score(y, labels)
            nmi = normalized_mutual_info_score(y, labels)

            agg_results.append({
                'Algorithm': "Agglomerative",
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

    agg_df = pd.DataFrame(agg_results)
    agg_df = agg_df.sort_values(by='Silhouette', ascending=False)
    return agg_df
