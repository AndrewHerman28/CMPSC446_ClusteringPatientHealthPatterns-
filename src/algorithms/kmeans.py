import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def run_Kmeans(subsets, X, y):
    kmeans_results = []
    k_values = [2, 3, 4, 5, 6]

    for subset_name, cols in subsets.items():
        X_sub = X[cols].values

        for k in k_values:
            km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
            labels = km.fit_predict(X_sub)

            sil = silhouette_score(X_sub, labels)
            dbi = davies_bouldin_score(X_sub, labels)
            chi = calinski_harabasz_score(X_sub, labels)

            ari = adjusted_rand_score(y, labels)
            nmi = normalized_mutual_info_score(y, labels)

            kmeans_results.append({
                'Algorithm': "KMeans",
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

    kmeans_df = pd.DataFrame(kmeans_results)
    kmeans_df = kmeans_df.sort_values(by='Silhouette', ascending=False)
    return kmeans_df
