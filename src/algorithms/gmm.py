import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


def run_GMM(subsets, X, y):
    gmm_results = []
    k_values = [2, 3, 4, 5, 6]

    for subset_name, cols in subsets.items():
      X_sub = X[cols].values

      for k in k_values:
        gmm = GaussianMixture(
            n_components=k, covariance_type='full', random_state=42
        )
        labels = gmm.fit_predict(X_sub)

        sil = silhouette_score(X_sub, labels)
        dbi = davies_bouldin_score(X_sub, labels)
        chi = calinski_harabasz_score(X_sub, labels)

        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        bic = gmm.bic(X_sub)

        gmm_results.append({
            'Algorithm': "GMM",
            'Subset': subset_name,
            'Num_Features': len(cols),
            'k': k,
            'Silhouette': round(sil, 4),
            'Davies_Bouldin': round(dbi, 4),
            'Calinski_Harabasz': round(chi, 4),
            'ARI': round(ari, 4),
            'NMI': round(nmi, 4),
            'BIC': round(bic, 2),
            'Labels': labels,
        })

    gmm_df = pd.DataFrame(gmm_results)
    gmm_df = gmm_df.sort_values(by='Silhouette', ascending=False)
    return gmm_df