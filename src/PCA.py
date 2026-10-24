import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, Birch
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


def apply_PCA(X, y):
    # 1. Fit PCA across all 6 scaled features
    pca_full = PCA().fit(X)
    explained_var_ratio = pca_full.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var_ratio)

    print(
        'Explained Variance by Component:',
        [round(val, 4) for val in explained_var_ratio],
    )
    print(
        'Cumulative Explained Variance:', [round(val, 4) for val in cumulative_var]
    )

    # 2. Test 2-component and 3-component PCA projections
    pca_dimensions = {
        'PCA_2D': PCA(n_components=2, random_state=42).fit_transform(X),
        'PCA_3D': PCA(n_components=3, random_state=42).fit_transform(X),
    }

    pca_results = []
    k_values = [2, 3, 4, 5, 6]

    for dim_name, X_pca in pca_dimensions.items():
      n_dims = X_pca.shape[1]

      for k in k_values:
        # Model A: k-Means++
        km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
        km_labels = km.fit_predict(X_pca)

        # Model B: Agglomerative (Ward linkage)
        agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
        agg_labels = agg.fit_predict(X_pca)

        # Model C: GMM (Full covariance)
        gmm = GaussianMixture(
            n_components=k, covariance_type='full', random_state=42
        )
        gmm_labels = gmm.fit_predict(X_pca)

        # Model D: Birch (Heirachal)
        birch = Birch(n_clusters=k, threshold=0.5, branching_factor=50)
        birch_labels = birch.fit_predict(X_pca)

        model_runs = [
            ('k-Means++', km_labels),
            ('Agglomerative', agg_labels),
            ('GMM', gmm_labels),
            ('BIRCH', birch_labels),
        ]

        for model_name, labels in model_runs:
          sil = silhouette_score(X_pca, labels)
          dbi = davies_bouldin_score(X_pca, labels)
          chi = calinski_harabasz_score(X_pca, labels)
          ari = adjusted_rand_score(y, labels)
          nmi = normalized_mutual_info_score(y, labels)

          pca_results.append({
              'Dimensionality': dim_name,
              'Num_Components': n_dims,
              'Algorithm': f"PCA_{model_name}",
              'k': k,
              'Silhouette': round(sil, 4),
              'Davies_Bouldin': round(dbi, 4),
              'Calinski_Harabasz': round(chi, 4),
              'ARI': round(ari, 4),
              'NMI': round(nmi, 4),
              'Labels': labels,
          })

    pca_df = pd.DataFrame(pca_results)
    pca_df = pca_df.sort_values(by='Silhouette', ascending=False)
    return pca_df

