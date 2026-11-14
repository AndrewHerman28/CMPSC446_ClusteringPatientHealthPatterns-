import json

import pandas as pd

from src.algorithms.agglomerative import run_Agglomerative
from src.algorithms.birch import run_BIRCH
from src.algorithms.dbscan import run_DBSCAN
from src.algorithms.gmm import run_GMM
from src.algorithms.kmeans import run_Kmeans
from src.paths import (
    MASTER_CSV, MODELS_DIR, PCA_RESULTS_CSV, RAW_CSV, Y_TRUE_CSV,
)
from src.PCA import apply_PCA
from src.preprocessing import preprocess, save_pca_base_coords


def save_results(df, path):
    """Save a results dataframe, storing Labels as a JSON list.

    (Saving raw numpy arrays would be truncated with '...' for >1000 rows.)
    """
    out = df.copy()
    out['Labels'] = out['Labels'].apply(lambda a: json.dumps([int(i) for i in a]))
    out.to_csv(path, index=False)


def main():
    df = pd.read_csv(RAW_CSV)
    X, y = preprocess(df)
    save_pca_base_coords(X)  # PC1-PC3 coordinates used by the cluster plots
    y.to_csv(Y_TRUE_CSV, index=False)  # ground truth, row-aligned with every Labels list

    # Feature subsets for comparing univariate vs. multivariate clustering
    subsets = {
        # UNIVARIATE BASELINES (Single vital signs alone)
        'univariate_bp': ['SystolicBP'],
        'univariate_bs': ['BS'],
        'univariate_hr': ['HeartRate'],

        # BIVARIATE / DOMAIN SUBSETS (Related body systems)
        'cardiovascular': ['SystolicBP', 'DiastolicBP', 'HeartRate'],
        'metabolic': ['BS', 'Age'],

        # Everything together
        'full_multivariate': list(X.columns)
    }

    # Run each clustering algorithm
    Kmeans_results = run_Kmeans(subsets, X, y)
    Agglomerative_results = run_Agglomerative(subsets, X, y)
    GMM_results = run_GMM(subsets, X, y)
    BIRCH_results = run_BIRCH(subsets, X, y)
    DBSCAN_results = run_DBSCAN(subsets, X, y)

    PCA_results = apply_PCA(X, y)  # Test PCA with each algorithm as well

    # Save individual clustering algorithm benchmark results to CSV
    save_results(Kmeans_results, MODELS_DIR / 'kmeans_results.csv')
    save_results(Agglomerative_results, MODELS_DIR / 'agglomerative_results.csv')
    save_results(GMM_results, MODELS_DIR / 'gmm_results.csv')
    save_results(BIRCH_results, MODELS_DIR / 'birch_results.csv')
    if DBSCAN_results.empty:
        print('[DBSCAN] No setting produced 2+ clusters; dbscan_results.csv not written.')
    else:
        save_results(DBSCAN_results, MODELS_DIR / 'dbscan_results.csv')
    save_results(PCA_results, PCA_RESULTS_CSV)

    # Master table (the accuracy step later rewrites it with Accuracy/Purity added)
    gmm_clean = GMM_results.drop(columns=['BIC'], errors='ignore')
    all_results = pd.concat(
        [Kmeans_results, Agglomerative_results, BIRCH_results, gmm_clean, DBSCAN_results],
        ignore_index=True,
    )
    save_results(all_results, MASTER_CSV)


if __name__ == "__main__":
    main()
