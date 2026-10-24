import pandas as pd

from src.algorithms.agglomerative import run_Agglomerative
from src.algorithms.birch import run_BIRCH
from src.algorithms.gmm import run_GMM
from src.algorithms.kmeans import run_Kmeans
from src.PCA import apply_PCA
from src.preprocessing import preprocess

if __name__ == "__main__":
    df = pd.read_csv('data/raw/Maternal Health Risk Data Set.csv')

    X, y = preprocess(df)

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

    PCA_results = apply_PCA(X, y)  # Test PCA with each algorithm as well

    # Save individual clustering algorithm benchmark results to CSV
    Kmeans_results.to_csv("data/Algorithm_Results/kmeans_results.csv", index=False)
    Agglomerative_results.to_csv("data/Algorithm_Results/agglomerative_results.csv", index=False)
    GMM_results.to_csv("data/Algorithm_Results/gmm_results.csv", index=False)
    BIRCH_results.to_csv("data/Algorithm_Results/birch_results.csv", index=False)
    PCA_results.to_csv("data/Algorithm_Results/pca_results.csv", index=False)

    # Create master table dataset
    gmm_clean = GMM_results.drop(columns=['BIC'], errors='ignore')
    all_results = pd.concat([Kmeans_results, Agglomerative_results, BIRCH_results, gmm_clean], ignore_index=True)
    all_results.to_csv("data/master_results.csv", index=False)
