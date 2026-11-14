from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_CSV = DATA / "raw" / "Maternal Health Risk Data Set.csv"
PROCESSED = DATA / "processed"
MODELS_DIR = PROCESSED / "individual_models"
PCA_BASE_CSV = PROCESSED / "pca_base_coords.csv"
Y_TRUE_CSV = PROCESSED / "y_true.csv"
PCA_RESULTS_CSV = PROCESSED / "pca_results.csv"
MASTER_CSV = PROCESSED / "master_results.csv"
PCA_VARIANCE_CSV = PROCESSED / "pca_explained_variance.csv"
FIGURES_ROOT = ROOT / "figures"
FIGURES = FIGURES_ROOT / "clusters"
ACCURACY_FIG = FIGURES_ROOT / "multivariate_accuracy_comparison.png"

for _d in (PROCESSED, MODELS_DIR, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)
