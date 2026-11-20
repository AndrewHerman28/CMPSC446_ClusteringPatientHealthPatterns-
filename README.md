# Clustering Patient Health Patterns

Unsupervised clustering of maternal health vital signs. The project compares five clustering algorithms across several feature subsets and PCA projections, then checks how well the discovered clusters line up with the dataset's known risk levels.

## Overview

The question: **do natural groupings in patients' vital signs correspond to clinical risk levels?**

The risk labels are never used to build the clusters. They are only used afterwards to evaluate them. The project:

- preprocesses and standardizes the data,
- clusters it with KMeans, Agglomerative, GMM, BIRCH and DBSCAN,
- repeats the analysis on single vital signs, related groups of vital signs and the full feature set,
- repeats it again in 2D and 3D PCA space,
- scores every run with internal and external metrics and saves tables and figures,
- explains the best clusterings with a surrogate model and SHAP, to show which features define each cluster.

## Dataset

The Maternal Health Risk dataset (UCI Machine Learning Repository): 1,014 records, six features and a three-level label.

| Feature | Description |
|---|---|
| `Age` | Age in years |
| `SystolicBP` / `DiastolicBP` | Blood pressure (mmHg) |
| `BS` | Blood sugar level |
| `BodyTemp` | Body temperature (F) |
| `HeartRate` | Resting heart rate (bpm) |
| `RiskLevel` | Label: low risk (406), mid risk (336), high risk (272). Used for evaluation only |

The file is included at `data/raw/Maternal Health Risk Data Set.csv`.

## Methods

### Preprocessing (`src/preprocessing.py`)

1. **Remove exact duplicate rows:** 562 of the 1,014 rows are duplicates, leaving 452.
2. **Remove implausible heart rates:** rows with `HeartRate <= 40` are dropped (one row), leaving 451.
3. **Standardize** all six features with `StandardScaler`.

The cleaning lives in a single function, `clean_data()`, so the clustering and the evaluation always use exactly the same rows.

### Feature subsets

| Subset | Features |
|---|---|
| `univariate_bp`, `univariate_bs`, `univariate_hr` | `SystolicBP`, `BS`, `HeartRate` alone |
| `cardiovascular` | `SystolicBP`, `DiastolicBP`, `HeartRate` |
| `metabolic` | `BS`, `Age` |
| `full_multivariate` | All six features |

### Algorithms

| Algorithm | Notes |
|---|---|
| KMeans (k-means++) | k = 2 to 6 |
| Agglomerative | Ward linkage, k = 2 to 6 |
| Gaussian Mixture (GMM) | Full covariance, k = 2 to 6 |
| BIRCH | k = 2 to 6. If the default threshold finds fewer than k subclusters, the threshold is halved until exactly k clusters are produced |
| DBSCAN | Finds its own number of clusters, so it runs over an `eps` and `min_samples` grid instead of a k sweep. Points labeled noise (-1) are kept in the results |

The same algorithms are also run on 2D and 3D PCA projections of the full standardized data.

### Evaluation

| Metric | What it measures |
|---|---|
| Silhouette, Davies-Bouldin, Calinski-Harabasz | Internal cluster quality (no labels). Only compare these between runs on the same feature set |
| Adjusted Rand Index (ARI), Normalized Mutual Information (NMI) | Agreement with the true risk levels, adjusted for chance and comparable across k |
| Accuracy | Strict one-to-one matching of clusters to risk levels (Hungarian algorithm). Extra clusters and DBSCAN noise points count as errors, so it cannot rise just because k grows |
| Purity | Each cluster credited with its most common class. Saved for reference only. It always increases with k, so don't use it to compare different k |
| Majority-class baseline | 0.517, the accuracy of always guessing "low risk" |

Results are ranked by ARI, then NMI.

### Explainability (`src/explainability/explain_clusters.py`)

Clustering algorithms don't say *why* patients ended up together, so the top three full-multivariate runs (by ARI, then NMI; DBSCAN excluded) are explained with a **surrogate model**: a Random Forest trained to predict each patient's cluster from the original, unscaled features.

| Output | What it shows |
|---|---|
| Cluster profiles | Feature averages per cluster in real units, and the risk-level mix of each cluster |
| SHAP importance and beeswarm | Which features push a patient into each cluster, and in which direction |
| Typical-patient plots | A local explanation for the patient closest to each cluster's center |
| Decision-tree rules | A depth-3 tree that reads as simple if/else rules |
| Clusters vs risk level | The same analysis on the true `RiskLevel`, to see whether clusters and risk depend on the same features |
| `surrogate_fidelity.csv` | Cross-validated accuracy of every surrogate |

The surrogate is only a trustworthy explanation if it predicts the clusters well, so its cross-validated accuracy is saved next to the results. If `shap` is not installed, the step falls back to permutation importance and skips the SHAP-only plots. SHAP explains the clustering, not the patients' health. Because `SystolicBP` and `DiastolicBP` are strongly correlated, SHAP may split credit between them.

## Predicting a new patient

`predict.py` classifies new patients with every model, shows each model's confidence, explains each decision, and summarizes them side by side. It needs `python main.py` to have been run first.

```bash
python predict.py --age 35 --systolic 140 --diastolic 90 --bs 13 --temp 98 --hr 80
python predict.py --csv new_patients.csv    # columns: Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate
python predict.py                           # prompts for each value
```

Clustering never saw the risk labels, so a cluster has no risk level of its own. A new patient is placed in a cluster using each algorithm's own rule (KMeans and BIRCH `predict`, GMM membership probabilities, Ward cost for Agglomerative, nearest core point within `eps` for DBSCAN). The cluster is then converted to a risk level using the risk mix of the training patients in that cluster. Each model is its best full-multivariate run by ARI.

- **Confidence** is the largest risk probability. For KMeans, BIRCH, Agglomerative and DBSCAN it is how risk-pure the patient's cluster is, and for GMM it also reflects membership uncertainty. It is not a calibrated probability.
- **Explanation** for each model comes from a surrogate Random Forest that predicts that model's cluster, explained locally for this patient (SHAP if installed, otherwise replacing one feature at a time with its average).
- **Consensus** averages the probabilities of the clustering models. A DBSCAN outlier (no cluster within `eps`) gets no prediction and is left out of the consensus.
- A supervised Random Forest trained on the true risk labels is shown as a reference, since it outputs real class probabilities.
- Values outside the range seen in training are flagged in the output.

Outputs: `data/processed/predictions/` (`model_predictions.csv`, `patient_summary.csv`, `prediction_explanations.csv`) and `figures/predictions/patient_N_summary.png` and `patient_N_explanations.png`. This is a course project, not a clinical tool.

## Web interface

```bash
pip install flask          # already in requirements.txt
python3 app.py             # opens http://127.0.0.1:5050
```

Enter a patient's six values and press **Run all models**. The **Report** tab then shows the consensus risk and how many models agree, the entered values next to the training ranges, a table of every model's prediction, confidence and top reasons, the summary and explanation charts, and a per-model breakdown of the features behind each decision. You can download the table as a CSV or print the report to PDF. Run `python3 main.py` once first so the results exist. The models are fitted at start-up (about 10 seconds) and everything runs locally.

## Getting started

Requires Python 3 (developed with 3.13).

```bash
git clone https://github.com/AndrewHerman28/CMPSC446_ClusteringPatientHealthPatterns-.git
cd CMPSC446_ClusteringPatientHealthPatterns-
pip install -r requirements.txt
python main.py
```

`main.py` runs the whole pipeline in order and takes under a minute:

1. Check the raw data
2. Preprocess, cluster and run PCA
3. Compute accuracy and purity
4. Save per-algorithm cluster figures
5. Save PCA cluster figures (2D and 3D)
6. Save the accuracy comparison figure
7. Explain the best clusterings (surrogate model and SHAP)

Options:

```bash
python main.py --show           # also display each figure in a window
python main.py --skip-figures   # stop after the CSV results and accuracy report (no figures, no explainability)
```

Run it from any location. All paths are resolved relative to the project folder.

## Project structure

```
main.py                      # runs the full pipeline
predict.py                   # classify new patients with every model (command line)
app.py                       # the same, as a local web interface
requirements.txt
data/
  raw/                       # original dataset
  processed/                 # generated results (CSV)
    individual_models/       # one results file per algorithm
    master_results.csv       # all algorithms in one table
    pca_results.csv          # clustering results in PCA space
    pca_base_coords.csv      # PC1-PC3 coordinates used for plots
    pca_explained_variance.csv
    y_true.csv               # true risk levels, row-aligned with every label list
    explainability/          # cluster profiles, importances, tree rules, fidelity scores
figures/                     # generated figures
  explainability/            # one folder per explained run (SHAP, profiles, tree)
src/
  paths.py                   # all file locations
  preprocessing.py           # cleaning, scaling, PCA coordinates
  run_and_save_results.py    # runs every algorithm and saves results
  PCA.py                     # clustering in 2D / 3D PCA space
  algorithms/                # kmeans, agglomerative, gmm, birch, dbscan
  evaluation/
    clustering_accuracy.py   # accuracy, purity, majority baseline
  ui/                        # web interface (Flask app + page template)
  prediction/
    predict_patient.py       # new-patient prediction, confidence, explanations
  explainability/
    explain_clusters.py      # surrogate model, SHAP, cluster profiles, rules
  visualization/             # cluster, PCA (2D/3D) and accuracy figures
```

## Outputs

- **`data/processed/master_results.csv`** has one row per run (algorithm, feature subset, k) with all metrics, plus the cluster label of every patient.
- **`figures/clusters/algorithm_clusters/`** has the full-multivariate clusters for each algorithm, for k = 2 to 6, projected onto PC1 and PC2.
- **`figures/clusters/pca_clusters/2D` and `3D/`** have the clusters found in PCA space for each algorithm.
- **`figures/multivariate_accuracy_comparison.png`** compares accuracy across algorithms and k, with the majority-class baseline.

Cluster labels are stored as JSON lists, in the same row order as `y_true.csv`.

- **`figures/explainability/<algorithm>_k<k>/`** has the explanation figures for each of the top three runs. Tables are in `data/processed/explainability/`.

![Accuracy comparison](figures/multivariate_accuracy_comparison.png)

![GMM clusters in 3D PCA space](figures/clusters/pca_clusters/3D/gmm_clusters.png)

## Key findings

On the full feature set (451 patients):

- **Clusters match risk levels only weakly.** The best ARI is 0.30 (BIRCH, k = 3), followed by GMM (0.29 at k = 4, 0.28 at k = 3) and DBSCAN (0.29 at k = 8, with 23% of points labeled noise).
- **k = 3 is the sweet spot.** Strict accuracy peaks at three clusters for most algorithms, matching the three risk levels. It then generally falls as k grows, which is expected because extra clusters can't be matched to a class.
- **Accuracy is only modestly above guessing.** The best accuracy is 0.65 (BIRCH, k = 3), against a majority-class baseline of 0.52.
- **PCA is informative but lossy.** PC1 to PC3 explain 39.7%, 19.1% and 14.7% of the variance (59% for two components, 73.5% for three).

## Notes and limitations

- **Silhouette is not comparable across feature sets.** Single, discrete features such as heart rate score very high on Silhouette (about 0.85) despite barely relating to risk level (ARI close to 0.02), which is why runs are ranked by ARI.
- **Deduplication is an assumption.** More than half the rows were exact duplicates, and with no patient ID there is no way to prove they are artifacts and not real patients with identical readings. In an exploratory check outside the pipeline, results without deduplication gave a lower ARI for the best configurations (BIRCH at k = 3 dropped from 0.30 to 0.15), so treat the exact scores with caution. The general pattern, weak agreement with risk level, held either way.
- **Features are coarse.** `BodyTemp` is 98.0 for most patients, so the few higher values weigh heavily after scaling. `SystolicBP` and `DiastolicBP` are strongly correlated (0.79).
- **DBSCAN needs settings tuned per feature space.** Its cluster count depends on `eps` and `min_samples`, so for each k the plots and the accuracy comparison show its best run by ARI. Only runs with 2 to 6 clusters appear in the figures.
- **Results are regenerated on every run.** Running a single step on its own (for example `run_and_save_results.py`) clears the accuracy columns, so use `main.py`.
