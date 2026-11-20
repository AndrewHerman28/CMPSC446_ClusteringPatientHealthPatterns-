"""Explain WHY the best clusterings group patients the way they do.

Clustering algorithms do not tell us what separates the clusters, so each
selected clustering is explained with a *surrogate model*: a classifier that
learns to predict the cluster label from the original (unscaled) features.
If the surrogate predicts the clusters well, what it relies on is a faithful
description of how the clusters were formed.

For each of the top full-multivariate runs (ranked by ARI, then NMI) this saves:
  * cluster profiles      : feature means per cluster (original units) and the
                            risk-level mix of each cluster
  * SHAP importance       : which features drive each cluster (RandomForest +
                            SHAP TreeExplainer); permutation importance is used
                            instead if the `shap` package is not installed
  * SHAP beeswarm         : direction of each feature's effect, per cluster
  * typical patients      : a local explanation for one representative patient
                            per cluster
  * decision-tree rules   : a depth-3 tree readable as if/else rules
  * comparison with risk  : the same analysis for the true RiskLevel, to see
                            whether clusters and risk levels depend on the same
                            features
  * surrogate_fidelity.csv: cross-validated accuracy of every surrogate

Caveats (also in the README): a surrogate explains the clustering, not the
patients' health. High fidelity means the explanation is trustworthy for that
clustering. SystolicBP and DiastolicBP are correlated (about 0.79), so SHAP
may split credit between them.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

from src.evaluation.clustering_accuracy import parse_labels
from src.paths import EXPLAIN_DIR, EXPLAIN_FIGURES, MASTER_CSV, RAW_CSV
from src.preprocessing import clean_data
from src.visualization.plot_utils import build_palette

TOP_N_RUNS = 3
K_RANGE = (2, 6)
RANDOM_STATE = 42
RISK_ORDER = ['low risk', 'mid risk', 'high risk']
RISK_MODEL_NAME = 'RiskLevel'


# --------------------------------------------------------------------------- #
# Data and run selection
# --------------------------------------------------------------------------- #
def load_features():
    """Cleaned features in ORIGINAL units, z-scores, and true risk levels.

    Uses clean_data(), the same function the clustering used, so rows line up
    one-to-one with every saved label list.
    """
    df = clean_data(pd.read_csv(RAW_CSV))
    X = df.drop(columns=['RiskLevel']).reset_index(drop=True)
    y_risk = df['RiskLevel'].reset_index(drop=True)
    X_z = (X - X.mean()) / X.std(ddof=0)  # same scaling as StandardScaler
    return X, X_z, y_risk


def select_runs(master, top_n=TOP_N_RUNS):
    """Top full-multivariate runs by ARI then NMI (DBSCAN excluded: it has
    noise points and a variable number of clusters)."""
    runs = master[
        (master['Subset'] == 'full_multivariate')
        & (master['Algorithm'] != 'DBSCAN')
        & (master['k'].between(*K_RANGE))
        ]
    return runs.sort_values(['ARI', 'NMI'], ascending=False).head(top_n)


def run_name(row):
    return f"{str(row['Algorithm']).lower().replace(' ', '_')}_k{int(row['k'])}"


# --------------------------------------------------------------------------- #
# Surrogate model + SHAP
# --------------------------------------------------------------------------- #
def make_forest():
    return RandomForestClassifier(
        n_estimators=300, class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1,
    )


def cv_fidelity(model, X, labels):
    """Cross-validated accuracy / balanced accuracy of predicting `labels`.

    Returns (nan, nan) if a class is too small to stratify.
    """
    counts = pd.Series(labels).value_counts()
    n_splits = int(min(5, counts.min()))
    if n_splits < 2:
        return np.nan, np.nan
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(model, X, labels, cv=cv)
    return accuracy_score(labels, pred), balanced_accuracy_score(labels, pred)


def normalize_shap(sv, n_samples, n_features):
    """Return SHAP values as (n_classes, n_samples, n_features).

    Older shap versions return a list (one array per class); newer ones return
    an array shaped (n_samples, n_features, n_classes).
    """
    if isinstance(sv, list):
        return np.stack([np.asarray(a) for a in sv])
    arr = np.asarray(sv)
    if arr.ndim == 2:
        return arr[None]
    if arr.shape[0] == n_samples and arr.shape[1] == n_features:
        return np.transpose(arr, (2, 0, 1))
    if arr.shape[1] == n_samples and arr.shape[2] == n_features:
        return arr
    raise ValueError(f'Unexpected SHAP output shape {arr.shape}')


def explain(X, labels, name):
    """Fit a surrogate on (X -> labels) and explain it.

    Returns a dict with the fitted model, class names, CV fidelity, and
    importance per class (n_classes x n_features) plus SHAP values when the
    `shap` package is available.
    """
    model = make_forest().fit(X, labels)
    classes = list(model.classes_)
    acc, bal = cv_fidelity(make_forest(), X, labels)
    out = {'model': model, 'classes': classes, 'cv_acc': acc, 'cv_bal': bal,
           'sv': None, 'method': 'permutation'}

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = normalize_shap(explainer.shap_values(X, check_additivity=False),
                            len(X), X.shape[1])
        out['sv'] = sv
        out['method'] = 'shap'
        out['class_importance'] = np.abs(sv).mean(axis=1)  # (classes, features)
        _check_additivity(explainer, model, X, sv, name)
    except ImportError:
        print(f'  [{name}] `shap` is not installed -> using permutation importance '
              '(pip install shap for the full analysis).')
        res = permutation_importance(model, X, labels, n_repeats=10,
                                     random_state=RANDOM_STATE, n_jobs=-1)
        glob = np.clip(res.importances_mean, 0, None)
        out['class_importance'] = np.tile(glob, (len(classes), 1))

    imp = out['class_importance'].mean(axis=0)
    out['global_importance'] = imp / imp.sum() if imp.sum() > 0 else imp
    return out


def _check_additivity(explainer, model, X, sv, name):
    """Sanity check: base value + sum of SHAP values should equal the model's
    predicted probability. Warn (do not crash) if it does not."""
    try:
        base = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
        proba = model.predict_proba(X)
        recon = sv.sum(axis=2).T + base[None, :]
        err = np.abs(recon - proba).max()
        if err > 1e-2:
            print(f'  [{name}] warning: SHAP additivity check off by {err:.3f}')
    except Exception as exc:  # purely diagnostic
        print(f'  [{name}] (additivity check skipped: {exc})')


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def cluster_profile(X, labels, y_risk):
    """Per cluster: size, feature means (original units), risk-level mix."""
    prof = X.groupby(labels).mean().round(2)
    prof.insert(0, 'Size', pd.Series(labels).value_counts().sort_index().values)
    mix = pd.crosstab(labels, y_risk, normalize='index').round(3)
    mix = mix.reindex(columns=[r for r in RISK_ORDER if r in mix.columns])
    mix.columns = [f'Share_{c.replace(" ", "_")}' for c in mix.columns]
    prof = prof.join(mix)
    prof.index.name = 'Cluster'
    return prof


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _finish(fig, path, show):
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'  Saved {path}')
    if show:
        plt.show()
    plt.close(fig)


def plot_profile_heatmap(X, X_z, labels, path, title, show):
    """Cluster x feature heatmap: colour = z-score, text = original-unit mean."""
    z = X_z.groupby(labels).mean()
    raw = X.groupby(labels).mean()
    sizes = pd.Series(labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(1.3 * z.shape[1] + 2, 0.9 * len(z) + 2))
    im = ax.imshow(z.values, cmap='coolwarm', vmin=-2, vmax=2, aspect='auto')
    ax.set_xticks(range(z.shape[1]), z.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(z)), [f'Cluster {c} (n={sizes[c]})' for c in z.index])
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f'{raw.values[i, j]:.1f}', ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=ax, label='Cluster mean (z-score vs all patients)')
    ax.set_title(title, fontweight='bold')
    _finish(fig, path, show)


def plot_importance_by_cluster(info, features, palette, path, title, show):
    """Grouped bars: mean |SHAP| of each feature, one bar per cluster."""
    imp = info['class_importance']
    order = np.argsort(info['global_importance'])[::-1]
    n_cls = len(info['classes'])
    width = 0.8 / n_cls
    fig, ax = plt.subplots(figsize=(1.4 * len(features) + 2, 4.8))
    for ci, cls in enumerate(info['classes']):
        ax.bar(np.arange(len(features)) + ci * width, imp[ci][order], width,
               color=palette.get(int(cls)), label=f'Cluster {cls}')
    ax.set_xticks(np.arange(len(features)) + width * (n_cls - 1) / 2,
                  [features[i] for i in order], rotation=30, ha='right')
    ax.set_ylabel('Mean |SHAP value|  (impact on cluster membership)')
    ax.set_title(title, fontweight='bold')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    _finish(fig, path, show)


def plot_global_importance(info, features, path, title, show):
    """Fallback (no shap): one bar per feature."""
    imp = info['global_importance']
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(features) + 1.5))
    ax.barh([features[i] for i in order], imp[order], color='#4c78a8')
    ax.set_xlabel('Permutation importance (share of total)')
    ax.set_title(title, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    _finish(fig, path, show)


def plot_beeswarm(info, X, X_z, features, palette, path, title, show):
    """One panel per cluster: SHAP value of each patient for that cluster,
    coloured by the patient's (standardized) feature value."""
    sv = info['sv']
    classes = info['classes']
    order = np.argsort(info['global_importance'])  # least important at bottom
    rng = np.random.default_rng(RANDOM_STATE)
    fig, axes = plt.subplots(1, len(classes), figsize=(4.8 * len(classes), 0.55 * len(features) + 2.5),
                             sharey=True, squeeze=False)
    sc = None
    for ci, (ax, cls) in enumerate(zip(axes[0], classes)):
        for row, fi in enumerate(order):
            sc = ax.scatter(sv[ci][:, fi], row + rng.uniform(-0.28, 0.28, len(X)),
                            c=X_z.iloc[:, fi].clip(-2, 2), cmap='coolwarm',
                            vmin=-2, vmax=2, s=10, alpha=0.8, linewidths=0)
        ax.axvline(0, color='#888', lw=0.8)
        ax.set_yticks(range(len(order)), [features[i] for i in order])
        ax.set_title(f'Cluster {cls}', fontweight='bold', color=palette.get(int(cls)))
        ax.set_xlabel('SHAP value (pushes toward this cluster ->)')
        ax.spines[['top', 'right']].set_visible(False)
    fig.colorbar(sc, ax=axes[0].tolist(), label='Feature value (z-score)', shrink=0.8)
    fig.suptitle(title, fontweight='bold')
    _finish(fig, path, show)


def plot_typical_patients(info, X, X_z, labels, features, palette, path, title, show):
    """Local explanation for the patient closest to each cluster's median."""
    sv, classes, model = info['sv'], info['classes'], info['model']
    fig, axes = plt.subplots(1, len(classes), figsize=(5.2 * len(classes), 4.2), squeeze=False)
    for ci, (ax, cls) in enumerate(zip(axes[0], classes)):
        members = np.where(labels == cls)[0]
        center = X_z.iloc[members].median()
        idx = members[np.argmin(((X_z.iloc[members] - center) ** 2).sum(axis=1).values)]
        vals = sv[ci][idx]
        order = np.argsort(np.abs(vals))
        names = [f'{features[i]} = {X.iloc[idx, i]:g}' for i in order]
        ax.barh(names, vals[order],
                color=[palette.get(int(cls)) if v > 0 else '#bdbdbd' for v in vals[order]])
        p = model.predict_proba(X.iloc[[idx]])[0][ci]
        ax.axvline(0, color='#888', lw=0.8)
        ax.set_title(f'Typical patient of cluster {cls}\nP(cluster {cls}) = {p:.2f}', fontsize=10)
        ax.set_xlabel('SHAP value')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(title, fontweight='bold')
    fig.tight_layout()
    _finish(fig, path, show)


def fit_rule_tree(X, labels, features, out_dir, fig_dir, title, show):
    """Depth-3 tree: simple if/else rules approximating the clustering."""
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5,
                                  class_weight='balanced', random_state=RANDOM_STATE)
    acc, _ = cv_fidelity(tree, X, labels)
    tree.fit(X, labels)
    rules = export_text(tree, feature_names=list(features))
    (out_dir / 'decision_tree_rules.txt').write_text(
        f'Depth-3 decision tree approximating the clusters\n'
        f'Cross-validated accuracy: {acc:.3f}\n\n{rules}')
    fig, ax = plt.subplots(figsize=(15, 7))
    plot_tree(tree, feature_names=list(features), class_names=[f'Cluster {c}' for c in tree.classes_],
              filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title(f'{title}\n(cross-validated accuracy {acc:.2f})', fontweight='bold')
    _finish(fig, fig_dir / 'decision_tree.png', show)
    return acc


def plot_cluster_vs_risk(imp_cluster, imp_risk, features, path, title, show):
    """Are clusters and true risk levels driven by the same features?"""
    order = np.argsort(imp_risk)[::-1]
    x = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(1.4 * len(features) + 2, 4.6))
    ax.bar(x - 0.2, imp_cluster[order], 0.4, label='Predicting cluster', color='#4c78a8')
    ax.bar(x + 0.2, imp_risk[order], 0.4, label='Predicting true RiskLevel', color='#f58518')
    ax.set_xticks(x, [features[i] for i in order], rotation=30, ha='right')
    ax.set_ylabel('Share of total importance')
    ax.set_title(title, fontweight='bold')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    _finish(fig, path, show)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(show=False, top_n=TOP_N_RUNS):
    master = pd.read_csv(MASTER_CSV)
    X, X_z, y_risk = load_features()
    features = list(X.columns)
    runs = select_runs(master, top_n)
    if runs.empty:
        raise SystemExit('No full_multivariate runs found in master_results.csv.')

    # Baseline model: the same analysis for the TRUE risk levels
    print('Fitting reference model on true RiskLevel ...')
    risk_info = explain(X, y_risk.values, RISK_MODEL_NAME)
    risk_acc, risk_bal = risk_info['cv_acc'], risk_info['cv_bal']
    majority = y_risk.value_counts(normalize=True).max()
    print(f'  RiskLevel from vitals: CV accuracy {risk_acc:.3f} '
          f'(majority-class baseline {majority:.3f})')

    summary = []
    for _, row in runs.iterrows():
        name = run_name(row)
        labels = parse_labels(row['Labels'])
        if len(labels) != len(X):
            raise ValueError(f'{name}: {len(labels)} labels but {len(X)} cleaned rows. '
                             'Re-run main.py to regenerate results.')
        print(f"\n{row['Algorithm']} k={int(row['k'])} (ARI {row['ARI']:.3f})")

        out_dir = EXPLAIN_DIR / name
        fig_dir = EXPLAIN_FIGURES / name
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        palette = build_palette(labels)
        label = f"{row['Algorithm']}, k={int(row['k'])}"

        # 1. profiles
        cluster_profile(X, labels, y_risk).to_csv(out_dir / 'cluster_profile.csv')
        plot_profile_heatmap(X, X_z, labels, fig_dir / 'cluster_profile_heatmap.png',
                             f'Cluster profiles: {label}', show)

        # 2. surrogate + SHAP
        info = explain(X, labels, name)
        print(f"  Surrogate fidelity (CV): accuracy {info['cv_acc']:.3f}, "
              f"balanced {info['cv_bal']:.3f}  [{info['method']}]")
        pd.DataFrame(info['class_importance'], index=[f'Cluster {c}' for c in info['classes']],
                     columns=features).to_csv(out_dir / 'feature_importance.csv')

        if info['method'] == 'shap':
            plot_importance_by_cluster(info, features, palette, fig_dir / 'shap_importance.png',
                                       f'What drives each cluster ({label})', show)
            plot_beeswarm(info, X, X_z, features, palette, fig_dir / 'shap_beeswarm.png',
                          f'SHAP value per patient ({label})', show)
            plot_typical_patients(info, X, X_z, labels, features, palette,
                                  fig_dir / 'shap_typical_patients.png',
                                  f'Local explanations ({label})', show)
        else:
            plot_global_importance(info, features, fig_dir / 'permutation_importance.png',
                                   f'Feature importance ({label})', show)

        # 3. rules
        tree_acc = fit_rule_tree(X, labels, features, out_dir, fig_dir,
                                 f'Rules approximating clusters ({label})', show)

        # 4. clusters vs risk levels
        plot_cluster_vs_risk(info['global_importance'], risk_info['global_importance'], features,
                             fig_dir / 'cluster_vs_risk_importance.png',
                             f'Same features? clusters vs risk level ({label})', show)

        top = [features[i] for i in np.argsort(info['global_importance'])[::-1][:3]]
        summary.append({
            'Algorithm': row['Algorithm'], 'k': int(row['k']), 'ARI': row['ARI'],
            'Method': info['method'],
            'Surrogate_CV_Accuracy': round(info['cv_acc'], 4),
            'Surrogate_CV_BalancedAccuracy': round(info['cv_bal'], 4),
            'RuleTree_CV_Accuracy': round(tree_acc, 4),
            'Top_Features': ', '.join(top),
            'RiskModel_CV_Accuracy': round(risk_acc, 4),
            'Majority_Baseline': round(majority, 4),
        })

    pd.DataFrame(summary).to_csv(EXPLAIN_DIR / 'surrogate_fidelity.csv', index=False)
    print(f'\nSummary: {EXPLAIN_DIR / "surrogate_fidelity.csv"}')


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    main(show=False)