"""Predict the risk level of new patients with every clustering model.

Clustering never saw the risk labels, so a cluster has no risk level of its own.
A new patient is categorized in two steps:

  1. Place the patient in a cluster, using the algorithm's own rule:
       KMeans / BIRCH : the model's predict()
       GMM            : soft membership probabilities
       Agglomerative  : the cluster with the smallest Ward merge cost (these
                        models cannot predict, so this mimics the linkage rule)
       DBSCAN         : the cluster of the nearest core point, if within eps;
                        otherwise the patient is an outlier and gets no prediction
  2. Turn the cluster into a risk level with the risk mix of the training
     patients in that cluster:  P(risk) = sum over clusters of
     P(cluster | patient) * P(risk | cluster).

CONFIDENCE is the largest P(risk). For hard-assignment models it is simply how
risk-pure the patient's cluster is, and it is not a calibrated probability.

Each model's configuration is its best full-multivariate run (k in 2-6, ranked by
ARI then NMI) from master_results.csv. A supervised Random Forest trained on the
true risk labels is shown as a reference, because it can output real class
probabilities.

EXPLANATION (per model, per patient): a surrogate Random Forest that predicts
the model's cluster from the original features is explained locally for this
patient, using SHAP when installed, otherwise by replacing one feature at a time
with its training average and measuring the drop in the cluster probability.
"""
import argparse
import sys
import warnings
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.algorithms.birch import fit_birch_exact_k
from src.evaluation.clustering_accuracy import parse_labels
from src.explainability.explain_clusters import (
    K_RANGE, RANDOM_STATE, RISK_ORDER, load_features, make_forest, normalize_shap,
)
from src.paths import MASTER_CSV, PREDICT_DIR, PREDICT_FIGURES

CLUSTER_MODELS = ['KMeans', 'Agglomerative', 'GMM', 'BIRCH', 'DBSCAN']
REFERENCE_MODEL = 'RandomForest (supervised reference)'
RISK_COLORS = {'low risk': '#4c9f70', 'mid risk': '#e0a030', 'high risk': '#c8553d'}
MAX_FIGURE_PATIENTS = 20

# CLI flag -> dataset column
INPUT_FLAGS = {
    'age': ('Age', 'age in years'),
    'systolic': ('SystolicBP', 'systolic blood pressure, mmHg'),
    'diastolic': ('DiastolicBP', 'diastolic blood pressure, mmHg'),
    'bs': ('BS', 'blood sugar, as in the dataset (about 6-19)'),
    'temp': ('BodyTemp', 'body temperature, F'),
    'hr': ('HeartRate', 'resting heart rate, bpm'),
}


@dataclass
class Bundle:
    """One fitted model plus everything needed to predict and explain."""
    name: str
    config: str
    membership: callable            # scaled row (1, f) -> {cluster: probability}
    risk_table: pd.DataFrame        # cluster x RISK_ORDER (training risk mix)
    cluster_means: pd.DataFrame     # cluster x feature (original units)
    surrogate: object               # RF: original features -> cluster / risk
    explainer: object = None
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def _best_runs(master):
    runs = master[(master['Subset'] == 'full_multivariate') & master['k'].between(*K_RANGE)]
    runs = runs.sort_values(['ARI', 'NMI'], ascending=False)
    return {a: runs[runs['Algorithm'] == a].iloc[0] for a in CLUSTER_MODELS
            if not runs[runs['Algorithm'] == a].empty}


def _one_hot(label):
    return {int(label): 1.0}


def _fit_one(name, row, X_s):
    """Refit one clustering model with the settings of its saved best run.

    Returns (labels, membership_function, config_text).
    """
    k = int(row['k'])
    if name == 'KMeans':
        m = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42).fit(X_s)
        return m.labels_, (lambda x: _one_hot(m.predict(x)[0])), f'k={k}'
    if name == 'GMM':
        m = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        labels = m.fit_predict(X_s)
        return labels, (lambda x: {int(c): float(p) for c, p in enumerate(m.predict_proba(x)[0])}), f'k={k}'
    if name == 'BIRCH':
        _, thr = fit_birch_exact_k(X_s, k)
        m = Birch(n_clusters=k, threshold=thr, branching_factor=50)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            labels = m.fit_predict(X_s)
        return labels, (lambda x: _one_hot(m.predict(x)[0])), f'k={k}'
    if name == 'Agglomerative':
        labels = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X_s)
        ids = np.unique(labels)
        cent = np.array([X_s[labels == c].mean(axis=0) for c in ids])
        size = np.array([(labels == c).sum() for c in ids])

        def member(x):  # Ward cost of adding one point to cluster c
            cost = size / (size + 1) * ((cent - x[0]) ** 2).sum(axis=1)
            return _one_hot(ids[np.argmin(cost)])
        return labels, member, f'k={k}, Ward'
    if name == 'DBSCAN':
        eps, ms = float(row['Eps']), int(row['Min_Samples'])
        m = DBSCAN(eps=eps, min_samples=ms).fit(X_s)
        core_labels = m.labels_[m.core_sample_indices_]
        nn = NearestNeighbors(n_neighbors=1).fit(m.components_)

        def member(x):
            dist, idx = nn.kneighbors(x)
            return _one_hot(core_labels[idx[0][0]]) if dist[0][0] <= eps else {}
        return m.labels_, member, f'eps={eps:g}, min_samples={ms}, k={k}'
    raise ValueError(name)


def build_models():
    """Fit all clustering models + the supervised reference on the cleaned data."""
    master = pd.read_csv(MASTER_CSV)
    X, _, y_risk = load_features()
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)
    runs = _best_runs(master)

    bundles = []
    for name in CLUSTER_MODELS:
        if name not in runs:
            print(f'  [{name}] no full_multivariate run found - skipped.')
            continue
        row = runs[name]
        labels, member, config = _fit_one(name, row, X_s)
        saved = parse_labels(row['Labels'])
        agree = adjusted_rand_score(saved, labels)
        if agree < 0.99:
            print(f'  [{name}] warning: refit differs from saved results (ARI {agree:.3f}); '
                  're-run main.py.')

        keep = labels != -1
        risk_table = (pd.crosstab(labels[keep], y_risk[keep], normalize='index')
                      .reindex(columns=RISK_ORDER, fill_value=0.0))
        surrogate = make_forest().fit(X[keep], labels[keep])
        bundles.append(Bundle(
            name=name, config=config, membership=member, risk_table=risk_table,
            cluster_means=X[keep].groupby(labels[keep]).mean(), surrogate=surrogate,
            extra={'sizes': pd.Series(labels[keep]).value_counts().to_dict()},
        ))
        print(f'  fitted {name} ({config})')

    # supervised reference: classes ARE risk levels
    rf = make_forest().fit(X, y_risk.values)
    ref_table = pd.DataFrame(np.eye(3), index=RISK_ORDER, columns=RISK_ORDER)
    bundles.append(Bundle(
        name=REFERENCE_MODEL, config='300 trees', membership=None, risk_table=ref_table,
        cluster_means=X.groupby(y_risk.values).mean(), surrogate=rf,
    ))
    context = {'X': X, 'scaler': scaler, 'features': list(X.columns),
               'lo': X.min(), 'hi': X.max(), 'mean': X.mean()}
    return bundles, context


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def predict_patient(bundle, x_row, x_scaled):
    """Return dict(cluster, probs{risk: p}, risk, confidence) or None for outliers."""
    if bundle.membership is None:  # supervised reference
        proba = dict(zip(bundle.surrogate.classes_, bundle.surrogate.predict_proba(x_row)[0]))
        probs = np.array([proba.get(r, 0.0) for r in RISK_ORDER])
        target = RISK_ORDER[int(np.argmax(probs))]
    else:
        member = bundle.membership(x_scaled)
        if not member:
            return None
        probs = sum(p * bundle.risk_table.loc[c].values for c, p in member.items())
        target = max(member, key=member.get)
    i = int(np.argmax(probs))
    return {'target': target, 'probs': dict(zip(RISK_ORDER, probs)),
            'risk': RISK_ORDER[i], 'confidence': float(probs[i])}


def local_contributions(bundle, x_row, target, X_train):
    """Per-feature contribution to P(target) for this patient, and the method used."""
    model = bundle.surrogate
    ci = list(model.classes_).index(target)
    try:
        import shap
        if bundle.explainer is None:
            bundle.explainer = shap.TreeExplainer(model)
        sv = normalize_shap(bundle.explainer.shap_values(x_row, check_additivity=False),
                            1, x_row.shape[1])
        return sv[ci][0], 'SHAP'
    except ImportError:
        base = model.predict_proba(x_row)[0][ci]
        contrib = []
        for f in x_row.columns:
            alt = x_row.copy()
            alt[f] = X_train[f].mean()
            contrib.append(base - model.predict_proba(alt)[0][ci])
        return np.array(contrib), 'feature replaced by average'


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _who(pid):
    return f'Patient {pid}' if pid != '' else 'Submitted patient'


def _inputs_text(x_row):
    return ' | '.join(f'{c} {x_row.iloc[0][c]:g}' for c in x_row.columns)


def plot_summary(pid, x_row, results, consensus, path, show):
    n_vote = sum(1 for r in results if r['model'] in CLUSTER_MODELS and r['probs'])
    names = [r['model'] for r in results] + [f'Consensus ({n_vote} clustering models)']
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(names) + 2.4))
    y = np.arange(len(names))[::-1]
    for yi, r in zip(y, results + [consensus]):
        if r['probs'] is None:
            ax.text(0.01, yi, 'Outlier: no cluster within eps, no prediction',
                    va='center', color='#666', fontsize=9)
            continue
        left = 0
        for risk in RISK_ORDER:
            w = r['probs'][risk]
            ax.barh(yi, w, left=left, color=RISK_COLORS[risk], height=0.6,
                    edgecolor='white', label=risk if yi == y[0] else None)
            left += w
        ax.text(1.01, yi, f"{r['risk'].split()[0].capitalize()} - {r['confidence']:.0%}",
                va='center', fontsize=10, fontweight='bold', color=RISK_COLORS[r['risk']])
    ax.set_yticks(y, [n if n != REFERENCE_MODEL else REFERENCE_MODEL for n in names])
    ax.axhline(0.5, color='#999', lw=0.8, ls='--')
    ax.axhline(1.5, color='#999', lw=0.8, ls=':')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability of each risk level')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.02), frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'{_who(pid)}: predicted risk by model\n{_inputs_text(x_row)}',
                 fontweight='bold', y=1.04, fontsize=11)
    fig.text(0.01, -0.03, 'Cluster models: confidence = risk purity of the patient\'s cluster '
             '(times membership certainty for GMM). Not a calibrated probability.',
             fontsize=8, color='#555')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


def plot_explanations(pid, x_row, results, path, show):
    n = len(results)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    for ax, r in zip(axes.ravel(), results):
        ax.spines[['top', 'right']].set_visible(False)
        if r['contrib'] is None:
            ax.text(0.5, 0.5, 'Outlier: no explanation', ha='center', va='center')
            ax.set_title(r['model'], fontsize=9, fontweight='bold')
            ax.axis('off')
            continue
        c = r['contrib']
        order = np.argsort(np.abs(c))
        labels = [f'{x_row.columns[i]} = {x_row.iloc[0, i]:g}' for i in order]
        ax.barh(labels, c[order],
                color=[RISK_COLORS[r['risk']] if v > 0 else '#bdbdbd' for v in c[order]])
        ax.axvline(0, color='#888', lw=0.8)
        what = f"cluster {r['target']}" if r['model'] != REFERENCE_MODEL else r['target']
        ax.set_title(f"{r['model']}\n-> {r['risk']} ({r['confidence']:.0%}), {what}",
                     fontsize=9, fontweight='bold')
        ax.set_xlabel(f"Effect on P({what})  [{r['method']}]", fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle(f'{_who(pid)}: why each model decided this\n{_inputs_text(x_row)}',
                 fontweight='bold', y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def analyze_patient(x_row, bundles, ctx):
    """Run every model on one patient (a one-row DataFrame in original units).

    Returns a dict with per-model results, the consensus of the clustering
    models, their vote counts, the supervised reference and range warnings.
    Used by both the command line and the web interface.
    """
    features = ctx['features']
    x_scaled = ctx['scaler'].transform(x_row)
    warns = [f'{f}={x_row.iloc[0][f]:g} outside training range '
             f'[{ctx["lo"][f]:g}, {ctx["hi"][f]:g}]'
             for f in features if not ctx['lo'][f] <= x_row.iloc[0][f] <= ctx['hi'][f]]

    results = []
    for b in bundles:
        pred = predict_patient(b, x_row, x_scaled)
        r = {'model': b.name, 'config': b.config, 'probs': None, 'contrib': None}
        if pred is not None:
            contrib, method = local_contributions(b, x_row, pred['target'], ctx['X'])
            r.update(pred, contrib=contrib, method=method)
        results.append(r)

    cluster_results = [r for r in results if r['model'] in CLUSTER_MODELS and r['probs']]
    ref = next(r for r in results if r['model'] == REFERENCE_MODEL)
    mean_p = np.mean([[r['probs'][k] for k in RISK_ORDER] for r in cluster_results], axis=0)
    ci = int(np.argmax(mean_p))
    consensus = {'probs': dict(zip(RISK_ORDER, mean_p)), 'risk': RISK_ORDER[ci],
                 'confidence': float(mean_p[ci])}
    votes = pd.Series([r['risk'] for r in cluster_results]).value_counts()
    return {'results': results, 'consensus': consensus, 'votes': votes,
            'reference': ref, 'warnings': warns, 'n_voting': len(cluster_results)}


def run_patients(patients, bundles, ctx, show=False):
    features = ctx['features']
    long_rows, wide_rows, expl_rows = [], [], []
    PREDICT_FIGURES.mkdir(parents=True, exist_ok=True)

    for pid, (_, p) in enumerate(patients.iterrows(), start=1):
        x_row = pd.DataFrame([p[features].astype(float).values], columns=features)
        a = analyze_patient(x_row, bundles, ctx)
        results, consensus, votes, ref, warns = (
            a['results'], a['consensus'], a['votes'], a['reference'], a['warnings'])
        for w in warns:
            print(f'  Patient {pid} warning: {w}')

        for r in results:
            row = {'Patient': pid, 'Model': r['model'], 'Config': r['config'],
                   'Cluster': r.get('target') if r['model'] != REFERENCE_MODEL else '',
                   'Predicted_Risk': r['risk'] if r['probs'] else 'outlier (no prediction)',
                   'Confidence': round(r['confidence'], 3) if r['probs'] else np.nan}
            for k in RISK_ORDER:
                row[f'P_{k.split()[0]}'] = round(r['probs'][k], 3) if r['probs'] else np.nan
            if r['contrib'] is not None:
                top = np.argsort(np.abs(r['contrib']))[::-1][:3]
                row['Top_Reasons'] = ', '.join(features[i] for i in top)
                for f, v in zip(features, r['contrib']):
                    expl_rows.append({'Patient': pid, 'Model': r['model'], 'Feature': f,
                                      'Value': x_row.iloc[0][f], 'Contribution': round(float(v), 4),
                                      'Method': r['method']})
            long_rows.append(row)

        wide_rows.append({
            'Patient': pid, **{f: x_row.iloc[0][f] for f in features},
            'Consensus_Risk': consensus['risk'], 'Consensus_Confidence': round(consensus['confidence'], 3),
            'Votes': ', '.join(f'{k}: {v}' for k, v in votes.items()),
            f'{REFERENCE_MODEL}_Risk': ref['risk'], f'{REFERENCE_MODEL}_Confidence': round(ref['confidence'], 3),
            'Warnings': '; '.join(warns),
        })

        if pid <= MAX_FIGURE_PATIENTS:
            plot_summary(pid, x_row, results, consensus, PREDICT_FIGURES / f'patient_{pid}_summary.png', show)
            plot_explanations(pid, x_row, results, PREDICT_FIGURES / f'patient_{pid}_explanations.png', show)
        elif pid == MAX_FIGURE_PATIENTS + 1:
            print(f'  Figures are made for the first {MAX_FIGURE_PATIENTS} patients only; '
                  'tables cover everyone.')

        print(f'\nPatient {pid}: {_inputs_text(x_row)}')
        for r in results:
            if r['probs']:
                print(f"  {r['model']:<36} {r['risk']:<10} confidence {r['confidence']:.0%}")
            else:
                print(f"  {r['model']:<36} outlier (no prediction)")
        print(f"  {'Consensus (clustering models)':<36} {consensus['risk']:<10} "
              f"confidence {consensus['confidence']:.0%}  (votes: {', '.join(f'{k} {v}' for k, v in votes.items())})")

    PREDICT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(long_rows).to_csv(PREDICT_DIR / 'model_predictions.csv', index=False)
    pd.DataFrame(wide_rows).to_csv(PREDICT_DIR / 'patient_summary.csv', index=False)
    pd.DataFrame(expl_rows).to_csv(PREDICT_DIR / 'prediction_explanations.csv', index=False)
    print(f'\nTables: {PREDICT_DIR}\nFigures: {PREDICT_FIGURES}')


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Predict risk level for new patients with every model.')
    for flag, (col, desc) in INPUT_FLAGS.items():
        ap.add_argument(f'--{flag}', type=float, help=f'{col}: {desc}')
    ap.add_argument('--csv', help='CSV of patients with columns ' + ', '.join(c for c, _ in INPUT_FLAGS.values()))
    ap.add_argument('--show', action='store_true', help='also display figures')
    return ap.parse_args(argv)


def read_patients(args, features):
    if args.csv:
        df = pd.read_csv(args.csv)
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise SystemExit(f'{args.csv} is missing columns: {", ".join(missing)}')
        df = df[features].dropna()
        if df.empty:
            raise SystemExit('No complete patient rows found in the CSV.')
        return df
    given = {col: getattr(args, flag) for flag, (col, _) in INPUT_FLAGS.items()}
    if all(v is None for v in given.values()):
        if not sys.stdin.isatty():
            raise SystemExit('Give patient values with flags (see --help) or use --csv.')
        print('Enter the patient\'s values:')
        for flag, (col, desc) in INPUT_FLAGS.items():
            while True:
                try:
                    given[col] = float(input(f'  {col} ({desc}): '))
                    break
                except ValueError:
                    print('    please enter a number')
    missing = [c for c, v in given.items() if v is None]
    if missing:
        raise SystemExit('Missing values for: ' + ', '.join(missing) + '. Give all six or none.')
    return pd.DataFrame([given])[features]


def main(argv=None):
    args = parse_args(argv)
    if not args.show:
        import matplotlib
        matplotlib.use('Agg')
    if not MASTER_CSV.exists():
        raise SystemExit('Results not found. Run `python main.py` first.')
    print('Fitting models on the cleaned training data ...')
    bundles, ctx = build_models()
    patients = read_patients(args, ctx['features'])
    run_patients(patients, bundles, ctx, show=args.show)


if __name__ == '__main__':
    main()