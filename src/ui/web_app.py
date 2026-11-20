"""Local web interface: enter a patient, run every model, read the report.

The models are fitted once at start-up (about 10 seconds); after that each
submission takes about a second. Nothing is written to disk and nothing leaves
your computer.
"""
import argparse
import base64
import io
import threading
import webbrowser
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from flask import Flask, jsonify, render_template, request  # noqa: E402

from src.paths import MASTER_CSV  # noqa: E402
from src.prediction.predict_patient import (  # noqa: E402
    CLUSTER_MODELS, REFERENCE_MODEL, RISK_ORDER, analyze_patient, build_models,
    plot_explanations, plot_summary,
)

# key, label, unit, hint, hard limits (inputs outside these are rejected)
FIELDS = [
    ('Age', 'Age', 'years', 'Age in years', 10, 100, 1),
    ('SystolicBP', 'Systolic blood pressure', 'mmHg', 'Upper number', 60, 250, 1),
    ('DiastolicBP', 'Diastolic blood pressure', 'mmHg', 'Lower number', 30, 150, 1),
    ('BS', 'Blood sugar', 'units', 'Same scale as the dataset (about 6 to 19)', 1, 40, 0.1),
    ('BodyTemp', 'Body temperature', '°F', 'Fahrenheit', 90, 110, 0.1),
    ('HeartRate', 'Resting heart rate', 'bpm', 'Beats per minute', 30, 220, 1),
]
EXAMPLES = {
    'Lower-risk example': dict(Age=22, SystolicBP=100, DiastolicBP=65, BS=7, BodyTemp=98, HeartRate=72),
    'Borderline example': dict(Age=29, SystolicBP=120, DiastolicBP=80, BS=7.5, BodyTemp=101, HeartRate=76),
    'Higher-risk example': dict(Age=35, SystolicBP=140, DiastolicBP=90, BS=13, BodyTemp=98, HeartRate=80),
}

_plot_lock = threading.Lock()  # matplotlib's pyplot is not thread-safe


def _png_b64(plot_fn, *args):
    buf = io.BytesIO()
    with _plot_lock:
        plot_fn('', *args, buf, False)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def validate(payload):
    """Return (values dict, error string or None)."""
    values = {}
    for key, label, unit, _, lo, hi, _ in FIELDS:
        raw = payload.get(key)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None, f'{label}: please enter a number.'
        if not np.isfinite(v) or not lo <= v <= hi:
            return None, f'{label}: {raw} is not plausible (allowed {lo:g} to {hi:g} {unit}).'
        values[key] = v
    if values['DiastolicBP'] >= values['SystolicBP']:
        return None, 'Diastolic pressure must be lower than systolic pressure.'
    return values, None


def create_app(bundles, ctx):
    app = Flask(__name__, template_folder=str(Path(__file__).parent / 'templates'))
    features = ctx['features']
    bundle_by_name = {b.name: b for b in bundles}

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/meta')
    def meta():
        fields = []
        for key, label, unit, hint, lo, hi, step in FIELDS:
            fields.append({'key': key, 'label': label, 'unit': unit, 'hint': hint,
                           'min': lo, 'max': hi, 'step': step,
                           'train_min': float(ctx['lo'][key]), 'train_max': float(ctx['hi'][key]),
                           'train_mean': round(float(ctx['mean'][key]), 2)})
        return jsonify({'fields': fields, 'examples': EXAMPLES, 'risk_order': RISK_ORDER,
                        'models': [{'name': b.name, 'config': b.config} for b in bundles],
                        'n_train': int(len(ctx['X']))})

    @app.post('/api/predict')
    def predict():
        values, err = validate(request.get_json(silent=True) or {})
        if err:
            return jsonify({'error': err}), 400
        x_row = pd.DataFrame([[values[f] for f in features]], columns=features)

        with _plot_lock:  # model objects are shared; keep each request serial
            a = analyze_patient(x_row, bundles, ctx)
        results, consensus = a['results'], a['consensus']

        models = []
        for r in results:
            b = bundle_by_name[r['model']]
            m = {'model': r['model'], 'config': r['config'],
                 'is_reference': r['model'] == REFERENCE_MODEL,
                 'predicted': r['probs'] is not None}
            if r['probs'] is not None:
                target = r['target']
                m.update(
                    risk=r['risk'], confidence=float(r['confidence']), method=r['method'],
                    probs={k: float(v) for k, v in r['probs'].items()},
                    target=str(target),
                    cluster_size=b.extra.get('sizes', {}).get(target),
                    risk_mix={k: float(v) for k, v in b.risk_table.loc[target].items()},
                    contributions=[{'feature': f, 'value': float(x_row.iloc[0][f]),
                                    'cluster_mean': _num(b.cluster_means.loc[target, f]),
                                    'effect': float(c)}
                                   for f, c in zip(features, r['contrib'])],
                )
            models.append(m)

        return jsonify({
            'inputs': values,
            'warnings': a['warnings'],
            'consensus': {'risk': consensus['risk'], 'confidence': float(consensus['confidence']),
                          'probs': {k: float(v) for k, v in consensus['probs'].items()},
                          'votes': {k: int(v) for k, v in a['votes'].items()},
                          'n_voting': a['n_voting'], 'n_clustering': len(CLUSTER_MODELS)},
            'models': models,
            'summary_png': _png_b64(plot_summary, x_row, results, consensus),
            'explain_png': _png_b64(plot_explanations, x_row, results),
        })

    return app


def main():
    ap = argparse.ArgumentParser(description='Run the patient risk web interface.')
    ap.add_argument('--port', type=int, default=5050)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()

    if not MASTER_CSV.exists():
        raise SystemExit('Results not found. Run `python3 main.py` first.')
    print('Fitting models on the cleaned training data (about 10 seconds) ...')
    bundles, ctx = build_models()
    app = create_app(bundles, ctx)

    url = f'http://127.0.0.1:{args.port}'
    print(f'\nReady. Open {url}   (Ctrl+C to stop)')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host='127.0.0.1', port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()