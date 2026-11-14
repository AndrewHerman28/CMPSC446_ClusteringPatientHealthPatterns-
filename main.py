"""Runs the whole project end to end, in the required order.

    python main.py                  # compute everything and save all figures
    python main.py --show           # also pop up each figure (blocks until closed)
    python main.py --skip-figures   # stop after the CSV results and accuracy report

Steps: check raw data -> preprocessing/clustering/PCA -> accuracy report ->
cluster figures -> PCA figures (2D / 3D) -> accuracy comparison figure.
The accuracy report must run after the clustering step, because that step
rewrites the result CSVs without Accuracy/Purity.
"""
import argparse
import time
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description='Run the full clustering pipeline.')
    parser.add_argument('--show', action='store_true',
                        help='display figures interactively (default: save only)')
    parser.add_argument('--skip-figures', action='store_true',
                        help='stop after the CSV results and accuracy report')
    return parser.parse_args()


def run_step(number, total, title, func):
    print(f'\n[{number}/{total}] {title}')
    print('-' * 60)
    start = time.time()
    func()
    print(f'Done in {time.time() - start:.1f}s')


def main():
    args = parse_args()

    # Must be chosen before any module imports pyplot
    if not args.show:
        import matplotlib
        matplotlib.use('Agg')
        warnings.filterwarnings('ignore', message='.*non-interactive.*')

    from src.paths import FIGURES, MODELS_DIR, RAW_CSV

    total = 3 if args.skip_figures else 6
    pipeline_start = time.time()

    def check_raw_data():
        if not RAW_CSV.exists():
            raise SystemExit(
                f'Raw dataset not found at:\n  {RAW_CSV}\n'
                'Place "Maternal Health Risk Data Set.csv" in data/raw/ and rerun.'
            )
        print(f'Found raw data: {RAW_CSV.name}')

    def compute_results():
        from src.run_and_save_results import main as run_all
        run_all()
        print(f'Results written to: {MODELS_DIR.parent}')

    def accuracy_report():
        from src.evaluation.clustering_accuracy import main as report
        report()

    def cluster_figures():
        from src.visualization.show_clusters import main as plot_algorithms
        plot_algorithms(show=args.show)

    def pca_figures():
        from src.visualization.show_pca import main as plot_pca
        plot_pca(show=args.show)

    def accuracy_figure():
        from src.visualization.show_accuracy import plot_accuracy
        plot_accuracy(show=args.show)

    run_step(1, total, 'Checking raw data', check_raw_data)
    run_step(2, total, 'Preprocessing, clustering, PCA, accuracy', compute_results)
    run_step(3, total, 'Accuracy report', accuracy_report)

    if not args.skip_figures:
        run_step(4, total, 'Cluster figures (per algorithm)', cluster_figures)
        run_step(5, total, 'PCA cluster figures (2D / 3D)', pca_figures)
        run_step(6, total, 'Accuracy comparison figure', accuracy_figure)
        print(f'\nFigures saved under: {FIGURES}')

    print(f'\nPipeline finished in {time.time() - pipeline_start:.1f}s')


if __name__ == '__main__':
    main()
