import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from train import train_one


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'dml'])
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min-delta', type=float, default=0.0)
    p.add_argument('--clahe', action='store_true')
    p.add_argument('--output-dir', default='checkpoints')
    p.add_argument('--report-dir', default='reports/training')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        p.error('epochs, batch-size and patience must be positive')
    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)
    rows = []
    for gamma in [0, 1, 2]:
        for fold in range(5):
            current = SimpleNamespace(**vars(args), gamma=gamma, fold=fold, max_batches=None)
            print(f'\n=== gamma={gamma}, fold={fold} ===', flush=True)
            row = train_one(current)
            rows.append(row)
            pd.DataFrame(rows).to_csv(report / 'fold_results.csv', index=False)
    df = pd.DataFrame(rows)
    summary = df.groupby('gamma', as_index=False).agg(
        auroc_mean=('mean_auroc', 'mean'), auroc_std=('mean_auroc', 'std'),
        auprc_mean=('mean_auprc', 'mean'), auprc_std=('mean_auprc', 'std'),
        ece_mean=('ece', 'mean'), ece_std=('ece', 'std'), n_folds=('fold', 'nunique'))
    if not (summary.n_folds == 5).all():
        raise RuntimeError('Some gamma does not have all five folds')
    summary.to_csv(report / 'gamma_comparison.csv', index=False)
    best = summary.sort_values(['auroc_mean', 'auprc_mean', 'ece_mean'], ascending=[False, False, True]).iloc[0]
    (report / 'selected_gamma.json').write_text(json.dumps({
        'gamma': int(best['gamma']), 'criterion': 'Highest five-fold mean AUROC; tie-breaker AUPRC then lower ECE',
        'note': 'Validation only. No test-set performance is used for selection.'
    }, indent=2), encoding='utf-8')
    print('\n=== Gamma comparison (fold mean +/- sample std) ===')
    print(summary.to_string(index=False))
    print('Selected gamma:', int(best['gamma']))


if __name__ == '__main__':
    main()
