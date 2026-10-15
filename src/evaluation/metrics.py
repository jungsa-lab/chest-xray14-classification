import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def calculate_metrics(y_true, y_prob, n_bins=15):
    """14-class macro AUROC, AUPRC, ECE. ECE = mean of per-class equal-width ECE."""
    true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(y_prob, dtype=np.float64)
    if true.ndim != 2 or true.shape != prob.shape or true.shape[1] != 14:
        raise ValueError('Expected true/prob arrays with shape [N, 14]')
    if len(true) == 0 or not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError('Empty or invalid probabilities')
    auroc, auprc, ece = [], [], []
    for j in range(14):
        y, p = true[:, j], prob[:, j]
        if np.unique(y).size < 2:
            raise ValueError(f'Class {j} has only one label in evaluation split')
        auroc.append(float(roc_auc_score(y, p)))
        auprc.append(float(average_precision_score(y, p)))
        bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
        cal_error = 0.0
        for k in range(n_bins):
            mask = bins == k
            if mask.any():
                cal_error += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
        ece.append(cal_error)
    return {'mean_auroc': float(np.mean(auroc)),
            'mean_auprc': float(np.mean(auprc)),
            'ece': float(np.mean(ece)),
            'class_auroc': auroc, 'class_auprc': auprc, 'class_ece': ece}
