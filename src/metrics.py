"""APCER/BPCER/ACER/EER/AUC hesaplama — Faz A.3 degerlendirme metrikleri.

Sozlesme (bkz. 03_build_splits.py): label 0=live (bona fide), 1=spoof (attack).
'score': modelin spoof sinifina verdigi olasilik (softmax[:,1]) — yuksek skor
daha 'spoof' demek. Tum fonksiyonlar bu sozlesmeye gore yazildi.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple:
    """(eer, esik) doner. EER: APCER(t) ile BPCER(t)'nin esitlendigi (ya da en
    yakin kesistigi) nokta — threshold-bagimsiz bir genel ayirt edicilik olcusu.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    apcer_curve = 1 - tpr  # spoof'un live sanilma orani (skor dusuk kalinca)
    bpcer_curve = fpr      # live'in spoof sanilma orani (skor yuksek cikinca)
    idx = int(np.nanargmin(np.abs(apcer_curve - bpcer_curve)))
    eer = float((apcer_curve[idx] + bpcer_curve[idx]) / 2)
    return eer, float(thresholds[idx])


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def compute_apcer_bpcer_acer(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Sabit bir esikte (skor >= threshold -> 'spoof' tahmini) APCER/BPCER/ACER."""
    preds = (scores >= threshold).astype(int)
    spoof_mask = labels == 1
    live_mask = labels == 0

    apcer = float(np.mean(preds[spoof_mask] == 0)) if spoof_mask.any() else 0.0
    bpcer = float(np.mean(preds[live_mask] == 1)) if live_mask.any() else 0.0
    return {"apcer": apcer, "bpcer": bpcer, "acer": (apcer + bpcer) / 2}
