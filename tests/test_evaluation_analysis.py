"""평가 분석(Calibration, Operating Point, Subgroup, External, Grad-CAM) 단위 테스트"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.constants import DISEASES
from src.evaluation import calibration as cal
from src.evaluation.external_validation import mapped_scores, target_matrix
from src.evaluation.metrics import _binary_ece
from src.evaluation.operating_point import confusion_metrics, roc_targets, thresholds_for
from src.evaluation.subgroup import add_groups, relative_gap
from src.evaluation.tta_ensemble import MODEL_ORDER, decide_tta, oof_final_probs
from src.xai.anatomy import region_fractions
from src.xai.gradcam import GradCAM, overlay, target_layer
from src.xai.gradcam_analysis import long_format, sample_positive_predictions, select_errors

rng = np.random.default_rng(0)


# --- TTA 결정 (OOF) ----------------------------------------------------------

def _oof_logits(flip_noise: float):
    """원본 logit은 정답과 상관, 반전 logit은 원본 + 잡음 (잡음이 크면 TTA가 손해)"""
    n = 2000
    y = (rng.random((n, len(DISEASES))) < 0.2).astype(np.float32)
    folds = np.arange(n) % 5
    logits = {}
    for m in MODEL_ORDER:
        orig = (y * 2 - 1) + rng.normal(0, 1.5, y.shape)
        logits[f"{m}/orig"] = orig
        logits[f"{m}/flip"] = orig + rng.normal(0, flip_noise, y.shape)
    return logits, y, folds


def test_tta_rejected_when_flip_hurts_oof():
    logits, y, folds = _oof_logits(flip_noise=5.0)
    d = decide_tta(logits, y, folds)
    assert d["use"] is False and d["oof_mean_diff"] < 0
    assert len(d["oof_fold_auroc_tta"]) == 5


def test_oof_final_probs_respects_tta_flag():
    logits, _, _ = _oof_logits(flip_noise=1.0)
    no_tta = oof_final_probs(logits, tta=False)
    expected = np.mean([1 / (1 + np.exp(-logits[f"{m}/orig"])) for m in MODEL_ORDER], axis=0)
    np.testing.assert_allclose(no_tta, expected, rtol=1e-6)
    assert not np.allclose(oof_final_probs(logits, tta=True), no_tta)


# --- Calibration -----------------------------------------------------------

def test_ece_matches_training_metric():
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    assert cal.ece_mce(y, p)[0] == pytest.approx(_binary_ece(y, p, cal.N_BINS))


def test_temperature_recovers_known_value():
    z_true = rng.normal(0, 2, 20000)
    y = (rng.random(z_true.size) < 1 / (1 + np.exp(-z_true))).astype(int)
    assert cal.fit_temperature(z_true * 2.5, y) == pytest.approx(2.5, rel=0.1)


def test_prior_correction_fixes_pos_weight_shift():
    """pos_weight w로 학습하면 logit이 log w만큼 밀린다 → prior 보정이 ECE를 크게 줄인다"""
    n, w = 20000, 20.0
    z_true = rng.normal(-3, 1.5, (n, len(DISEASES)))
    y = (rng.random(z_true.shape) < 1 / (1 + np.exp(-z_true))).astype(int)
    p_shifted = 1 / (1 + np.exp(-(z_true + np.log(w))))
    pos_weight = {d: w for d in DISEASES}

    before = cal.mean_ece_mce(y, p_shifted)[0]
    params = cal.fit("prior_temperature", p_shifted, y, pos_weight)
    after = cal.mean_ece_mce(y, cal.apply(p_shifted, params))[0]
    assert before > 0.2 and after < 0.02
    assert all(t == pytest.approx(1.0, abs=0.1) for t in params["temperature"].values())


def test_calibration_keeps_ranking():
    p = rng.random((300, len(DISEASES)))
    y = rng.integers(0, 2, p.shape)
    for method in cal.METHODS:
        out = cal.apply(p, cal.fit(method, p, y, {d: 5.0 for d in DISEASES}))
        assert (np.argsort(out[:, 0]) == np.argsort(p[:, 0])).all()


def test_choose_prefers_temperature_family():
    assert cal.choose({"none": .3, "temperature": .2, "temperature_per_disease": .09,
                       "prior_temperature": .01, "platt_per_disease": .005}) == "temperature_per_disease"
    assert cal.choose({"none": .3, "temperature": .2, "temperature_per_disease": .2,
                       "prior_temperature": .05, "platt_per_disease": .01}) == "prior_temperature"
    assert cal.choose({"none": .3, "temperature": .2, "temperature_per_disease": .2,
                       "prior_temperature": .15, "platt_per_disease": .12}) == "platt_per_disease"


# --- Operating point -------------------------------------------------------

def test_thresholds_meet_targets_on_same_data():
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    thr = thresholds_for(y, p)
    assert confusion_metrics(y, p, thr["sens90"])["sensitivity"] >= 0.9
    assert confusion_metrics(y, p, thr["spec90"])["specificity"] >= 0.9
    assert thr["spec90"] >= thr["youden"] >= thr["sens90"] or thr["spec90"] >= thr["sens90"]
    t = roc_targets(y, p)
    assert 0 <= t["spec_at_sens90"] <= 1 and 0 <= t["sens_at_spec90"] <= 1


def test_confusion_metrics_values():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1])
    m = confusion_metrics(y, p, 0.5)
    assert (m["sensitivity"], m["specificity"], m["ppv"], m["npv"]) == (0.5, 0.5, 0.5, 0.5)


# --- Subgroup --------------------------------------------------------------

def test_age_groups_boundaries():
    df = pd.DataFrame({"Patient Gender": ["M", "F", "M", "F", "M"],
                       "Patient Age Years": [39.0, 40.0, 59.0, 60.0, 150.0],
                       "Age Valid": [True, True, True, True, False],
                       "View Position": ["PA", "AP", "PA", "AP", "PA"]})
    g = add_groups(df)
    assert g["Age Group"].tolist()[:4] == ["0-40", "40-60", "40-60", "60+"]
    assert pd.isna(g["Age Group"].iloc[4])
    assert g["Gender"].tolist()[:2] == ["Male", "Female"]


def test_relative_gap():
    assert relative_gap(pd.Series([0.8, 0.72])) == pytest.approx(0.1)
    assert np.isnan(relative_gap(pd.Series([0.8, np.nan])))


# --- External --------------------------------------------------------------

def test_chexpert_uncertainty_policies():
    df = pd.DataFrame({c: [1.0, -1.0, np.nan] for c in
                       ["Atelectasis", "Cardiomegaly", "Pleural Effusion", "Pneumonia", "Pneumothorax",
                        "Consolidation", "Edema", "Lung Lesion"]})
    _, ign = target_matrix(df, "u_ignore")
    _, zer = target_matrix(df, "u_zeros")
    _, one = target_matrix(df, "u_ones")
    assert ign[0, 0] == 1 and np.isnan(ign[1, 0]) and ign[2, 0] == 0      # 빈칸 = 음성
    assert zer[1, 0] == 0 and one[1, 0] == 1


def test_mapped_scores_lung_lesion_is_max_of_mass_nodule():
    p = np.zeros((1, len(DISEASES)))
    p[0, DISEASES.index("Mass")], p[0, DISEASES.index("Nodule")] = 0.2, 0.7
    p[0, DISEASES.index("Effusion")] = 0.4
    s = mapped_scores(p)
    assert s[0, -1] == pytest.approx(0.7)
    assert s[0, 2] == pytest.approx(0.4)          # Effusion → Pleural Effusion (3번째)


# --- Grad-CAM --------------------------------------------------------------

@pytest.mark.parametrize("name", ["densenet121", "efficientnet_b0"])
def test_gradcam_shape_and_range(name):
    from src.training.model import build_model
    model = build_model(name, pretrained=False)
    cam = GradCAM(model, target_layer(name, model))
    heat, probs = cam(torch.randn(3, 224, 224), [0, 1])
    cam.remove()
    assert heat.shape == (2, 224, 224) and probs.shape == (14,)
    assert heat.min() >= 0 and heat.max() <= 1
    # Hook 제거 후 일반 추론이 그대로 동작
    assert model(torch.randn(1, 3, 224, 224)).shape == (1, 14)


def test_overlay_and_region_fraction():
    gray = np.zeros((224, 224), np.uint8)
    cam = np.zeros((224, 224), np.float32)
    cam[100:120, 100:120] = 1
    assert overlay(gray, cam).shape == (224, 224, 3)
    masks = {k: np.zeros((224, 224), bool) for k in ("lung", "heart", "mediastinum", "diaphragm", "bone")}
    masks["lung"][90:130, 90:130] = True
    assert region_fractions(cam, masks, "Nodule")["expected"] == pytest.approx(1.0)
    masks["lung"][:] = False
    masks["bone"][100:120, 100:120] = True
    fr = region_fractions(cam, masks, "Nodule")
    assert fr["expected"] == 0 and fr["bone"] == pytest.approx(1.0)


def _toy_predictions():
    n = 40
    df = pd.DataFrame({"Image Index": [f"img{i}.png" for i in range(n)]})
    for d in DISEASES:
        df[d] = rng.integers(0, 2, n)
        df[f"prob_{d}"] = rng.random(n)
    return df


def test_error_selection_is_confident_and_diverse():
    df = _toy_predictions()
    long = long_format(df, {d: 0.5 for d in DISEASES})
    fp = select_errors(long, "FP", 5)
    fn = select_errors(long, "FN", 5)
    assert len(fp) == 5 and len(fn) == 5
    assert ((fp["label"] == 0) & (fp["prob"] >= 0.5)).all()
    assert ((fn["label"] == 1) & (fn["prob"] < 0.5)).all()
    assert fp["disease"].is_unique and fp["Image Index"].is_unique


def test_positive_sample_balances_diseases():
    long = long_format(_toy_predictions(), {d: 0.5 for d in DISEASES})
    s = sample_positive_predictions(long, 28)
    assert (s["prob"] >= s["threshold"]).all()
    assert s["disease"].value_counts().max() <= 2
