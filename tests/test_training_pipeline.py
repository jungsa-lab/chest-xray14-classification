"""학습 파이프라인 테스트: 설정, Early Stopping, Fold 학습·이어서 학습, 시간 제한, 요약표, 작업 대기열

실제 데이터 대신 작은 가짜 이미지와 아주 작은 모델을 사용해 수 초 안에 끝난다.
"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.constants import DISEASES
from src.training import model as model_module
from src.training import run as run_module
from src.training import run_queue
from src.training.config import TrainConfig, load_config, parse_overrides
from src.training.summarize import write_summary
from src.training.trainer import (
    BEST_FILE, HISTORY_FILE, LAST_FILE, PREDICTION_FILE, RESULT_FILE,
    BestTracker, TimeLimitReached, fold_dir, subsample, train_fold,
)
from tests._tiny_run import TinyNet, register

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_fake_metadata(raw_dir: Path, n_train: int = 40, n_test: int = 5) -> pd.DataFrame:
    """Fold마다 모든 질환에 양성·음성이 있는 가짜 메타데이터와 이미지

    i % 5로 Fold를 정하고 (i + j) % 3 == 0으로 질환 j의 양성을 정한다.
    5와 3이 서로소라 Fold 안에서도 양성·음성이 모두 나온다.
    """
    raw_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_train + n_test):
        name = f"img_{i:03d}.png"
        Image.fromarray(rng.integers(0, 256, (224, 224), dtype=np.uint8)).save(raw_dir / name)
        is_test = i >= n_train
        row = {
            "Image Index": name,
            "Image Path": name,
            "Patient ID": i,
            "Split": "test" if is_test else "train",
            "Fold": -1 if is_test else i % 5,
        }
        row.update({d: int((i + j) % 3 == 0) for j, d in enumerate(DISEASES)})
        rows.append(row)
    return pd.DataFrame(rows)


class PipelineTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        register()

    @classmethod
    def tearDownClass(cls):
        model_module.MODEL_BUILDERS.pop("tiny", None)
        model_module.MODEL_BUILDERS.pop("tiny2", None)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.metadata = make_fake_metadata(self.root / "raw")
        self.cfg = TrainConfig(
            model="tiny", pretrained=False, batch_size=8, epochs=3, lr=1e-2,
            early_stop_patience=10, num_workers=0, device="cpu", amp=False,
            raw_dir=str(self.root / "raw"), output_dir=str(self.root / "outputs"),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def train(self, cfg=None, fold=0, **kwargs):
        return train_fold(cfg or self.cfg, fold, metadata=self.metadata,
                          log=lambda *_: None, **kwargs)


# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):

    def test_project_config_loads(self):
        cfg = load_config(PROJECT_ROOT / "configs" / "train.yaml")
        self.assertEqual(cfg.model, "densenet121")
        self.assertIsInstance(cfg.lr, float)
        # 요구사항: Early Stopping patience=5
        self.assertEqual(cfg.early_stop_patience, 5)

    def test_exponent_string_becomes_float(self):
        # PyYAML은 1e-4를 문자열로 읽음
        cfg = load_config(None, {"lr": "1e-4", "gamma": 1})
        self.assertEqual(cfg.lr, 1e-4)
        self.assertIsInstance(cfg.gamma, float)

    def test_unknown_key_and_bad_types_raise(self):
        with self.assertRaises(ValueError):
            load_config(None, {"typo": 1})
        with self.assertRaises(ValueError):
            TrainConfig(amp="yes")
        with self.assertRaises(ValueError):
            TrainConfig(batch_size=8.5)
        with self.assertRaises(ValueError):
            TrainConfig(gamma=-1)
        with self.assertRaises(ValueError):
            TrainConfig(seed=-1)
        for name in ("a/b", "a\\b", ".."):
            with self.assertRaises(ValueError):
                TrainConfig(name=name)

    def test_parse_overrides(self):
        self.assertEqual(
            parse_overrides(["gamma=0", "amp=false", "name=exp a"]),
            {"gamma": 0, "amp": False, "name": "exp a"},
        )
        with self.assertRaises(ValueError):
            parse_overrides(["gamma"])

    def test_fingerprint_tracks_only_result_affecting_fields(self):
        base = TrainConfig()
        self.assertEqual(base.fingerprint(), TrainConfig(num_workers=0, device="cpu").fingerprint())
        self.assertNotEqual(base.fingerprint(), TrainConfig(lr=3e-4).fingerprint())
        self.assertNotEqual(base.fingerprint(), TrainConfig(epochs=1).fingerprint())
        self.assertTrue(base.run_id.startswith("densenet121_g2_clahe-"))


class TestBestTracker(unittest.TestCase):

    def test_stops_after_patience(self):
        t = BestTracker(patience=2)
        self.assertTrue(t.update(0.70, 1))
        self.assertFalse(t.update(0.69, 2))
        self.assertFalse(t.should_stop)
        self.assertFalse(t.update(0.70, 3))     # 같은 점수는 개선 아님
        self.assertTrue(t.should_stop)
        self.assertEqual((t.best_score, t.best_epoch), (0.70, 1))

    def test_min_delta_and_reset(self):
        t = BestTracker(patience=3, min_delta=0.01)
        t.update(0.70, 1)
        self.assertFalse(t.update(0.705, 2))
        self.assertTrue(t.update(0.72, 3))
        self.assertEqual(t.stale_epochs, 0)

    def test_rejects_nan(self):
        with self.assertRaises(ValueError):
            BestTracker(patience=1).update(float("nan"), 1)


class TestSubsample(unittest.TestCase):

    def test_every_disease_has_both_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = make_fake_metadata(Path(tmp) / "raw")
        small = subsample(df, n=5, seed=0)
        for d in DISEASES:
            self.assertGreater(small[d].sum(), 0)
            self.assertGreater((small[d] == 0).sum(), 0)
        self.assertFalse(small.index.duplicated().any())


# ---------------------------------------------------------------------------

class TestTrainFold(PipelineTestCase):

    def test_outputs_and_no_test_leak(self):
        result = self.train()
        out = fold_dir(self.cfg, 0)

        for name in (BEST_FILE, PREDICTION_FILE, HISTORY_FILE, RESULT_FILE):
            self.assertTrue((out / name).is_file(), name)
        self.assertFalse((out / LAST_FILE).exists())    # 완료 후 삭제

        self.assertEqual((result["n_train"], result["n_val"]), (32, 8))
        self.assertEqual(result["epochs_run"], 3)
        self.assertEqual(set(result["per_disease"]), set(DISEASES))

        preds = pd.read_csv(out / PREDICTION_FILE)
        test_ids = set(self.metadata.loc[self.metadata["Split"].eq("test"), "Patient ID"])
        self.assertEqual(len(preds), 8)
        self.assertTrue(set(preds["Patient ID"]).isdisjoint(test_ids))

        # best.pt는 weights_only로 안전하게 읽히고 같은 구조에 다시 불러올 수 있음
        ckpt = torch.load(out / BEST_FILE, map_location="cpu", weights_only=True)
        TinyNet().load_state_dict(ckpt["state_dict"])
        self.assertEqual(ckpt["epoch"], result["best_epoch"])

    def test_completed_fold_is_skipped(self):
        first = self.train()
        history = fold_dir(self.cfg, 0) / HISTORY_FILE
        mtime = history.stat().st_mtime_ns
        self.assertEqual(self.train(), first)
        self.assertEqual(history.stat().st_mtime_ns, mtime)

    def test_changed_config_goes_to_new_folder(self):
        self.train()
        other = TrainConfig(**{**self.cfg.to_dict(), "lr": 1e-3})
        self.assertNotEqual(fold_dir(self.cfg, 0), fold_dir(other, 0))
        self.assertEqual(self.train(other)["config"]["lr"], 1e-3)

    def test_resume_matches_uninterrupted_run(self):
        reference = self.train()
        ref_history = pd.read_csv(fold_dir(self.cfg, 0) / HISTORY_FILE)

        # 다른 폴더에서 2번째 Epoch 직후 중단
        cfg = TrainConfig(**{**self.cfg.to_dict(), "output_dir": str(self.root / "resume")})

        class Interrupt(Exception):
            pass

        def interrupt_at_epoch_2(msg):
            if "epoch 02" in msg:
                raise Interrupt

        with self.assertRaises(Interrupt):
            train_fold(cfg, 0, metadata=self.metadata, log=interrupt_at_epoch_2)
        self.assertTrue((fold_dir(cfg, 0) / LAST_FILE).is_file())

        resumed = self.train(cfg)
        history = pd.read_csv(fold_dir(cfg, 0) / HISTORY_FILE)

        self.assertEqual(list(history["epoch"]), [1, 2, 3])
        cols = ["train_loss", "val_loss", "val_mean_auroc"]
        np.testing.assert_allclose(history[cols], ref_history[cols], rtol=1e-5)
        self.assertAlmostEqual(resumed["mean_auroc"], reference["mean_auroc"], places=6)

        # 학습 시간에는 중단 전 Epoch도 포함
        self.assertGreaterEqual(resumed["seconds"], history["seconds"].sum() - 0.5)

    @unittest.skipUnless(
        torch.cuda.is_available() or torch.backends.mps.is_available(),
        "GPU/MPS 없음",
    )
    def test_resume_on_accelerator(self):
        """CPU가 아닌 장치에서도 체크포인트(RNG 상태 포함)를 복원할 수 있음"""
        device = "cuda" if torch.cuda.is_available() else "mps"
        cfg = TrainConfig(**{**self.cfg.to_dict(), "device": device, "amp": device == "cuda"})

        class Interrupt(Exception):
            pass

        def interrupt_at_epoch_1(msg):
            if "epoch 01" in msg:
                raise Interrupt

        with self.assertRaises(Interrupt):
            train_fold(cfg, 0, metadata=self.metadata, log=interrupt_at_epoch_1)

        resumed = self.train(cfg)
        self.assertEqual(resumed["epochs_run"], 3)
        self.assertEqual(resumed["device"].split(":")[0], device)

    def test_early_stopping(self):
        cfg = TrainConfig(**{**self.cfg.to_dict(), "epochs": 10, "lr": 1e-9,
                             "early_stop_patience": 1, "min_delta": 0.5})
        result = self.train(cfg)
        self.assertEqual(result["epochs_run"], 2)
        self.assertTrue(result["stopped_early"])

    def test_cosine_scheduler(self):
        self.train()
        lr = pd.read_csv(fold_dir(self.cfg, 0) / HISTORY_FILE)["lr"].to_numpy()
        # CosineAnnealingLR(T_max=3): lr0, lr0·(1+cos(π/3))/2, lr0·(1+cos(2π/3))/2
        expected = [self.cfg.lr * (1 + np.cos(np.pi * k / 3)) / 2 for k in range(3)]
        np.testing.assert_allclose(lr, expected, rtol=1e-6)


class TestBuildModel(unittest.TestCase):

    def test_models_output_14_logits(self):
        x = torch.randn(2, 3, 224, 224)
        for name in ("densenet121", "efficientnet_b0"):
            model = model_module.build_model(name, pretrained=False).eval()
            with torch.no_grad():
                self.assertEqual(tuple(model(x).shape), (2, len(DISEASES)), name)

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            model_module.build_model("resnet50", pretrained=False)


class TestSummary(PipelineTestCase):

    def test_summary_excludes_smoke(self):
        self.train(fold=0)
        self.train(fold=1)
        self.train(fold=2, limit=4)     # smoke → 제외

        smoke_result = fold_dir(self.cfg, 2, smoke=True) / RESULT_FILE
        self.assertTrue(json.loads(smoke_result.read_text())["smoke"])

        summary = write_summary(self.root / "outputs", self.root / "reports")
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.loc[0, "n_folds"], 2)
        self.assertIn("±", summary.loc[0, "mean_auroc"])

        per_disease = pd.read_csv(self.root / "reports" / "per_disease_auroc.csv")
        self.assertEqual(len(per_disease), len(DISEASES))

    def test_no_results(self):
        self.assertIsNone(write_summary(self.root / "outputs", self.root / "reports"))

    def test_gamma_comparison_and_selection(self):
        reports = self.root / "reports"
        quick = {**self.cfg.to_dict(), "epochs": 1}

        # gamma 0은 5-Fold 완료, gamma 1은 아직 3개 Fold → 선택 불가
        for fold in range(5):
            self.train(TrainConfig(**{**quick, "gamma": 0.0}), fold=fold)
        for fold in range(3):
            self.train(TrainConfig(**{**quick, "gamma": 1.0}), fold=fold)
        write_summary(self.root / "outputs", reports)
        self.assertEqual(json.loads((reports / "selected_gamma.json").read_text()), {})
        self.assertIn("아직 선택 불가", (reports / "gamma_comparison.md").read_text())

        # gamma 1도 5-Fold 완료 → 같은 비교 그룹에서 AUROC가 높은 gamma 선택
        for fold in range(3, 5):
            self.train(TrainConfig(**{**quick, "gamma": 1.0}), fold=fold)
        summary = write_summary(self.root / "outputs", reports)

        self.assertEqual(summary["comparison_group"].nunique(), 1)
        selected = json.loads((reports / "selected_gamma.json").read_text())
        (choice,) = selected.values()
        best = summary.sort_values("mean_auroc_mean", ascending=False).iloc[0]
        self.assertEqual(choice["gamma"], best["gamma"])
        self.assertEqual(choice["compared_gammas"], [0.0, 1.0])

        table = pd.read_csv(reports / "gamma_comparison.csv")
        self.assertEqual(list(table["gamma"]), [0.0, 1.0])
        self.assertIn("(BCE 동일)", (reports / "gamma_comparison.md").read_text())

    def test_other_settings_are_separate_groups(self):
        quick = {**self.cfg.to_dict(), "epochs": 1}
        self.train(TrainConfig(**{**quick, "gamma": 0.0}))
        self.train(TrainConfig(**{**quick, "gamma": 1.0, "use_clahe": False}))
        summary = write_summary(self.root / "outputs", self.root / "reports")
        self.assertEqual(summary["comparison_group"].nunique(), 2)



class TestTimeLimit(PipelineTestCase):

    def test_stops_before_epoch_that_cannot_finish(self):
        # 기록이 없으면 첫 Epoch를 first_epoch_estimate(기본 30분)로 추정 → 시작하지 않음
        with self.assertRaises(TimeLimitReached):
            self.train(deadline=time.time() + 60)
        self.assertFalse((fold_dir(self.cfg, 0) / HISTORY_FILE).exists())

    def test_pause_then_resume_to_completion(self):
        class Interrupt(Exception):
            pass

        def interrupt_at_epoch_1(msg):
            if "epoch 01" in msg:
                raise Interrupt

        with self.assertRaises(Interrupt):
            train_fold(self.cfg, 0, metadata=self.metadata, log=interrupt_at_epoch_1)

        # 직전 Epoch 시간으로 추정 → 남은 시간이 거의 없으면 last.pt를 그대로 두고 멈춤
        with self.assertRaises(TimeLimitReached):
            self.train(deadline=time.time() + 1e-3)
        self.assertTrue((fold_dir(self.cfg, 0) / LAST_FILE).is_file())

        result = self.train(deadline=time.time() + 3600)
        self.assertEqual(result["epochs_run"], 3)

    def test_run_main_returns_exit_code(self):
        metadata_path = self.root / "metadata.csv"
        self.metadata.to_csv(metadata_path, index=False)
        sets = []
        for key, value in [("model", "tiny"), ("pretrained", "false"), ("device", "cpu"),
                           ("num_workers", "0"), ("amp", "false"), ("epochs", "1"),
                           ("batch_size", "8"), ("metadata_path", str(metadata_path)),
                           ("raw_dir", str(self.root / "raw")),
                           ("output_dir", str(self.root / "outputs"))]:
            sets += ["--set", f"{key}={value}"]

        paused = run_module.main(["--folds", "0", "--no-summary", "--time-limit-hours", "0.01", *sets])
        self.assertEqual(paused, run_module.EXIT_TIME_LIMIT)
        self.assertEqual(run_module.main(["--folds", "0", "--no-summary", *sets]), 0)


class TestRunQueue(PipelineTestCase):
    """실제 서브프로세스를 띄워 2개 'GPU'에서 1단계(gamma 0/1 × 5-Fold) → 2단계(5-Fold)까지 실행"""

    def queue_args(self):
        metadata_path = self.root / "metadata.csv"
        self.metadata.to_csv(metadata_path, index=False)
        sets = ["model=tiny", "pretrained=false", "device=cpu", "num_workers=0", "amp=false",
                "epochs=1", "batch_size=8", "lr=0.01",
                f"metadata_path={metadata_path}", f"raw_dir={self.root / 'raw'}",
                f"output_dir={self.root / 'outputs'}"]
        self.sets = sets
        argv = ["--gpus", "a", "b", "--gammas", "0", "1", "--second-model", "tiny2",
                "--report-dir", str(self.root / "reports"), "--poll-seconds", "0.2"]
        for item in sets:
            argv += ["--set", item]
        return argv

    def test_full_queue(self):
        command = (sys.executable, "-m", "tests._tiny_run")
        self.assertEqual(run_queue.main(self.queue_args(), command=command), 0)

        selected = json.loads((self.root / "reports" / "selected_gamma.json").read_text())
        (choice,) = selected.values()
        # 2단계는 선택된 gamma로 실행됨 (큐와 같은 방식으로 설정을 만들어 확인)
        stage2 = run_queue.make_jobs(
            run_queue.DEFAULT_CONFIG_PATH, [*self.sets, "model=tiny2", f"gamma={choice['gamma']:g}"]
        )
        self.assertTrue(all(job.done for job in stage2))
        self.assertEqual(stage2[0].cfg.gamma, choice["gamma"])

        logs = list((self.root / "outputs" / "_logs").glob("*.log"))
        self.assertEqual(len(logs), 15)

        # 다시 실행하면 할 일이 없으므로 바로 완료
        self.assertEqual(run_queue.main(self.queue_args(), command=command), 0)
        self.assertEqual(len(list((self.root / "outputs" / "_logs").glob("*.log"))), 15)

    def test_gamma_override_is_rejected(self):
        with self.assertRaises(SystemExit):
            run_queue.parse_args(["--set", "gamma=1"])


if __name__ == "__main__":
    unittest.main()
