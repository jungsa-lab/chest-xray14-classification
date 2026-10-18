import unittest

import numpy as np
import pandas as pd

from src.analysis.measure_clahe_bbox import (
    entropy,
    lesion_masks,
    lesion_visibility,
    scale_box,
    summarize_change,
)
from src.analysis.plot_conditional_cooccurrence import (
    conditional_cooccurrence,
    single_label_rates,
    top_pairs,
)
from src.analysis.summarize_prevalence import LABELS, prevalence_table


class TestPrevalenceTable(unittest.TestCase):

    def test_image_and_patient_level(self):
        df = pd.DataFrame(0, index=range(4), columns=LABELS)
        df["Patient ID"] = [1, 1, 2, 3]
        df.loc[[0, 1], "Nodule"] = 1
        df.loc[[2, 3], "No Finding"] = 1

        table = prevalence_table(df).set_index("Disease")

        self.assertEqual(table.loc["Nodule", "Positive Images"], 2)
        self.assertAlmostEqual(table.loc["Nodule", "Image Prevalence (%)"], 50.0)
        # 환자 1의 두 이미지는 한 명으로 센다
        self.assertEqual(table.loc["Nodule", "Positive Patients"], 1)
        self.assertAlmostEqual(table.loc["Nodule", "Patient Prevalence (%)"], 100 / 3)
        self.assertEqual(table.loc["No Finding", "Positive Patients"], 2)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            prevalence_table(pd.DataFrame(columns=[*LABELS, "Patient ID"]))


class TestConditionalCooccurrence(unittest.TestCase):

    def setUp(self):
        # 열: A, B, C
        self.labels = np.array([
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [0, 0, 0],
        ])
        self.names = ["A", "B", "C"]

    def test_rates(self):
        rates = conditional_cooccurrence(self.labels)

        self.assertAlmostEqual(rates[0, 1], 50.0)   # A 2장 중 1장에 B
        self.assertAlmostEqual(rates[2, 1], 100.0)  # C 1장 모두 B
        np.testing.assert_allclose(np.diag(rates), 100.0)

    def test_zero_positive_row_is_nan(self):
        labels = np.array([[1, 0], [1, 0]])
        rates = conditional_cooccurrence(labels)
        self.assertTrue(np.isnan(rates[1]).all())

    def test_top_pairs_both_directions(self):
        pairs = top_pairs(self.labels, self.names, n=1)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.loc[0, "Images"], 1)

    def test_single_label_rates(self):
        single = single_label_rates(self.labels, self.names).set_index("Disease")

        self.assertAlmostEqual(single.loc["A", "Single-label Rate (%)"], 50.0)
        self.assertAlmostEqual(single.loc["C", "Single-label Rate (%)"], 0.0)


class TestLesionVisibility(unittest.TestCase):

    def test_masks_are_disjoint(self):
        inside, ring = lesion_masks((50, 50), 20, 20, 30, 30)

        self.assertEqual(inside.sum(), 100)
        self.assertFalse((inside & ring).any())
        self.assertGreater(ring.sum(), 0)

    def test_bright_square_on_flat_background(self):
        pixels = np.full((60, 60), 100, dtype=np.uint8)
        pixels[25:35, 25:35] = 160

        metrics = lesion_visibility(pixels, 25, 25, 35, 35)

        self.assertAlmostEqual(metrics["Contrast"], 60.0)
        self.assertEqual(metrics["Lesion Std"], 0.0)
        self.assertEqual(metrics["Lesion Entropy"], 0.0)

    def test_noise_lowers_cnr(self):
        rng = np.random.default_rng(0)
        clean = np.full((60, 60), 100, dtype=np.uint8)
        clean[25:35, 25:35] = 140
        noisy = np.clip(clean + rng.normal(0, 15, clean.shape), 0, 255).astype(np.uint8)

        self.assertGreater(
            lesion_visibility(clean, 25, 25, 35, 35)["CNR"],
            lesion_visibility(noisy, 25, 25, 35, 35)["CNR"],
        )

    def test_entropy(self):
        self.assertEqual(entropy(np.zeros(8, dtype=np.uint8)), 0.0)
        self.assertAlmostEqual(entropy(np.array([0, 255], dtype=np.uint8)), 1.0)

    def test_scale_box_from_1024(self):
        box = scale_box(512, 256, 128, 64, 224, 224)
        np.testing.assert_allclose(box, (112, 56, 140, 70))

    def test_scale_box_clips_to_image(self):
        box = scale_box(1000, 0, 100, 50, 224, 224)
        self.assertEqual(box[2], 224)


class TestSummarizeChange(unittest.TestCase):

    def test_median_change_and_improved_share(self):
        rows = []
        for box, (base, boosted) in enumerate([(10, 15), (10, 8)]):
            for clip in [0.0, 1.0, 2.0, 3.0, 4.0]:
                value = base if clip == 0 else boosted
                rows.append({
                    "Box ID": box, "Finding Label": "Nodule", "Clip Limit": clip,
                    "Contrast": value, "CNR": value, "Lesion Std": value, "Lesion Entropy": value,
                })

        summary = summarize_change(pd.DataFrame(rows), ["Finding Label"])
        clip2 = summary.loc[summary["Clip Limit"].eq(2.0)].iloc[0]

        self.assertEqual(clip2["Boxes"], 2)
        self.assertAlmostEqual(clip2["Contrast Median Change (%)"], 15.0)  # median(+50, -20)
        self.assertAlmostEqual(clip2["Contrast Improved Boxes (%)"], 50.0)


if __name__ == "__main__":
    unittest.main()
