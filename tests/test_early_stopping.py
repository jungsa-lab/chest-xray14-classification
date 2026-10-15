import unittest

from src.training.early_stopping import EarlyStopping

class TestEarlyStopping(unittest.TestCase):

    def test_first_score_sets_best(self):   # 첫 번째 점수를 최고 점수로 저장하는지 확인
        stopper = EarlyStopping(patience=5)

        self.assertFalse(stopper.step(0.70))
        self.assertTrue(stopper.improved)
        self.assertEqual(stopper.best_score, 0.70)
        self.assertEqual(stopper.bad_epochs, 0)

    def test_stops_after_five_epochs_without_improvement(self): # 연속 5회 성능 미개선에 따른 종료 확인
        stopper = EarlyStopping(patience=5)
        stopper.step(0.75)

        for _ in range(4):
            self.assertFalse(stopper.step(0.74))

        self.assertTrue(stopper.step(0.74))
        self.assertTrue(stopper.should_stop)
        self.assertEqual(stopper.bad_epochs, 5)

    def test_improvement_resets_counter(self):  # 성능 개선 시 미개선 횟수가 0으로 초기화되는지 확인
        stopper = EarlyStopping(patience=2)
        stopper.step(0.70)
        stopper.step(0.69)

        self.assertEqual(stopper.bad_epochs, 1)

        self.assertFalse(stopper.step(0.72)) # step() 리턴값이 False면 계속 진행(should_stop = False)
        self.assertTrue(stopper.improved)
        self.assertEqual(stopper.bad_epochs, 0)
        self.assertEqual(stopper.best_score, 0.72)

    def test_equal_score_is_not_improvement(self):  # 동일한 점수는 성능 개선으로 인정하지 않는지 확인
        stopper = EarlyStopping(patience=1)
        stopper.step(0.70)

        self.assertTrue(stopper.step(0.70))
        self.assertFalse(stopper.improved)  # improved가 False인지 확인

    def test_min_delta(self):   # min_delta를 초과해야 성능 개선으로 인정하는지 확인
        stopper = EarlyStopping(
            patience=2,
            min_delta=0.01,
        )

        stopper.step(0.70)

        self.assertFalse(stopper.step(0.705))
        self.assertFalse(stopper.improved)
        self.assertEqual(stopper.bad_epochs, 1)

        self.assertFalse(stopper.step(0.72))
        self.assertTrue(stopper.improved)
        self.assertEqual(stopper.bad_epochs, 0)



    def test_invalid_configuration(self):   # 잘못된 설정값 입력 시 ValueError가 발생하는지 확인
        for value in [0, -1, True, 1.5]:
            with self.subTest(patience=value):
                with self.assertRaises(ValueError):
                    EarlyStopping(patience=value)

        for value in [
            -0.1,
            float("nan"),
            float("inf"),
        ]:
            with self.subTest(min_delta=value):
                with self.assertRaises(ValueError):
                    EarlyStopping(min_delta=value)

    def test_invalid_scores(self):  # 유효하지 않은 AUROC 입력 시 ValueError가 발생하는지 확인
        stopper = EarlyStopping()

        for value in [
            -0.1,
            1.1,
            float("nan"),
            float("inf"),
        ]:
            with self.subTest(score=value):
                with self.assertRaises(ValueError):
                    stopper.step(value)

    def test_does_not_resume_after_stopping(self):  # 학습 중단 이후에는 종료 상태가 유지되는지 확인
        stopper = EarlyStopping(patience=1)

        stopper.step(0.7)
        stopper.step(0.6)

        self.assertTrue(stopper.step(0.8))
        self.assertFalse(stopper.improved)
        self.assertEqual(stopper.best_score, 0.7)


if __name__ == "__main__":
    unittest.main()