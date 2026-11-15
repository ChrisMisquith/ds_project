import unittest
from src.feature_selection import choose_budget, model_for
from sklearn.linear_model import LogisticRegression


class FeatureSelectionTests(unittest.TestCase):
    def test_smallest_eligible_budget(self):
        self.assertEqual(choose_budget({3: 0.85, 5: 0.895, 7: 0.902, 11: 0.90}), 5)

    def test_full_model_fallback(self):
        self.assertEqual(choose_budget({3: 0.7, 5: 0.8, 7: 0.85, 11: 0.9}), 11)

    def test_group_is_original_variable(self):
        model = model_for(["RestingECG", "Age"])
        transformer = model.named_steps["preprocess"]
        self.assertEqual(transformer.transformers[1][2], ["RestingECG"])
        self.assertIsInstance(model.named_steps["model"], LogisticRegression)
