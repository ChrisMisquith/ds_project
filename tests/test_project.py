"""Behavioral checks for mistakes that could invalidate the experiment."""

import unittest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src.project import (
    InvalidMeasurementCleaner,
    calculate_metrics,
    make_model_pipeline,
    wilson_interval,
)


class EvaluationTests(unittest.TestCase):
    def test_confusion_matrix_orientation(self):
        # Deliberately unequal FP and FN: catches swapped recall/precision.
        actual = np.array([1] * 68 + [0] * 81)
        predicted = np.array([1] * 60 + [0] * 8 + [1] * 11 + [0] * 70)
        metrics = calculate_metrics(actual, predicted, predicted.astype(float))
        self.assertAlmostEqual(metrics["recall"], 60 / 68)
        self.assertAlmostEqual(metrics["precision"], 60 / 71)
        self.assertAlmostEqual(metrics["specificity"], 70 / 81)
        self.assertEqual(
            (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]),
            (70, 11, 8, 60),
        )

    def test_valid_zero_and_negative_oldpeak_are_retained(self):
        frame = pd.DataFrame(
            {
                "RestingBP": [0, 120],
                "Cholesterol": [0, 200],
                "FastingBS": [0, 1],
                "Oldpeak": [0, -1.2],
            }
        )
        clean = InvalidMeasurementCleaner().fit_transform(frame)
        self.assertTrue(pd.isna(clean.loc[0, "RestingBP"]))
        self.assertTrue(pd.isna(clean.loc[0, "Cholesterol"]))
        self.assertEqual(clean.loc[0, "FastingBS"], 0)
        self.assertEqual(clean.loc[0, "Oldpeak"], 0)
        self.assertEqual(clean.loc[1, "Oldpeak"], -1.2)
        self.assertEqual(frame.loc[0, "Cholesterol"], 0)  # Source is never mutated.

    def test_imputer_never_learns_from_test_data(self):
        train = pd.DataFrame(
            {"Cholesterol": [100.0, 200.0, 0.0, 300.0], "Sex": ["M", "F", "M", "F"]}
        )
        test = pd.DataFrame({"Cholesterol": [0.0, 9999.0], "Sex": ["F", "UNKNOWN"]})
        model = make_model_pipeline(
            DecisionTreeClassifier(random_state=42), features=list(train.columns)
        )
        model.fit(train, [0, 1, 0, 1])
        imputer = (
            model.named_steps["preprocess"]
            .named_transformers_["numeric"]
            .named_steps["impute"]
        )
        self.assertEqual(imputer.statistics_[0], 200.0)
        model.predict(test)  # Unknown category must not fail or refit the encoder.
        self.assertEqual(imputer.statistics_[0], 200.0)

    def test_zero_denominator_interval(self):
        low, high = wilson_interval(0, 0)
        self.assertTrue(np.isnan(low) and np.isnan(high))

    def test_interval_has_uncertainty_even_with_no_errors(self):
        low, high = wilson_interval(10, 10)
        self.assertLess(low, 1.0)
        self.assertAlmostEqual(high, 1.0)


if __name__ == "__main__":
    unittest.main()
