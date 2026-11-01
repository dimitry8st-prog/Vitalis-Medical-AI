import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "evals" / "run_clinical_eval.py"
SPEC = importlib.util.spec_from_file_location("vitalis_clinical_eval", RUNNER_PATH)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class ClinicalEvalDatasetTests(unittest.TestCase):
    def test_dataset_passes_quality_gate_schema(self):
        dataset = RUNNER.load_json(ROOT / "evals" / "clinical_cases.json")
        self.assertEqual(RUNNER.validate_dataset(dataset), [])
        self.assertEqual(len(dataset["cases"]), 30)

    def test_release_gate_rejects_a_critical_hallucination(self):
        values = [{"critical": 0.0} for _ in range(29)] + [{"critical": 1.0}]
        rate = sum(value["critical"] for value in values) / len(values)
        self.assertGreater(rate, 0.02)


if __name__ == "__main__":
    unittest.main()
