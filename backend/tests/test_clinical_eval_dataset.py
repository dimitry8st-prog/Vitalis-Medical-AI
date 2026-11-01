import importlib.util
import json
import tempfile
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

    def test_remote_collection_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "--allow-remote-api"):
            RUNNER.normalize_api_url("https://example.org/api")
        self.assertEqual(
            RUNNER.normalize_api_url(
                "https://example.org/api/", allow_remote=True
            ),
            "https://example.org/api",
        )
        with self.assertRaisesRegex(ValueError, "credentials"):
            RUNNER.normalize_api_url(
                "https://user:secret@example.org/api", allow_remote=True
            )
        with self.assertRaisesRegex(ValueError, "must use https"):
            RUNNER.normalize_api_url("http://example.org/api", allow_remote=True)

    def test_collection_writes_versioned_checkpoint(self):
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {"question": "synthetic", "search": {"routed": "clinical"}}
                ).encode()

        requests = []

        def opener(request, timeout):
            requests.append((request, timeout))
            return FakeResponse()

        ticks = iter((10.0, 10.125))
        dataset = {
            "dataset_version": "test-1",
            "data_policy": "synthetic_only",
            "cases": [
                {
                    "id": "case-1",
                    "question": "Синтетический вопрос",
                    "mode": "rf",
                    "use_web": False,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.json"
            bundle = RUNNER.collect_responses(
                dataset,
                output,
                api_url="http://127.0.0.1:8000",
                metadata={
                    "model": "snapshot",
                    "prompt_version": "prompt-sha",
                    "knowledge_version": "vault-sha",
                    "tool_version": "tool-sha",
                },
                timeout=20,
                delay_seconds=0,
                resume=False,
                opener=opener,
                clock=lambda: next(ticks),
                sleeper=lambda _seconds: None,
            )
            stored = RUNNER.load_json(output)

        self.assertTrue(bundle["complete"])
        self.assertTrue(stored["complete"])
        self.assertEqual(stored["responses"][0]["latency_ms"], 125.0)
        self.assertEqual(stored["metadata"]["data_policy"], "synthetic_only")
        request, timeout = requests[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8000/ask")
        self.assertEqual(timeout, 20)
        self.assertNotIn("sensitive_tokens", json.loads(request.data))


if __name__ == "__main__":
    unittest.main()
