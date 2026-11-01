import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from app.rag.source_registry import (
    approved_for_ingestion,
    load_registry,
    validate_registry,
)


class SourceRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry(
            Path(__file__).resolve().parents[2] / "vault" / "source_registry.json"
        )

    def test_checked_candidates_are_valid_but_not_auto_ingested(self):
        self.assertEqual(validate_registry(self.registry), [])
        self.assertEqual(approved_for_ingestion(self.registry), [])

    def test_unapproved_source_cannot_enable_ingestion(self):
        registry = deepcopy(self.registry)
        registry["sources"][0]["ingest_allowed"] = True
        errors = validate_registry(registry)
        self.assertTrue(any("only approved" in error for error in errors))

    def test_lookalike_or_plain_http_domain_is_rejected(self):
        for url in (
            "http://www.who.int/guideline",
            "https://who.int.evil.example/guideline",
        ):
            registry = deepcopy(self.registry)
            registry["sources"][0]["url"] = url
            self.assertTrue(validate_registry(registry), url)

    def test_invalid_registry_file_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text("{}", encoding="utf-8")
            self.assertTrue(validate_registry(load_registry(path)))


if __name__ == "__main__":
    unittest.main()
