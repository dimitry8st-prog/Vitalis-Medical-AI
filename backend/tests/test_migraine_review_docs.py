"""Clinical review materials stay outside the automatic Vitalis vault/RAG."""

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
VITALIS = ROOT / "docs" / "clinical_review" / "migraine_first_visit_vikelis_2026.md"
GEM = ROOT / "docs" / "gems" / "GEM_MIGRAINE_FIRST_VISIT_2026.md"
VAULT = ROOT / "vault"


class MigraineReviewDocsTests(unittest.TestCase):
    def test_clinical_checklist_is_review_candidate_outside_vault(self):
        text = VITALIS.read_text(encoding="utf-8")
        self.assertIn("needs_review", text)
        self.assertIn("не активирован в RAG", text)
        self.assertIn("10.1177/25158163261507712", text)
        self.assertIn("не клинические рекомендации", text.lower())
        self.assertIn("## 1.", text)
        self.assertIn("## 8.", text)
        self.assertFalse(VITALIS.is_relative_to(VAULT))
        self.assertFalse(any(path.name == VITALIS.name for path in VAULT.rglob("*.md")))

    def test_gem_instructions_prohibit_unverified_prescriptions(self):
        text = GEM.read_text(encoding="utf-8")
        self.assertIn("10.1177/25158163261507712", text)
        self.assertIn("без автономной диагностики", text)
        self.assertIn("не подбирай индивидуальную терапию", text)
        self.assertIn("РФ/INT", text)
        self.assertFalse(GEM.is_relative_to(VAULT))


if __name__ == "__main__":
    unittest.main()
