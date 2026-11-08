import json
import tempfile
import unittest
from pathlib import Path

from scripts.production_gate import run_gate


class ProductionGateTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        vault = root / "vault"
        vault.mkdir()
        registry = {
            "schema_version": 1,
            "default_ingestion_policy": "deny",
            "sources": [
                {
                    "id": "synthetic-registry-source",
                    "title": "Synthetic metadata",
                    "url": "https://cr.minzdrav.gov.ru/view-cr/295_4",
                    "organization": "Минздрав России",
                    "jurisdiction": "RU",
                    "language": "ru",
                    "document_type": "clinical_guideline_metadata",
                    "content_scope": "bibliographic_metadata_only",
                    "publication_year": 2026,
                    "source_version": "295_4",
                    "checked_at": "2026-11-07",
                    "status": "approved",
                    "ingest_allowed": True,
                    "topics": ["neurology"],
                    "verification_level": "official_registry_metadata",
                    "review_notes": "Synthetic test record.",
                    "reviewed_at": "2026-11-07",
                    "reviewed_by": "Test reviewer",
                }
            ],
        }
        (vault / "source_registry.json").write_text(
            json.dumps(registry), encoding="utf-8"
        )
        generated = vault / "generated/guidelines/rf"
        generated.mkdir(parents=True)
        (generated / "minzdrav-full-295_4.md").write_text(
            "synthetic", encoding="utf-8"
        )
        manifest = {
            "sync_status": "complete",
            "failures": [],
            "records": [
                {
                    "external_id": "295_4",
                    "content_scope": "official_full_text_extracted",
                    "sha256": "a" * 64,
                    "page_count": 92,
                    "character_count": 100_000,
                    "output": "minzdrav-full-295_4.md",
                    "authorized_by": "Test owner",
                    "authorized_at": "2026-11-07",
                    "source_url": "https://cr.minzdrav.gov.ru/view-cr/295_4",
                    "document_url": "https://apicr.minzdrav.gov.ru/API.ashx?op=GetClinrecPdf&id=295_4",
                }
            ],
        }
        (generated / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        env = root / "production.env"
        env.write_text(
            "LLM_MODE=live\n"
            "EVIDENCE_GATE_MODE=strict\n"
            "DOCUMENT_EXTERNAL_LLM_ENABLED=false\n"
            "GIGACHAT_ENABLED=false\n"
            "DOCUMENT_REQUIRE_DEIDENTIFIED_CONFIRMATION=true\n"
            "DEEPSEEK_API_KEY=synthetic-test-value\n"
            "CORS_ORIGINS=https://vitalis.example\n",
            encoding="utf-8",
        )
        report = root / "clinical-report.json"
        report.write_text(
            json.dumps({"variants": {"candidate": {"release_gate_passed": True}}}),
            encoding="utf-8",
        )
        return vault, env, report

    def test_complete_private_pilot_fixture_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            vault, env, report = self._fixture(Path(directory))
            errors = run_gate(
                vault=vault,
                env_file=env,
                clinical_report=report,
                variant="candidate",
                expected_fulltexts=1,
                lifeos_manifest=None,
            )
            self.assertEqual(errors, [])

    def test_missing_clinical_approval_and_strict_mode_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault, env, report = self._fixture(root)
            env.write_text("LLM_MODE=live\nDEEPSEEK_API_KEY=test\n", encoding="utf-8")
            report.write_text(
                json.dumps(
                    {"variants": {"candidate": {"release_gate_passed": False}}}
                ),
                encoding="utf-8",
            )
            errors = run_gate(
                vault=vault,
                env_file=env,
                clinical_report=report,
                variant="candidate",
                expected_fulltexts=1,
                lifeos_manifest=None,
            )
            self.assertTrue(any("EVIDENCE_GATE_MODE" in error for error in errors))
            self.assertTrue(any("release_gate_passed" in error for error in errors))

    def test_lifeos_article_import_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault, env, report = self._fixture(root)
            lifeos_dir = vault / "generated/lifeos/portal"
            lifeos_dir.mkdir(parents=True)
            manifest_path = lifeos_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "automatic_ingestion": False,
                        "imported": 1,
                        "rejected": [{"record": "bad", "reason": "synthetic"}],
                        "records": [
                            {
                                "source_id": "portal-article-test",
                                "content_scope": "reviewed_excerpt",
                                "reviewed_by": "Test reviewer",
                                "reviewed_at": "2026-11-07T10:00:00+00:00",
                                "source_sha256": "a" * 64,
                                "source_url": "https://st8dom.ru/materials/test/",
                                "output": "portal-article-test.md",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            errors = run_gate(
                vault=vault,
                env_file=env,
                clinical_report=report,
                variant="candidate",
                expected_fulltexts=1,
                lifeos_manifest=manifest_path,
            )
            self.assertTrue(
                any("статьи портала не допускаются" in error for error in errors)
            )

    def test_lifeos_candidate_inventory_does_not_change_candidate_statuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vault, env, report = self._fixture(root)
            manifest_path = root / "lifeos-inventory.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "automatic_ingestion": False,
                        "imported": 0,
                        "records": [],
                        "candidates": [
                            {
                                "source_id": "lifeos-material-1",
                                "review_status": "needs_medical_review",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            errors = run_gate(
                vault=vault,
                env_file=env,
                clinical_report=report,
                variant="candidate",
                expected_fulltexts=1,
                lifeos_manifest=manifest_path,
            )
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
