import json
import tempfile
import unittest
from pathlib import Path

from app.rag.portal_guidelines import (
    CONTENT_SCOPE,
    PortalGuidelineError,
    select_portal_guidelines,
    sync_portal_guidelines,
)


def _item(external_id: str, title: str, tags: list[str], url: str | None = None):
    return {
        "source_key": "minzdrav",
        "external_id": external_id,
        "title": title,
        "organization": "Профильная медицинская организация",
        "version": "2026",
        "codes": "G43.0, G43.1",
        "specialties": "Неврология",
        "specialty_tags": tags,
        "url": url or f"https://cr.minzdrav.gov.ru/view-cr/{external_id}",
        "published_on": "2026-10-25",
        "status": "active",
    }


class PortalGuidelineImportTests(unittest.TestCase):
    def test_imports_only_target_profiles_and_marks_metadata_scope(self):
        portal = {
            "schema_version": 1,
            "items": [
                _item("295_4", "Мигрень", ["neurology"]),
                _item("732_2", "Очаговая травма", ["neurosurgery"]),
                _item("87_5", "Остеопороз", []),
            ],
        }
        source_registry = {
            "schema_version": 1,
            "updated_at": "2026-10-01",
            "default_ingestion_policy": "deny",
            "sources": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            portal_path = root / "portal.json"
            source_path = root / "source_registry.json"
            vault = root / "vault"
            portal_path.write_text(json.dumps(portal), encoding="utf-8")
            source_path.write_text(json.dumps(source_registry), encoding="utf-8")

            stats = sync_portal_guidelines(
                portal_path,
                vault,
                source_path,
                reviewed_at="2026-11-01",
                reviewed_by="Владелец Vitalis (метаданные)",
            )

            self.assertEqual(stats["written"], 2)
            files = sorted((vault / "guidelines" / "rf").glob("*.md"))
            self.assertEqual([path.name for path in files], [
                "minzdrav-295_4.md",
                "minzdrav-732_2.md",
            ])
            text = files[0].read_text(encoding="utf-8")
            self.assertIn(f'content_scope: "{CONTENT_SCOPE}"', text)
            self.assertIn("не является источником утверждений", text)
            registry = json.loads(source_path.read_text(encoding="utf-8"))
            self.assertEqual(len(registry["sources"]), 2)
            self.assertTrue(all(row["ingest_allowed"] for row in registry["sources"]))
            self.assertTrue(
                all(row["content_scope"] == CONTENT_SCOPE for row in registry["sources"])
            )

    def test_rejects_lookalike_http_or_mismatched_official_url(self):
        for url in (
            "http://cr.minzdrav.gov.ru/view-cr/295_4",
            "https://cr.minzdrav.gov.ru.evil.example/view-cr/295_4",
            "https://cr.minzdrav.gov.ru/view-cr/999_1",
        ):
            with self.subTest(url=url), self.assertRaises(PortalGuidelineError):
                select_portal_guidelines(
                    {
                        "schema_version": 1,
                        "items": [_item("295_4", "Мигрень", ["neurology"], url)],
                    }
                )


if __name__ == "__main__":
    unittest.main()
