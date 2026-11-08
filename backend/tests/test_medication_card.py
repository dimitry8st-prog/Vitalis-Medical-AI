import unittest
from datetime import date, timedelta

from app.services.medication_card import build_medication_card


class MedicationCardTests(unittest.TestCase):
    def _rag_hit(self, **meta_overrides):
        today = date(2026, 11, 8)
        meta = {
            "source_url": "https://cr.minzdrav.gov.ru/view-cr/295_4",
            "content_scope": "official_full_text_extracted",
            "full_text_checked": "true",
            "language": "ru",
            "jurisdiction": "RU",
            "year": "2026",
            "source_version": "2026.1",
            "guideline_status": "active",
            "current_version_checked_at": today.isoformat(),
        }
        meta.update(meta_overrides)
        return {
            "title": "Синтетическая клиническая рекомендация",
            "text": (
                "[Страница 42] Синтетический препарат 50 мг внутрь 1 раз в сутки "
                "в течение 5 дней."
            ),
            "year": 2026,
            "stale": False,
            "meta": meta,
        }

    def test_builds_literal_regimen_with_provenance(self):
        card = build_medication_card(
            [self._rag_hit()],
            [],
            lang="ru",
            as_of=date(2026, 11, 8),
        )

        self.assertEqual(card["status"], "verified")
        self.assertEqual(len(card["items"]), 1)
        item = card["items"][0]
        self.assertEqual(item["doses"], ["50 мг"])
        self.assertEqual(item["routes"], ["внутрь"])
        self.assertEqual(item["page"], 42)
        self.assertEqual(item["source_version"], "2026.1")
        self.assertTrue(item["literal_source_match"])
        self.assertIn("50 мг внутрь", item["evidence_quote"])

    def test_hides_dose_when_current_version_status_is_missing(self):
        card = build_medication_card(
            [self._rag_hit(guideline_status="")],
            [],
            lang="ru",
            as_of=date(2026, 11, 8),
        )

        self.assertEqual(card["status"], "blocked_unconfirmed_current_version")
        self.assertEqual(card["items"], [])
        self.assertEqual(card["blocked_statements"], 1)

    def test_hides_dose_when_version_check_is_expired(self):
        old_check = date(2026, 11, 8) - timedelta(days=46)
        card = build_medication_card(
            [self._rag_hit(current_version_checked_at=old_check.isoformat())],
            [],
            lang="ru",
            as_of=date(2026, 11, 8),
        )

        self.assertEqual(card["status"], "blocked_unconfirmed_current_version")
        self.assertEqual(card["items"], [])

    def test_old_publication_is_allowed_when_active_status_was_rechecked(self):
        hit = self._rag_hit(year="2018")
        hit["year"] = 2018
        hit["stale"] = True
        card = build_medication_card(
            [hit],
            [],
            lang="ru",
            as_of=date(2026, 11, 8),
        )

        self.assertEqual(card["status"], "verified")
        self.assertEqual(card["items"][0]["doses"], ["50 мг"])

    def test_rejects_metadata_card_even_if_it_contains_a_dose(self):
        card = build_medication_card(
            [
                self._rag_hit(
                    content_scope="bibliographic_metadata_only",
                    full_text_checked="false",
                )
            ],
            [],
            lang="ru",
            as_of=date(2026, 11, 8),
        )

        self.assertEqual(card["status"], "not_found")
        self.assertEqual(card["items"], [])

    def test_web_excerpt_also_requires_confirmed_current_version(self):
        hit = {
            "url": "https://www.nice.org.uk/guidance/example",
            "title": "Synthetic guidance",
            "lang": "en",
            "year": 2026,
            "jurisdiction": "UK",
            "full_text_checked": True,
            "guideline_status": "current",
            "source_version": "v2",
            "current_version_checked_at": "2026-11-08",
            "evidence_excerpt": "Use synthetic medicine 25 mg orally once daily.",
        }
        card = build_medication_card(
            [],
            [hit],
            lang="en",
            as_of=date(2026, 11, 8),
        )

        self.assertEqual(card["status"], "verified")
        self.assertEqual(card["items"][0]["doses"], ["25 mg"])

    def test_keeps_weight_based_daily_dose_literal(self):
        hit = self._rag_hit()
        hit["text"] = (
            "[Страница 17] Синтетический препарат 0,5–1 мг/кг/сут "
            "внутривенно в течение 3 дней."
        )
        card = build_medication_card(
            [hit],
            [],
            lang="ru",
            as_of=date(2026, 11, 8),
        )

        self.assertEqual(card["status"], "verified")
        self.assertEqual(card["items"][0]["doses"], ["0,5–1 мг/кг/сут"])
        self.assertEqual(card["items"][0]["routes"], ["внутривенно"])


if __name__ == "__main__":
    unittest.main()
