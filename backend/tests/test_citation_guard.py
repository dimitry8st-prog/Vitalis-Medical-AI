import unittest

from app.services.citation_guard import (
    UNSUPPORTED_DOSE_MARKER,
    UNVERIFIED_URL_MARKER,
    guard_clinical_payload,
)


class CitationGuardTests(unittest.TestCase):
    def test_keeps_only_urls_present_in_retrieved_context(self):
        payload = {
            "content": (
                "Источник: https://www.who.int/guide. "
                "Ложная ссылка: https://who.int.evil.example/guide"
            )
        }
        guarded = guard_clinical_payload(
            payload,
            web_hits=[{"url": "https://www.who.int/guide"}],
            rag_hits=[],
            strict=False,
        )
        self.assertIn("https://www.who.int/guide", guarded["content"])
        self.assertNotIn("who.int.evil.example", guarded["content"])
        self.assertIn(UNVERIFIED_URL_MARKER, guarded["content"])
        self.assertEqual(guarded["citation_validation"]["removed_urls"], 1)

    def test_hides_dose_without_full_text_support(self):
        guarded = guard_clinical_payload(
            {"content": "Назначить синтетический препарат 50 мг/сут."},
            web_hits=[
                {
                    "url": "https://www.who.int/guide",
                    "snippet": "50 мг/сут",
                    "full_text_checked": False,
                }
            ],
            rag_hits=[],
            strict=False,
        )
        self.assertIn(UNSUPPORTED_DOSE_MARKER, guarded["content"])
        self.assertNotIn("50 мг/сут", guarded["content"])

    def test_keeps_dose_explicitly_present_in_checked_excerpt(self):
        guarded = guard_clinical_payload(
            {"content": "Руководство указывает 50 мг/сут."},
            web_hits=[
                {
                    "url": "https://www.who.int/guide",
                    "evidence_excerpt": "В синтетическом примере указано 50 мг/сут.",
                    "full_text_checked": True,
                }
            ],
            rag_hits=[],
            strict=False,
        )
        self.assertIn("50 мг/сут", guarded["content"])
        self.assertEqual(guarded["citation_validation"]["supported_dose_statements"], 1)

    def test_strict_mode_blocks_unknown_citation(self):
        guarded = guard_clinical_payload(
            {"content": "Источник https://unknown.example/claim"},
            web_hits=[],
            rag_hits=[],
            strict=True,
        )
        self.assertTrue(guarded["citation_validation"]["blocked"])
        self.assertIn("Ответ заблокирован", guarded["content"])

    def test_strict_mode_blocks_answer_without_available_source_url(self):
        guarded = guard_clinical_payload(
            {"content": "Синтетическое утверждение без ссылки."},
            web_hits=[{"url": "https://www.who.int/guide"}],
            rag_hits=[],
            strict=True,
        )
        self.assertTrue(guarded["citation_validation"]["missing_required_citation"])
        self.assertTrue(guarded["citation_validation"]["blocked"])

    def test_metadata_only_rag_url_does_not_validate_clinical_citation(self):
        url = "https://cr.minzdrav.gov.ru/view-cr/295_4"
        guarded = guard_clinical_payload(
            {"content": f"Утверждение: {url}"},
            web_hits=[],
            rag_hits=[
                {
                    "meta": {
                        "url": url,
                        "content_scope": "bibliographic_metadata_only",
                    }
                }
            ],
            strict=False,
        )
        self.assertIn(UNVERIFIED_URL_MARKER, guarded["content"])
        self.assertNotIn(url, guarded["content"])

    def test_unfetched_registry_seed_does_not_validate_clinical_citation(self):
        url = "https://cr.minzdrav.gov.ru/view-cr/523_3"
        guarded = guard_clinical_payload(
            {"content": f"Утверждение: {url}"},
            web_hits=[
                {
                    "url": url,
                    "content_scope": "bibliographic_metadata_only",
                    "full_text_checked": False,
                }
            ],
            rag_hits=[],
            strict=False,
        )
        self.assertIn(UNVERIFIED_URL_MARKER, guarded["content"])


if __name__ == "__main__":
    unittest.main()
