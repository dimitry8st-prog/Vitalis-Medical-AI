import unittest

from app.llm.deepseek import DeepSeekClient
from app.rag.web_search import (
    _balanced_results,
    _build_queries,
    _detect_intent,
    _english_variant,
    _extract_focus,
    _rf_search_focus,
)


class ClinicalSearchPlanTests(unittest.TestCase):
    def test_extracts_question_focus(self) -> None:
        self.assertEqual(
            _extract_focus(
                "Как лечить инсульт ишемический по актуальным рекомендациям?"
            ),
            "инсульт ишемический",
        )

    def test_translates_specific_nosology_without_duplicate_generic_term(self) -> None:
        self.assertEqual(
            _english_variant("ишемический инсульт"),
            "ischemic stroke",
        )

    def test_detects_treatment_intent(self) -> None:
        self.assertEqual(
            _detect_intent("Как лечить ишемический инсульт?"),
            ("лечение", "treatment"),
        )

    def test_rf_plan_contains_only_rf_queries_and_profiled_sources(self) -> None:
        queries = _build_queries(
            "Как лечить ишемический инсульт?",
            bilingual=False,
            freshness_extra=True,
            scope="rf",
            en_focus="ischemic stroke",
        )
        self.assertTrue(queries)
        self.assertTrue(all(lang == "ru" for _, lang, _ in queries))
        self.assertTrue(any("cr.minzdrav.gov.ru" in q for q, _, _ in queries))
        self.assertTrue(any("НМИЦ" in q for q, _, _ in queries))
        self.assertTrue(any("журнал" in q for q, _, _ in queries))
        self.assertTrue(any(kind == "rf_society" for _, _, kind in queries))
        self.assertTrue(all('"' not in q for q, _, _ in queries))

    def test_rf_focus_keeps_clinical_keywords_and_removes_search_noise(self) -> None:
        focus = _rf_search_focus(
            "Какие актуальные российские рекомендации по лечению ишемического инсульта?"
        )
        self.assertIn("ишемического", focus)
        self.assertIn("инсульта", focus)
        self.assertNotIn("актуальные", focus)
        self.assertNotIn("рекомендации", focus)

    def test_rf_results_keep_source_type_coverage_and_reject_intl_domains(self) -> None:
        hits = [
            {"url": "https://cr.minzdrav.gov.ru/schema/1", "lang": "ru", "query_kind": "rf_registry", "relevance_score": 4, "year": 2025},
            {"url": "https://nmic.example.ru/article", "lang": "ru", "query_kind": "rf_institute", "relevance_score": 3, "year": 2026},
            {"url": "https://journal.example.ru/review", "lang": "ru", "query_kind": "rf_journal", "relevance_score": 2, "year": 2026},
            {"url": "https://who.int/guideline", "lang": "ru", "query_kind": "rf_guideline", "relevance_score": 20, "year": 2026},
        ]
        selected = _balanced_results(hits, "rf", 2)
        self.assertTrue(any(item["query_kind"] == "rf_registry" for item in selected))
        self.assertTrue(any(item["query_kind"] == "rf_institute" for item in selected))
        self.assertTrue(any(item["query_kind"] == "rf_journal" for item in selected))
        self.assertFalse(any("who.int" in item["url"] for item in selected))

    def test_rf_mock_does_not_return_international_recommendations(self) -> None:
        content = DeepSeekClient._mock("российский контур", "тестовый запрос")
        self.assertIn("Российский контур", content)
        self.assertNotIn("ESC", content)
        self.assertNotIn("AHA", content)

    def test_international_plan_contains_guidelines_and_literature(self) -> None:
        queries = _build_queries(
            "Как лечить ишемический инсульт?",
            bilingual=True,
            freshness_extra=True,
            scope="intl",
            en_focus="ischemic stroke",
        )
        self.assertTrue(queries)
        self.assertTrue(all(lang == "en" for _, lang, _ in queries))
        self.assertTrue(any("WHO NICE" in q for q, _, _ in queries))
        self.assertTrue(any("NIH CDC" in q for q, _, _ in queries))
        self.assertTrue(any("systematic review" in q for q, _, _ in queries))


if __name__ == "__main__":
    unittest.main()
