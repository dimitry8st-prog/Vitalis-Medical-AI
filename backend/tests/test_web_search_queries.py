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
from app.rag.source_policy import classify_international_source
from app.rag.document_fetch import select_relevant_excerpt
from app.services.ask import AskService


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
        self.assertTrue(any("site:who.int" in q for q, _, _ in queries))
        self.assertTrue(any("site:nice.org.uk" in q for q, _, _ in queries))
        self.assertTrue(any("NIH CDC" in q for q, _, _ in queries))
        self.assertTrue(any("Cochrane" in q for q, _, _ in queries))

    def test_drug_question_adds_regulatory_sources(self) -> None:
        queries = _build_queries(
            "Дозировка цитиколина после ишемического инсульта",
            bilingual=True,
            freshness_extra=False,
            scope="intl",
            en_focus="citicoline ischemic stroke",
        )
        self.assertTrue(any("fda.gov" in q or "dailymed" in q for q, _, _ in queries))
        self.assertTrue(any("ema.europa.eu" in q or "bnf.nice" in q for q, _, _ in queries))

    def test_international_source_policy_tracks_jurisdiction(self) -> None:
        eso = classify_international_source("https://eso-stroke.org/guidelines/example")
        self.assertEqual(eso["source_tier"], "official_guideline")
        self.assertEqual(eso["jurisdiction"], "EU")
        self.assertTrue(eso["official_domain"])

    def test_intl_balancer_rejects_unknown_and_russian_domains(self) -> None:
        hits = [
            {"url": "https://who.int/guideline", "lang": "en", "query_kind": "intl_who", "relevance_score": 12, "year": 2025},
            {"url": "https://pubmed.ncbi.nlm.nih.gov/1/", "lang": "en", "query_kind": "pubmed", "relevance_score": 8, "year": 2024},
            {"url": "https://cr.minzdrav.gov.ru/schema/1", "lang": "en", "query_kind": "intl_guideline", "relevance_score": 50, "year": 2026},
            {"url": "https://random-blog.example/guideline", "lang": "en", "query_kind": "intl_guideline", "relevance_score": 100, "year": 2026},
        ]
        selected = _balanced_results(hits, "intl", 4)
        urls = {item["url"] for item in selected}
        self.assertIn("https://who.int/guideline", urls)
        self.assertIn("https://pubmed.ncbi.nlm.nih.gov/1/", urls)
        self.assertNotIn("https://cr.minzdrav.gov.ru/schema/1", urls)
        self.assertNotIn("https://random-blog.example/guideline", urls)

    def test_local_russian_protocol_does_not_enter_intl_context(self) -> None:
        hits = [
            {"title": "РФ", "meta": {"source": "Минздрав РФ"}},
            {"title": "INT", "meta": {"source": "WHO", "jurisdiction": "international"}},
        ]
        self.assertEqual([h["title"] for h in AskService._scope_rag_hits(hits, "ru")], ["РФ"])
        self.assertEqual([h["title"] for h in AskService._scope_rag_hits(hits, "en")], ["INT"])

    def test_full_text_excerpt_prefers_clinically_relevant_passage(self) -> None:
        text = (
            "Navigation and cookie policy. "
            "Stroke rehabilitation should include task-specific repetitive practice. "
            "The guideline recommends multidisciplinary assessment after ischemic stroke."
        )
        excerpt = select_relevant_excerpt(text, "ischemic stroke rehabilitation")
        self.assertIn("rehabilitation", excerpt.lower())
        self.assertIn("guideline", excerpt.lower())


if __name__ == "__main__":
    unittest.main()
