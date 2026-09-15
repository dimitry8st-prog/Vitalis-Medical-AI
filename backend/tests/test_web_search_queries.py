import unittest

from app.rag.web_search import (
    _build_queries,
    _detect_intent,
    _english_variant,
    _extract_focus,
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
