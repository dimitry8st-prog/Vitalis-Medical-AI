from app.rag.web_search import (
    _build_queries,
    _detect_intent,
    _english_variant,
    _extract_focus,
)


def test_extracts_question_focus() -> None:
    assert _extract_focus(
        "Как лечить инсульт ишемический по актуальным рекомендациям?"
    ) == "инсульт ишемический"


def test_translates_specific_nosology_without_duplicate_generic_term() -> None:
    assert _english_variant("ишемический инсульт") == "ischemic stroke"


def test_detects_treatment_intent() -> None:
    assert _detect_intent("Как лечить ишемический инсульт?") == (
        "лечение",
        "treatment",
    )


def test_rf_plan_contains_only_rf_queries_and_profiled_sources() -> None:
    queries = _build_queries(
        "Как лечить ишемический инсульт?",
        bilingual=False,
        freshness_extra=True,
        scope="rf",
        en_focus="ischemic stroke",
    )
    assert queries
    assert all(lang == "ru" for _, lang, _ in queries)
    assert any("cr.minzdrav.gov.ru" in query for query, _, _ in queries)
    assert any("НМИЦ" in query for query, _, _ in queries)
    assert any("журнал" in query for query, _, _ in queries)


def test_international_plan_contains_guidelines_and_literature() -> None:
    queries = _build_queries(
        "Как лечить ишемический инсульт?",
        bilingual=True,
        freshness_extra=True,
        scope="intl",
        en_focus="ischemic stroke",
    )
    assert queries
    assert all(lang == "en" for _, lang, _ in queries)
    assert any("WHO NICE" in query for query, _, _ in queries)
    assert any("NIH CDC" in query for query, _, _ in queries)
    assert any("systematic review" in query for query, _, _ in queries)
