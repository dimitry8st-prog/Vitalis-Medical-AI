import unittest
from datetime import UTC, datetime

from app.config import Settings
from app.documents.insurance_search import (
    InsuranceOfficialSearch,
    detect_insurance_type,
    detect_insurer,
    is_official_insurer_url,
    select_insurance_excerpt,
)
from app.documents.models import DocumentPage, InsuranceAnalysis


class InsuranceSearchTests(unittest.IsolatedAsyncioTestCase):
    def test_excerpt_prefers_coverage_terms(self) -> None:
        text = (
            "Главная страница компании. Новости и вакансии.\n\n"
            "Программа ДМС включает амбулаторную помощь и лечение в стационаре.\n\n"
            "Контакты пресс-службы."
        )
        excerpt = select_insurance_excerpt(text, "ДМС", max_chars=500)
        self.assertIn("амбулаторную помощь", excerpt)
        self.assertNotIn("вакансии", excerpt)

    def test_detects_allowlisted_insurer_and_insurance_type(self) -> None:
        pages = (DocumentPage(number=1, text="СПАО Ингосстрах. Полис КАСКО."),)
        insurer = detect_insurer(pages)
        self.assertIsNotNone(insurer)
        self.assertEqual(insurer.domain, "ingos.ru")
        self.assertEqual(detect_insurance_type(pages), "КАСКО")

    def test_official_url_check_rejects_lookalike_and_credentials(self) -> None:
        self.assertTrue(
            is_official_insurer_url("https://www.sogaz.ru/health/", "sogaz.ru")
        )
        self.assertFalse(
            is_official_insurer_url("https://sogaz.ru.evil.example/", "sogaz.ru")
        )
        self.assertFalse(is_official_insurer_url("http://sogaz.ru/", "sogaz.ru"))
        self.assertFalse(is_official_insurer_url("https://user@sogaz.ru/", "sogaz.ru"))

    async def test_lookup_keeps_only_official_domain(self) -> None:
        captured: list[str] = []

        async def fake_search(query: str, settings: Settings, limit: int):
            captured.append(query)
            return [
                {
                    "title": "ДМС СОГАЗ",
                    "url": "https://www.sogaz.ru/health/dms/",
                    "snippet": "Описание программы ДМС",
                },
                {
                    "title": "Поддельный сайт",
                    "url": "https://sogaz.ru.evil.example/dms",
                    "snippet": "Не использовать",
                },
            ]

        async def fake_fetch(
            url: str,
            domain: str,
            query: str,
            settings: Settings,
        ):
            return "Покрытие: амбулаторная помощь.", datetime.now(UTC)

        settings = Settings(insurance_official_search_enabled=True)
        service = InsuranceOfficialSearch(
            settings,
            search_provider=fake_search,
            excerpt_fetcher=fake_fetch,
        )
        pages = (DocumentPage(number=1, text="СОГАЗ. Полис ДМС."),)
        result = await service.lookup_missing_coverage(
            pages,
            InsuranceAnalysis(summary="Синтетический договор", insurance_type="ДМС"),
            requested=True,
        )
        self.assertEqual(result.status, "found")
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].domain, "sogaz.ru")
        self.assertIn("site:sogaz.ru", captured[0])

    async def test_unknown_insurer_never_starts_web_search(self) -> None:
        calls = 0

        async def fake_search(query: str, settings: Settings, limit: int):
            nonlocal calls
            calls += 1
            return []

        service = InsuranceOfficialSearch(
            Settings(),
            search_provider=fake_search,
        )
        result = await service.lookup_missing_coverage(
            (DocumentPage(number=1, text="Неизвестная компания. Полис ДМС."),),
            InsuranceAnalysis(summary="Синтетический договор"),
            requested=True,
        )
        self.assertEqual(result.status, "unsupported_insurer")
        self.assertEqual(calls, 0)

    async def test_empty_search_uses_only_allowlisted_insurer_homepage(self) -> None:
        fetched: list[tuple[str, str]] = []

        async def empty_search(query: str, settings: Settings, limit: int):
            return []

        async def fake_fetch(
            url: str,
            domain: str,
            query: str,
            settings: Settings,
        ):
            fetched.append((url, domain))
            return "ДМС включает медицинскую помощь.", datetime.now(UTC)

        service = InsuranceOfficialSearch(
            Settings(),
            search_provider=empty_search,
            excerpt_fetcher=fake_fetch,
        )
        result = await service.lookup_missing_coverage(
            (DocumentPage(number=1, text="sogaz.ru Policy DMS"),),
            InsuranceAnalysis(summary="Синтетический договор"),
            requested=True,
        )
        self.assertEqual(result.status, "found")
        self.assertEqual(fetched, [("https://sogaz.ru/", "sogaz.ru")])

    async def test_provider_and_fetch_failures_are_non_fatal(self) -> None:
        async def failed_search(query: str, settings: Settings, limit: int):
            raise RuntimeError("synthetic provider failure")

        async def failed_fetch(
            url: str,
            domain: str,
            query: str,
            settings: Settings,
        ):
            raise RuntimeError("synthetic fetch failure")

        service = InsuranceOfficialSearch(
            Settings(),
            search_provider=failed_search,
            excerpt_fetcher=failed_fetch,
        )
        result = await service.lookup_missing_coverage(
            (DocumentPage(number=1, text="СОГАЗ. Полис ДМС."),),
            InsuranceAnalysis(summary="Синтетический договор"),
            requested=True,
        )
        self.assertEqual(result.status, "not_found")


if __name__ == "__main__":
    unittest.main()
