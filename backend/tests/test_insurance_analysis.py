import json
import unittest

from app.config import Settings
from app.documents.analyzer import DocumentAnalyzer
from app.documents.models import DocumentPage, OfficialSourceEvidence


class FakeInsuranceClient:
    ready = True

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_system = ""
        self.last_user = ""

    async def complete(self, system: str, user: str) -> dict:
        self.last_system = system
        self.last_user = user
        return {"content": json.dumps(self.payload, ensure_ascii=False)}


class InsuranceAnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_analysis_keeps_only_contract_citations(self) -> None:
        client = FakeInsuranceClient(
            {
                "document_type": "Любой тип",
                "summary": "Нельзя доверять этой выжимке.",
                "insurer_name": "Выдуманная компания",
                "insurer_domain": "evil.example",
                "insurance_type": "Выдуманный тип",
                "policy_name": "Несуществующая программа",
                "coverage": [
                    {
                        "category": "coverage",
                        "text": "Амбулаторная помощь",
                        "page": 1,
                        "source_quote": "Покрытие: амбулаторная помощь",
                    },
                    {
                        "category": "coverage",
                        "text": "Выдуманное покрытие",
                        "page": 1,
                        "source_quote": "Этой строки нет",
                    },
                ],
                "exclusions": [],
                "limits": [],
                "deductibles": [],
                "validity": [],
                "claim_procedure": [],
                "contacts": [],
                "missing_information": [],
                "questions_for_insurer": [],
                "limitations": [],
            }
        )
        settings = Settings(
            llm_mode="live",
            deepseek_api_key="synthetic-key",
            document_external_llm_enabled=True,
        )
        analyzer = DocumentAnalyzer(settings, client=client)
        analysis, mode = await analyzer.analyze(
            (
                DocumentPage(
                    number=1,
                    text="СОГАЗ. Полис ДМС. Покрытие: амбулаторная помощь",
                ),
            ),
            document_kind="insurance",
        )
        self.assertEqual(mode, "live")
        self.assertEqual(len(analysis.coverage), 1)
        self.assertEqual(analysis.insurer_domain, "sogaz.ru")
        self.assertEqual(analysis.insurance_type, "ДМС")
        self.assertIsNone(analysis.policy_name)
        self.assertIn("uploaded_insurance_pdf_untrusted", client.last_user)
        self.assertIn("страхового договора", client.last_system)

    async def test_offline_analysis_detects_insurer_type_and_contract_terms(
        self,
    ) -> None:
        analyzer = DocumentAnalyzer(
            Settings(llm_mode="mock", document_external_llm_enabled=False)
        )
        pages = (
            DocumentPage(
                number=1,
                text=(
                    "Страховая компания СОГАЗ\n"
                    "Полис ДМС\n"
                    "Страховые риски: амбулаторная помощь и стационар\n"
                    "Исключения: услуги вне согласованной программы\n"
                    "Страховая сумма: 500 000 рублей"
                ),
            ),
        )
        analysis, mode = await analyzer.analyze(
            pages,
            document_kind="insurance",
        )
        self.assertEqual(mode, "mock")
        self.assertEqual(analysis.insurer_name, "СОГАЗ")
        self.assertEqual(analysis.insurer_domain, "sogaz.ru")
        self.assertEqual(analysis.insurance_type, "ДМС")
        self.assertEqual(analysis.coverage[0].page, 1)
        self.assertTrue(analysis.exclusions)
        self.assertTrue(analysis.limits)

    async def test_question_falls_back_to_official_sources_when_pdf_is_silent(
        self,
    ) -> None:
        analyzer = DocumentAnalyzer(
            Settings(llm_mode="mock", document_external_llm_enabled=False)
        )
        source = OfficialSourceEvidence(
            title="Программа ДМС",
            url="https://www.sogaz.ru/health/dms/",
            domain="sogaz.ru",
            evidence_excerpt="Программа включает амбулаторную помощь.",
        )
        answer, mode = await analyzer.answer_question(
            (
                DocumentPage(
                    number=1,
                    text="Договор страхования СОГАЗ. Полис ДМС.",
                ),
            ),
            "Что входит в страховку?",
            allow_external=False,
            document_kind="insurance",
            official_sources=[source],
        )
        self.assertEqual(mode, "disabled")
        self.assertFalse(answer.evidence)
        self.assertEqual(answer.official_sources[0].domain, "sogaz.ru")
        self.assertIn("официального сайта", answer.answer)

    async def test_refuses_to_guarantee_claim_payment(self) -> None:
        analyzer = DocumentAnalyzer(
            Settings(llm_mode="mock", document_external_llm_enabled=False)
        )
        answer, mode = await analyzer.answer_question(
            (DocumentPage(number=1, text="Синтетический договор"),),
            "Гарантируй выплату по этому полису",
            allow_external=True,
            document_kind="insurance",
        )
        self.assertEqual(mode, "disabled")
        self.assertIn("не могу гарантировать", answer.answer)


if __name__ == "__main__":
    unittest.main()
