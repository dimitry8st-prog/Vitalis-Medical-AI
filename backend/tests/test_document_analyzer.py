import json
import unittest

from app.config import Settings
from app.documents.analyzer import DocumentAnalyzer
from app.documents.models import DocumentPage


class FakeClient:
    ready = True

    def __init__(self, payload: dict):
        self.payload = payload

    async def complete(self, system: str, user: str) -> dict:
        self.last_system = system
        self.last_user = user
        return {
            "provider": "fake",
            "mode": "live",
            "content": json.dumps(self.payload, ensure_ascii=False),
        }


class FailingClient:
    ready = True

    async def complete(self, system: str, user: str) -> dict:
        raise RuntimeError("sensitive upstream detail")


def live_settings() -> Settings:
    return Settings(
        llm_mode="live",
        deepseek_api_key="synthetic-key",
        document_external_llm_enabled=True,
    )


class DocumentAnalyzerTests(unittest.IsolatedAsyncioTestCase):
    async def test_keeps_only_verifiable_citations(self) -> None:
        client = FakeClient(
            {
                "document_type": "Лабораторный результат",
                "summary": "Синтетический документ.",
                "diagnoses": [],
                "measurements": [
                    {
                        "category": "measurement",
                        "text": "Глюкоза 5.4 mmol/L",
                        "page": 1,
                        "source_quote": "Glucose 5.4 mmol/L",
                    },
                    {
                        "category": "measurement",
                        "text": "Выдуманный показатель",
                        "page": 1,
                        "source_quote": "This quote is absent",
                    },
                ],
                "medications": [],
                "findings": [],
                "contradictions": [],
                "red_flags": [],
                "missing_information": [],
                "questions_for_doctor": [],
                "limitations": [],
            }
        )
        analyzer = DocumentAnalyzer(live_settings(), client=client)
        result, mode = await analyzer.analyze(
            (DocumentPage(number=1, text="Glucose 5.4 mmol/L"),)
        )
        self.assertEqual(mode, "live")
        self.assertEqual(len(result.measurements), 1)
        self.assertIn("Исключено пунктов", result.limitations[-1])
        self.assertIn("Glucose 5.4 mmol/L", result.summary)
        self.assertNotIn("Синтетический документ", result.summary)
        self.assertIn("uploaded_medical_pdf_untrusted", client.last_user)

    async def test_external_analysis_can_be_deterministically_blocked(self) -> None:
        analyzer = DocumentAnalyzer(live_settings(), client=FakeClient({}))
        result, mode = await analyzer.analyze(
            (DocumentPage(number=1, text="Diagnosis: synthetic"),),
            allow_external=False,
        )
        self.assertEqual(mode, "disabled")
        self.assertTrue(result.limitations)

    async def test_refuses_treatment_action_question(self) -> None:
        analyzer = DocumentAnalyzer(live_settings(), client=FakeClient({}))
        answer, mode = await analyzer.answer_question(
            (DocumentPage(number=1, text="Synthetic text"),),
            "Назначь лечение для меня",
            allow_external=True,
        )
        self.assertEqual(mode, "disabled")
        self.assertIn("не могу", answer.answer)

    async def test_hides_upstream_error_details(self) -> None:
        analyzer = DocumentAnalyzer(live_settings(), client=FailingClient())
        with self.assertRaisesRegex(Exception, "временно недоступен") as caught:
            await analyzer.analyze((DocumentPage(number=1, text="Synthetic text"),))
        self.assertNotIn("sensitive", str(caught.exception))

    async def test_offline_question_returns_exact_document_evidence(self) -> None:
        settings = Settings(llm_mode="mock", document_external_llm_enabled=False)
        analyzer = DocumentAnalyzer(settings)
        answer, mode = await analyzer.answer_question(
            (DocumentPage(number=4, text="Glucose result is 5.4 mmol/L"),),
            "What is the glucose result?",
            allow_external=True,
        )
        self.assertEqual(mode, "mock")
        self.assertEqual(answer.evidence[0].page, 4)
        self.assertIn("Glucose", answer.evidence[0].source_quote)


if __name__ == "__main__":
    unittest.main()
