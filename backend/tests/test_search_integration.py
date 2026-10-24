import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from app.config import Settings
from app.rag.vault import VaultRAG
from app.services.ask import AskService


class SearchIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.rag = VaultRAG(root / "vault", root / "index")
        self.rag.build()
        self.service = AskService(Settings(llm_mode="mock"), self.rag)
        self.calls = []

        async def complete(system, user):
            self.calls.append((system, user))
            return {
                "provider": "synthetic",
                "mode": "mock",
                "content": "Synthetic medical reference",
            }

        self.service.deepseek.complete = complete

    async def asyncTearDown(self):
        self.directory.cleanup()

    async def test_both_contours_receive_only_their_sources(self):
        rows = [
            {
                "url": "https://cr.minzdrav.gov.ru/ru-only",
                "title": "RU_ONLY_GUIDELINE",
                "lang": "ru",
                "year": 2026,
                "snippet": "Russian evidence",
            },
            {
                "url": "https://who.int/en-only",
                "title": "INT_ONLY_GUIDELINE",
                "lang": "en",
                "year": 2026,
                "snippet": "International evidence",
            },
        ]
        with patch("app.services.ask.web_search", return_value=rows):
            result = await self.service.ask(
                "Лечение ишемического инсульта", mode="both"
            )
        self.assertEqual(len(self.calls), 2)
        self.assertIn("RU_ONLY_GUIDELINE", self.calls[0][1])
        self.assertNotIn("INT_ONLY_GUIDELINE", self.calls[0][1])
        self.assertIn("INT_ONLY_GUIDELINE", self.calls[1][1])
        self.assertNotIn("RU_ONLY_GUIDELINE", self.calls[1][1])
        self.assertEqual(result["search"]["missing_web_languages"], [])

    async def test_strict_gate_blocks_generation_when_evidence_is_missing(self):
        self.service.settings.evidence_gate_mode = "strict"
        with patch("app.services.ask.web_search", return_value=[]):
            result = await self.service.ask(
                "Лечение ишемического инсульта", mode="intl"
            )
        self.assertFalse(self.calls)
        self.assertEqual(result["search"]["evidence_status"], "blocked")

    async def test_warn_gate_marks_provider_failure_without_claiming_verification(self):
        with patch(
            "app.services.ask.web_search", side_effect=TimeoutError("synthetic timeout")
        ):
            result = await self.service.ask("Лечение ишемического инсульта", mode="rf")
        self.assertEqual(result["search"]["evidence_status"], "limited")
        self.assertIn("ОГРАНИЧЕННЫЙ РЕЖИМ", self.calls[0][1])
        self.assertIsNone(result["intl"])

    async def test_greeting_does_not_search_or_generate_medical_answer(self):
        with patch("app.services.ask.web_search", new_callable=AsyncMock) as search:
            result = await self.service.ask("Привет", mode="both")
        search.assert_not_called()
        self.assertFalse(self.calls)
        self.assertEqual(result["search"]["routed"], "faq")


class MainApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_api_routes_and_validation_without_external_services(self):
        # Import/build indexes only in a disposable directory, using synthetic/no-key configuration.
        import os

        from app.config import get_settings

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "FAISS_INDEX_PATH": directory,
                    "LLM_MODE": "mock",
                    "DEEPSEEK_API_KEY": "",
                    "DOCUMENT_EXTERNAL_LLM_ENABLED": "false",
                },
            ),
        ):
            get_settings.cache_clear()
            from app import main

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app), base_url="http://test"
            ) as client:
                health = await client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertFalse(health.json()["document_external_llm"])
                self.assertEqual(
                    (await client.post("/mode", json={"llm_mode": "live"})).status_code,
                    409,
                )
                self.assertEqual(
                    (
                        await client.post(
                            "/ask", json={"question": "Привет", "mode": "both"}
                        )
                    ).status_code,
                    200,
                )
                self.assertEqual(
                    (
                        await client.post(
                            "/ask", json={"question": "valid", "mode": "unknown"}
                        )
                    ).status_code,
                    422,
                )
                self.assertEqual((await client.post("/rag/rebuild")).status_code, 200)
                with patch.object(
                    main.ask_service,
                    "ask",
                    side_effect=RuntimeError("synthetic private upstream detail"),
                ):
                    failed = await client.post(
                        "/ask", json={"question": "Лечение инсульта"}
                    )
                self.assertEqual(failed.status_code, 502)
                self.assertNotIn("private upstream", failed.text)
            get_settings.cache_clear()
