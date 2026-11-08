import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, call, patch

import httpx
from app.config import Settings
from app.rag.vault import VaultRAG
from app.schemas import AskRequest
from app.services.ask import AskService
from pydantic import ValidationError


class SearchIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.root = root
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
                "document_type": "clinical_guideline",
                "full_text_checked": True,
                "evidence_excerpt": "Russian evidence from the full guideline",
            },
            {
                "url": "https://who.int/en-only",
                "title": "INT_ONLY_GUIDELINE",
                "lang": "en",
                "year": 2026,
                "snippet": "International evidence",
                "document_type": "clinical_guideline",
                "full_text_checked": True,
                "evidence_excerpt": "International evidence from the full guideline",
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

    async def test_clinical_pii_is_redacted_before_web_and_model(self):
        rows = [
            {
                "url": "https://cr.minzdrav.gov.ru/example",
                "title": "Synthetic guideline",
                "lang": "ru",
                "year": 2026,
                "snippet": "Synthetic evidence",
                "document_type": "clinical_guideline",
                "full_text_checked": True,
                "evidence_excerpt": "Synthetic full guideline evidence",
            }
        ]
        with patch("app.services.ask.web_search", return_value=rows) as search:
            result = await self.service.ask(
                "Лечение гипертонии, email patient@example.org",
                mode="rf",
            )

        external_query = search.call_args.args[0]
        self.assertNotIn("patient@example.org", external_query)
        self.assertIn("REDACTED_EMAIL", external_query)
        self.assertNotIn("patient@example.org", self.calls[0][1])
        self.assertIn("REDACTED_EMAIL", self.calls[0][1])
        self.assertTrue(result["privacy"]["external_query_sanitized"])

    async def test_residual_pii_blocks_all_external_calls(self):
        with patch("app.services.ask.web_search", new_callable=AsyncMock) as search:
            result = await self.service.ask(
                "Лечение гипертонии, паспорт: неизвестный формат",
                mode="rf",
            )

        search.assert_not_called()
        self.assertFalse(self.calls)
        self.assertEqual(result["search"]["routed"], "privacy_blocked")
        self.assertTrue(result["privacy"]["external_calls_blocked"])

    async def test_identifier_only_query_is_blocked_after_redaction(self):
        with patch("app.services.ask.web_search", new_callable=AsyncMock) as search:
            result = await self.service.ask("Email: user@example.org", mode="rf")

        search.assert_not_called()
        self.assertFalse(self.calls)
        self.assertEqual(result["search"]["routed"], "privacy_blocked")

    async def test_use_web_false_skips_search_and_marks_limited_mode(self):
        with patch("app.services.ask.web_search", new_callable=AsyncMock) as search:
            result = await self.service.ask(
                "Лечение ишемического инсульта",
                mode="rf",
                use_web=False,
            )

        search.assert_not_called()
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(result["search"]["web_forced"])
        self.assertTrue(result["search"]["web_disabled"])
        self.assertEqual(result["search"]["evidence_status"], "limited")

    async def test_explicit_mkb_is_included_in_local_rag_lookup(self):
        with (
            patch.object(self.rag, "search", return_value=[]) as search,
            patch("app.services.ask.web_search", new_callable=AsyncMock) as web,
        ):
            await self.service.ask(
                "Лечение мигрени", mode="rf", use_web=False, mkb="G43"
            )

        web.assert_not_called()
        self.assertEqual(
            search.call_args_list,
            [
                call("Лечение мигрени G43", k=6),
                call(
                    "Лечение мигрени G43 лекарственная терапия препарат "
                    "дозировка доза путь введения",
                    k=8,
                ),
            ],
        )

    async def test_official_local_full_text_satisfies_rf_evidence_gate(self):
        vault = self.root / "vault"
        vault.mkdir(parents=True, exist_ok=True)
        url = "https://cr.minzdrav.gov.ru/view-cr/295_4"
        today = date.today()
        (vault / "migraine.md").write_text(
            "---\n"
            'title: "Мигрень"\n'
            'mkb: "G43"\n'
            'mkb_codes: "G43, G43.0"\n'
            f'year: "{today.year}"\n'
            'source: "Минздрав России"\n'
            'jurisdiction: "RU"\n'
            'language: "ru"\n'
            'document_type: "clinical_guideline_full_text"\n'
            'source_version: "current-test-version"\n'
            'guideline_status: "active"\n'
            f'current_version_checked_at: "{today.isoformat()}"\n'
            f'source_url: "{url}"\n'
            f'url: "{url}"\n'
            'specialty_tags: "neurology"\n'
            'content_scope: "official_full_text_extracted"\n'
            'full_text_checked: true\n'
            'page_count: 92\n'
            'full_text_sha256: "abc123"\n'
            "---\n\n## Страница 12\n\n"
            "Диагностика и лечение мигрени G43. Синтетический препарат "
            "50 мг внутрь 1 раз в сутки.",
            encoding="utf-8",
        )
        self.rag.build()
        self.service.settings.evidence_gate_mode = "strict"

        async def complete(system, user):
            self.calls.append((system, user))
            return {
                "provider": "synthetic",
                "mode": "mock",
                "content": f"Краткий ответ. Источник: {url}",
            }

        self.service.deepseek.complete = complete
        with patch("app.services.ask.web_search", new_callable=AsyncMock) as search:
            result = await self.service.ask(
                "Лечение мигрени G43", mode="rf", use_web=False
            )
        search.assert_not_called()
        self.assertEqual(result["search"]["evidence_status"], "sources_found")
        self.assertEqual(result["search"]["missing_web_languages"], [])
        self.assertIn("[Страница 12]", self.calls[0][1])
        source = result["card"]["tabs"]["rf"]["sources"][0]
        self.assertTrue(source["full_text_checked"])
        self.assertEqual(source["page_count"], 92)
        self.assertEqual(source["full_text_sha256"], "abc123")
        medications = result["card"]["tabs"]["rf"]["medications"]
        self.assertEqual(medications["status"], "verified")
        self.assertEqual(medications["items"][0]["doses"], ["50 мг"])
        self.assertEqual(medications["items"][0]["page"], 12)

    def test_mkb_field_accepts_code_but_rejects_free_text(self):
        self.assertEqual(AskRequest(question="Лечение", mkb="I21.9").mkb, "I21.9")
        with self.assertRaises(ValidationError):
            AskRequest(question="Лечение", mkb="Иванов 123456")

    def test_card_uses_portal_guideline_title_codes_and_official_url(self):
        hit = {
            "title": "Геморрагический инсульт",
            "path": "generated/guidelines/rf/minzdrav-full-523_3.md",
            "text": "[Страница 14] Официальный полный текст клинической рекомендации",
            "score": 0.8,
            "year": 2025,
            "stale": False,
            "meta": {
                "mkb": "I60",
                "mkb_codes": "I60, I61, I62, Q28.2",
                "url": "https://cr.minzdrav.gov.ru/view-cr/523_3",
                "document_type": "clinical_guideline_full_text",
                "content_scope": "official_full_text_extracted",
                "full_text_checked": "true",
                "specialty_tags": "neurology neurosurgery",
            },
        }
        card = self.service._build_card(
            "Диагностика геморрагического инсульта",
            None,
            None,
            None,
            [hit],
            [],
        )
        self.assertEqual(card["title"], "Геморрагический инсульт")
        self.assertEqual(card["mkb"], "I60")
        self.assertEqual(card["mkb_codes"], "I60, I61, I62, Q28.2")
        self.assertEqual(card["specialty_tags"], ["neurology", "neurosurgery"])
        source = card["tabs"]["rf"]["sources"][0]
        self.assertEqual(
            source["ref"], "https://cr.minzdrav.gov.ru/view-cr/523_3"
        )
        self.assertEqual(source["local_ref"], hit["path"])

    def test_hemorrhagic_stroke_is_not_misclassified_as_hemorrhoids(self):
        card = self.service._build_card(
            "Диагностика геморрагического инсульта",
            None,
            None,
            None,
            [],
            [],
        )
        self.assertNotEqual(card["mkb"], "K64")
        self.assertNotEqual(card["title"], "Геморрой")

    def test_metadata_only_card_is_excluded_from_clinical_rag(self):
        hit = {
            "title": "Мигрень",
            "path": "guidelines/rf/minzdrav-295_4.md",
            "text": "Официальная библиографическая карточка",
            "score": 0.7,
            "year": 2024,
            "stale": False,
            "meta": {
                "document_type": "clinical_guideline_metadata",
                "content_scope": "bibliographic_metadata_only",
            },
        }
        self.assertEqual(
            self.service._filter_relevant_rag("Лечение мигрени", [hit]), []
        )

    def test_unfetched_registry_seed_does_not_satisfy_web_evidence_gate(self):
        missing = self.service._missing_required_web(
            "rf",
            [
                {
                    "lang": "ru",
                    "url": "https://cr.minzdrav.gov.ru/view-cr/523_3",
                    "provider": "portal_registry",
                    "content_scope": "bibliographic_metadata_only",
                }
            ],
        )
        self.assertEqual(missing, ["ru"])

    def test_active_rechecked_old_guideline_satisfies_rf_evidence_gate(self):
        missing = self.service._missing_required_web(
            "rf",
            [],
            [
                {
                    "stale": True,
                    "meta": {
                        "source_url": "https://cr.minzdrav.gov.ru/view-cr/295_4",
                        "content_scope": "official_full_text_extracted",
                        "full_text_checked": "true",
                        "source_version": "active-old-version",
                        "guideline_status": "active",
                        "current_version_checked_at": date.today().isoformat(),
                    },
                }
            ],
        )

        self.assertEqual(missing, [])


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
