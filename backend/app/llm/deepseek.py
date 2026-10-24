from __future__ import annotations

from typing import Any

import httpx

from app.config import Settings


class DeepSeekClient:
    """DeepSeek (OpenAI-compatible). Без ключа — mock."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def ready(self) -> bool:
        return self.settings.deepseek_ready and self.settings.llm_mode == "live"

    async def complete(self, system: str, user: str) -> dict[str, Any]:
        if not self.ready:
            return {
                "provider": "deepseek",
                "mode": "mock",
                "content": self._mock(system, user),
            }

        url = self.settings.deepseek_base_url.rstrip("/") + "/chat/completions"
        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.settings.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.deepseek_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.2,
                },
            )
            res.raise_for_status()
            data = res.json()
        content = data["choices"][0]["message"]["content"]
        return {"provider": "deepseek", "mode": "live", "content": content, "raw": data}

    @staticmethod
    def _mock(system: str, user: str) -> str:
        if "российский контур" in system.lower():
            return (
                "## Краткий ответ\n\n"
                "Российский контур работает в демо-режиме и не формирует медицинские рекомендации.\n\n"
                "## Ограничения\n\n"
                f"Запрос: «{user[:200]}». Для проверяемого ответа добавьте ключ API и установите LLM_MODE=live.\n\n"
                "## Источники\n\n"
                "Источники не загружены: демо-режим.\n\n"
                "Информация носит справочный характер. Решение принимает лечащий врач."
            )
        return (
            "## Международный контур (демо-режим)\n\n"
            f"Запрос: «{user[:200]}»\n\n"
            "Медицинские рекомендации в демо-режиме не формируются. "
            "Для проверяемого ответа настройте API и проверьте найденные источники.\n\n"
            "_Демо-режим. Добавьте ключи API и LLM_MODE=live._"
        )
