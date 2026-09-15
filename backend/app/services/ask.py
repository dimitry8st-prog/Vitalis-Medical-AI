from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from app.config import Settings
from app.llm.deepseek import DeepSeekClient
from app.rag.vault import VaultRAG
from app.rag.web_search import web_search
from app.services.routing import canned_answer, classify_intent, is_nonclinical

DISCLAIMER = "Информация носит справочный характер. Решение принимает лечащий врач."
STALE_YEARS = 5  # STRICT MODE: >5 лет = устарел

IDENTITY_RULES = """
# ИДЕНТИЧНОСТЬ
Ты — «Система Vitalis». Никогда не упоминай названия моделей, провайдеров ИИ, LLM, GigaChat, DeepSeek, Claude и т.п.
Ссылайся только на медицинские источники (документы, организации, годы).
"""

ROUTING_LOGIC = """
# ЛОГИКА ОБРАБОТКИ ЗАПРОСОВ (ROUTING LOGIC)

1. КЛАССИФИКАЦИЯ ЗАПРОСА:
   - ПРИВЕТСТВИЕ / ФУНКЦИОНАЛ / ИНТЕРФЕЙС (non-clinical):
     → ИГНОРИРУЙ правило «только свежие источники».
     → Используй файл Vitalis_FAQ_Skills или знания о системе.
     → Отвечай приветливо и профессионально.
     → На «Привет» НИКОГДА не выдавай медицинский протокол.
   - МЕДИЦИНСКИЙ (диагноз, лечение, дозировки):
     → СТРОГО правило актуальности (Web Search > RAG).
   - НЕМЕДИЦИНСКИЙ / ОПАСНЫЙ / НЕЯСНЫЙ:
     → [Вежливый отказ] + [специализация] + [предложение медпомощи].
     → Не отвечай по существу.

2. ТОН:
   - Приветствия: «Здравствуйте, коллега…»
   - Медицина: сухой, фактологический, структурированный.

3. ЗАПРЕТЫ:
   - Не генерируй немедицинский контент даже по просьбе.
   - Не давай юридических, финансовых, бытовых советов.
   - Не обсуждай политику и религию.
"""

STRICT_FRESHNESS = """
# АКТУАЛЬНОСТЬ (STRICT) — для медицинских запросов
1. Документ >5 лет = устаревший; предпочитай веб 2023–2026.
2. Веб > RAG при противоречии.
3. Указывай год источника. Не омолаживай даты.
4. В первую очередь используй источники, переданные в контексте.
5. Если веб-поиск не вернул источники, явно пометь ответ как справочный обзор без веб-проверки.
   В этом режиме не выдумывай ссылки и документы, не выдавай индивидуальные назначения
   и не указывай точные дозировки, если они не подтверждены контекстом.
6. Не выдумывай названия документов, рекомендации, дозировки, ссылки и даты.
"""

RF_SYSTEM = f"""Ты — Система Vitalis (контур РФ-рекомендаций).
Отвечай на русском. Фокус: Минздрав РФ, национальные клинрек, профильные ассоциации РФ.
Можно кратко упомянуть международный контекст, если это помогает, но приоритет — РФ.

{IDENTITY_RULES}
{ROUTING_LOGIC}
{STRICT_FRESHNESS}

# ОБЯЗАТЕЛЬНЫЙ ФОРМАТ МЕДИЦИНСКОГО ОТВЕТА
Заполни КАЖДЫЙ применимый раздел РЕАЛЬНЫМИ фактами (маркированные списки, 3–8 пунктов).
Запрещено писать только «Минздрав РФ» или оставлять раздел пустым.

## Определение / суть (если вопрос «что такое…»)
## 📋 Диагностика
## 💊 Лечение
## 🔄 Реабилитация и прогноз
## ⚠️ Важные предупреждения / Противопоказания

В каждом разделе — факты + источник с годом (если год неизвестен — так и напиши).
В конце:
{DISCLAIMER}
"""

INTL_SYSTEM = f"""Ты — Система Vitalis (международный контур рекомендаций).
Отвечай на русском. Фокус: WHO, CDC, NICE, ESC/ERS, PubMed, Medscape, Mayo.
Если уместно — сравни с подходом РФ (Минздрав).

{IDENTITY_RULES}
{ROUTING_LOGIC}
{STRICT_FRESHNESS}

Полные разделы с фактами, не пустые заголовки.
В конце МЕДИЦИНСКОГО ответа:
{DISCLAIMER}
"""

SECTION_DEFS = [
    ("def", "📖 Определение", r"определен|суть|что\s+такое|этиолог|патоген"),
    ("d", "📋 Диагностика", r"диагност"),
    ("l", "💊 Лечение", r"лечен"),
    ("r", "🔄 Реабилитация", r"реабилитац|прогноз"),
    ("w", "⚠️ Предупреждения", r"предупрежд|противопоказан|неотлож|важно"),
]


class AskService:
    def __init__(self, settings: Settings, rag: VaultRAG):
        self.settings = settings
        self.rag = rag
        # В личном режиме оба клинических контура используют DeepSeek.
        # GigaChat сохранён в проекте только как отключённая заглушка для будущей интеграции.
        self.deepseek = DeepSeekClient(settings)

    async def ask(
        self,
        question: str,
        mode: str = "both",
        use_web: bool = True,
        mkb: str | None = None,
    ) -> dict[str, Any]:
        intent = classify_intent(question)
        ready = canned_answer(question, intent)

        # Non-clinical: мгновенный ответ из FAQ / routing, без веб-протоколов
        if is_nonclinical(intent) and ready:
            payload = {
                "provider": "vitalis",
                "mode": "faq",
                "content": ready,
            }
            card = {
                "mkb": "—",
                "title": {
                    "greeting": "Приветствие",
                    "skills": "Возможности системы",
                    "tech": "О системе",
                    "offtopic": "Вне специализации",
                    "unsafe": "Запрос вне компетенции",
                    "vague": "Уточнение запроса",
                }.get(intent, "Система Vitalis"),
                "identity": "Система Vitalis",
                "intent": intent,
                "tabs": {
                    "rf": {
                        "label": "РФ · рекомендации",
                        "content": ready,
                        "sections": [{"key": "all", "title": "Ответ", "body": ready}],
                        "sources": [
                            {
                                "type": "vault",
                                "title": "Vitalis FAQ Skills",
                                "ref": "Vitalis_FAQ_Skills.md",
                                "year": 2026,
                                "stale": False,
                            }
                        ],
                    },
                    "intl": {
                        "label": "Международные · рекомендации",
                        "content": ready,
                        "sections": [{"key": "all", "title": "Ответ", "body": ready}],
                        "sources": [],
                    },
                },
            }
            return {
                "question": question,
                "mode": mode,
                "llm_mode": self.settings.llm_mode,
                "card": card,
                "rf": payload if mode in ("rf", "both") else None,
                "intl": payload if mode in ("intl", "both") else None,
                "rag": [],
                "web": [],
                "search": {
                    "intent": intent,
                    "routed": "faq",
                    "rag_total": 0,
                    "rag_relevant": 0,
                    "rag_stale": 0,
                    "web_total": 0,
                    "web_forced": False,
                    "freshness_extra": False,
                    "stale_years": STALE_YEARS,
                    "use_web_requested": use_web,
                },
                "disclaimer": DISCLAIMER if intent not in {"greeting", "skills", "tech", "vague", "offtopic", "unsafe"} else "",
            }

        rag_raw = self.rag.search(question, k=6)
        rag_hits = self._annotate_rag_age(rag_raw)
        relevant_rag = self._filter_relevant_rag(question, rag_hits)
        # FAQ-файл не должен становиться «клиническим» источником лечения
        relevant_rag = [
            h for h in relevant_rag
            if "faq" not in (h.get("path") or "").lower()
            and "vitalis_faq" not in (h.get("path") or "").lower()
        ]
        has_stale = any(h.get("stale") for h in relevant_rag) or not relevant_rag

        web_hits: list[dict[str, Any]] = []
        web_error = None
        try:
            web_hits = await web_search(
                question,
                self.settings,
                max_results=4,
                bilingual=True,
                freshness_extra=True,
            )
        except Exception as e:
            web_hits = []
            web_error = f"{type(e).__name__}: {e}"

        missing_web = self._missing_required_web(mode, web_hits)
        if missing_web and self.settings.evidence_gate_mode == "strict":
            return self._evidence_blocked_response(
                question=question,
                mode=mode,
                mkb=mkb,
                intent=intent,
                rag_hits=relevant_rag,
                web_hits=web_hits,
                web_error=web_error,
                missing_web=missing_web,
                use_web=use_web,
            )

        context = self._format_context(relevant_rag, rag_hits, web_hits, has_stale)
        if missing_web:
            missing_label = "+".join(code.upper() for code in missing_web)
            context += (
                "\n\n# ОГРАНИЧЕННЫЙ РЕЖИМ ИСТОЧНИКОВ\n"
                f"Веб-поиск не вернул обязательные источники: {missing_label}. "
                "Начни ответ с предупреждения: «Веб-проверка источников сейчас недоступна; "
                "ниже приведён общий справочный обзор, требующий проверки по действующим "
                "клиническим рекомендациям». Не выдумывай ссылки, названия документов и даты. "
                "Не давай индивидуальных назначений и неподтверждённых точных дозировок."
            )
        # Подмешиваем FAQ-контекст только если релевантен onboarding
        faq_hits = [h for h in rag_hits if "faq" in (h.get("path") or "").lower()]
        if faq_hits:
            context += "\n\n# Справка системы (Vitalis_FAQ_Skills)\n" + "\n".join(
                f"- {h['title']}: {h['text'][:400]}" for h in faq_hits[:2]
            )

        user_blob = self._user_prompt(question, context, mkb)
        user_blob = f"Классификация запроса: {intent} (clinical).\n" + user_blob

        rf = None
        intl = None
        if mode in ("rf", "both"):
            rf = await self.deepseek.complete(RF_SYSTEM, user_blob)
            rf = self._sanitize_model_names(rf)
        if mode in ("intl", "both"):
            intl = await self.deepseek.complete(INTL_SYSTEM, user_blob)
            intl = self._sanitize_model_names(intl)

        card = self._build_card(question, mkb, rf, intl, relevant_rag, web_hits)
        card["intent"] = intent

        return {
            "question": question,
            "mode": mode,
            "llm_mode": self.settings.llm_mode,
            "card": card,
            "rf": rf,
            "intl": intl,
            "rag": relevant_rag,
            "web": web_hits,
            "search": {
                "intent": intent,
                "routed": "clinical",
                "rag_total": len(rag_hits),
                "rag_relevant": len(relevant_rag),
                "rag_stale": sum(1 for h in relevant_rag if h.get("stale")),
                "web_total": len(web_hits),
                "web_forced": True,
                "freshness_extra": True,
                "stale_years": STALE_YEARS,
                "use_web_requested": use_web,
                "web_error": web_error,
            },
            "disclaimer": DISCLAIMER,
        }

    @staticmethod
    def _missing_required_web(mode: str, web_hits: list[dict[str, Any]]) -> list[str]:
        available = {
            str(hit.get("lang") or "").lower()
            for hit in web_hits
            if hit.get("provider") != "error" and (hit.get("url") or hit.get("title"))
        }
        required = {"ru", "en"} if mode == "both" else {mode}
        # Названия режимов отличаются от языковых кодов.
        if "rf" in required:
            required.remove("rf")
            required.add("ru")
        if "intl" in required:
            required.remove("intl")
            required.add("en")
        return sorted(required - available)

    def _evidence_blocked_response(
        self,
        *,
        question: str,
        mode: str,
        mkb: str | None,
        intent: str,
        rag_hits: list[dict[str, Any]],
        web_hits: list[dict[str, Any]],
        web_error: str | None,
        missing_web: list[str],
        use_web: bool,
    ) -> dict[str, Any]:
        names = {"ru": "российские", "en": "международные"}
        missing_label = " и ".join(names.get(code, code) for code in missing_web)
        message = (
            f"Актуальные подтверждённые {missing_label} веб-источники не найдены. "
            "Медицинский ответ не сформирован, чтобы избежать недостоверных рекомендаций. "
            "Уточните запрос или повторите поиск позже."
        )
        payload = {"provider": "vitalis", "mode": "evidence_blocked", "content": message}

        def sources_for(lang: str) -> list[dict[str, Any]]:
            return [
                {
                    "type": "web",
                    "group": "rf" if lang == "ru" else "intl",
                    "title": hit.get("title", ""),
                    "ref": hit.get("url", ""),
                    "lang": lang,
                    "year": hit.get("year"),
                    "stale": False,
                }
                for hit in web_hits
                if hit.get("lang") == lang and (hit.get("title") or hit.get("url"))
            ]

        tab = lambda label, lang: {
            "label": label,
            "content": message,
            "sections": [{"key": "w", "title": "Источников недостаточно", "body": message}],
            "sources": sources_for(lang),
        }
        card = {
            "mkb": mkb or "—",
            "title": "Ответ не сформирован: недостаточно источников",
            "identity": "Система Vitalis",
            "intent": intent,
            "tabs": {
                "rf": tab("РФ · рекомендации", "ru"),
                "intl": tab("Международные · рекомендации", "en"),
            },
        }
        return {
            "question": question,
            "mode": mode,
            "llm_mode": self.settings.llm_mode,
            "card": card,
            "rf": payload if mode in ("rf", "both") else None,
            "intl": payload if mode in ("intl", "both") else None,
            "rag": rag_hits,
            "web": web_hits,
            "search": {
                "intent": intent,
                "routed": "evidence_blocked",
                "rag_total": len(rag_hits),
                "rag_relevant": len(rag_hits),
                "rag_stale": sum(1 for hit in rag_hits if hit.get("stale")),
                "web_total": len(web_hits),
                "web_forced": True,
                "web_required": True,
                "missing_web_languages": missing_web,
                "use_web_requested": use_web,
                "web_error": web_error,
                "evidence_status": "limited" if missing_web else "verified_web",
                "missing_web_languages": missing_web,
                "evidence_gate_mode": self.settings.evidence_gate_mode,
            },
            "disclaimer": DISCLAIMER,
        }

    @staticmethod
    def _sanitize_model_names(payload: dict | None) -> dict | None:
        if not payload or not payload.get("content"):
            return payload
        text = payload["content"]
        for name in (
            "GigaChat",
            "Gigachat",
            "DeepSeek",
            "Deepseek",
            "Claude",
            "ChatGPT",
            "GPT-4",
            "GPT-4o",
            "LLM",
        ):
            text = re.sub(rf"\b{re.escape(name)}\b", "Система Vitalis", text, flags=re.I)
        payload = dict(payload)
        payload["content"] = text
        return payload

    def _annotate_rag_age(self, hits: list[dict]) -> list[dict]:
        current_year = datetime.now().year
        out = []
        for h in hits:
            item = dict(h)
            meta = item.get("meta") or {}
            year = _parse_year(str(meta.get("year") or ""))
            # также ищем год в тексте
            if not year:
                year = _parse_year(item.get("text") or "")
            stale = bool(year and (current_year - year) >= STALE_YEARS)
            item["year"] = year
            item["stale"] = stale
            out.append(item)
        return out

    @staticmethod
    def _filter_relevant_rag(question: str, hits: list[dict], min_score: float = 0.12) -> list[dict]:
        q_tokens = set(_tokens(question))
        relevant = []
        for h in hits:
            score = float(h.get("score") or 0)
            title_tokens = set(_tokens(h.get("title", "")))
            text_tokens = set(_tokens((h.get("text") or "")[:400]))
            overlap = len(q_tokens & (title_tokens | text_tokens))
            if score >= min_score and overlap >= 1:
                relevant.append(h)
            elif overlap >= 2 and score >= 0.05:
                relevant.append(h)
        return relevant

    @staticmethod
    def _format_context(
        relevant: list[dict],
        all_rag: list[dict],
        web_hits: list[dict],
        has_stale: bool,
    ) -> str:
        current_year = datetime.now().year
        parts = [
            "# ПРИОРИТЕТ: веб-поиск (live) выше локальной базы (RAG).",
            f"# STRICT: документ старше {STALE_YEARS} лет = УСТАРЕВШИЙ (сейчас {current_year}).",
        ]

        parts.append("\n# Уровень A — Веб-поиск (приоритетный)")
        fresh_web = [w for w in web_hits if (w.get("year") or 0) >= current_year - 3]
        old_web = [w for w in web_hits if w not in fresh_web]
        if not web_hits:
            parts.append(
                "(Веб-поиск не вернул результатов. Если нет свежих данных — "
                "честно напиши, что актуальных протоколов за последние 3 года не найдено.)"
            )
        else:
            if fresh_web:
                parts.append("## Свежие веб-результаты (2023+)")
                for w in fresh_web:
                    parts.append(
                        f"- [{w.get('lang')}] ({w.get('year')}) {w.get('title')}: "
                        f"{w.get('snippet')} | {w.get('url')}"
                    )
            if old_web:
                parts.append("## Прочие веб-результаты (могут быть устаревшими)")
                for w in old_web:
                    y = w.get("year") or "?"
                    parts.append(
                        f"- [{w.get('lang')}] ({y}) {w.get('title')}: "
                        f"{w.get('snippet')} | {w.get('url')}"
                    )

        parts.append("\n# Уровень B — Локальная база (вторична)")
        if has_stale:
            parts.append(
                "Внимание: релевантный RAG отсутствует или содержит устаревшие документы. "
                "Не строй лечение на устаревшем RAG, если есть свежий веб."
            )
        if not relevant:
            parts.append("(Релевантных документов в локальной базе нет.)")
        else:
            for h in relevant:
                tag = ""
                if h.get("stale"):
                    tag = f" [УСТАРЕЛ >{STALE_YEARS} лет — НЕ основа для лечения]"
                year = f", year={h['year']}" if h.get("year") else ", year=неизвестен"
                parts.append(
                    f"## {h['title']} ({h['path']}{year}, score={h['score']:.3f}){tag}\n{h['text']}"
                )

        return "\n\n".join(parts)

    @staticmethod
    def _user_prompt(question: str, context: str, mkb: str | None) -> str:
        mkb_line = f"Код МКБ (если указан): {mkb}\n" if mkb else ""
        return (
            f"{mkb_line}Клинический вопрос врача:\n{question}\n\n"
            f"{context}\n\n"
            "Ответь как Система Vitalis. Не упоминай модели ИИ. "
            "У каждого источника указывай реальный год. Не выдумывай даты."
        )

    @staticmethod
    def _build_card(
        question: str,
        mkb: str | None,
        rf: dict | None,
        intl: dict | None,
        rag_hits: list[dict],
        web_hits: list[dict],
    ) -> dict[str, Any]:
        title = _resolve_title(question, mkb, rag_hits, rf, intl)
        code = mkb or _guess_mkb(question) or "—"

        rf_sections = _extract_sections((rf or {}).get("content") or "")
        intl_sections = _extract_sections((intl or {}).get("content") or "")

        vault_sources = [
            {
                "type": "vault",
                "group": "rf",
                "title": h["title"],
                "ref": h["path"],
                "year": h.get("year"),
                "stale": h.get("stale", False),
            }
            for h in rag_hits[:5]
        ]
        web_sources = [
            {
                "type": "web",
                "group": "intl" if w.get("lang") == "en" else "rf",
                "title": w.get("title", ""),
                "ref": w.get("url", ""),
                "lang": w.get("lang"),
                "year": w.get("year"),
                "stale": bool(
                    w.get("year") and (datetime.now().year - int(w["year"])) >= STALE_YEARS
                ),
            }
            for w in web_hits
            if w.get("provider") != "error" and (w.get("title") or w.get("url"))
        ]

        return {
            "mkb": code,
            "title": title,
            "identity": "Система Vitalis",
            "tabs": {
                "rf": {
                    "label": "РФ · рекомендации",
                    "content": (rf or {}).get("content", ""),
                    "sections": rf_sections,
                    "sources": vault_sources
                    + [s for s in web_sources if s.get("lang") == "ru"],
                },
                "intl": {
                    "label": "Международные · рекомендации",
                    "content": (intl or {}).get("content", ""),
                    "sections": intl_sections,
                    "sources": [s for s in web_sources if s.get("lang") == "en"]
                    or web_sources[:4],
                },
            },
        }


def _parse_year(raw: str) -> int | None:
    m = re.search(r"(20\d{2}|19\d{2})", str(raw))
    return int(m.group(1)) if m else None


def _tokens(text: str) -> list[str]:
    stop = {
        "как", "что", "для", "или", "при", "это", "the", "and", "for", "with",
        "лечение", "лечить", "диагностика", "протокол", "боли",
    }
    return [
        t
        for t in re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]{3,}", (text or "").lower())
        if t not in stop
    ]


def _resolve_title(
    question: str,
    mkb: str | None,
    rag_hits: list[dict],
    rf: dict | None,
    intl: dict | None,
) -> str:
    # Не берём устаревший RAG-заголовок, если вопрос о другом
    fresh_rag = [h for h in rag_hits if not h.get("stale")]
    if fresh_rag:
        return fresh_rag[0]["title"]
    if rag_hits and not _guess_title(question):
        return rag_hits[0]["title"]
    guessed = _guess_title(question)
    if guessed:
        return guessed
    for blob in ((rf or {}).get("content") or "", (intl or {}).get("content") or ""):
        m = re.search(r"(?m)^#\s+(.+)$", blob)
        if m:
            return m.group(1).strip()[:120]
    q = re.sub(r"\s+", " ", question.strip())
    return (q[:80] + "…") if len(q) > 80 else q or "Клинический вопрос"


def _guess_title(q: str) -> str | None:
    q_low = q.lower()
    mapping = [
        (("инфаркт", "окс", "коронар"), "Острый коронарный синдром"),
        (("пневмония",), "Внебольничная пневмония"),
        (("гипертенз", "гипертония"), "Артериальная гипертензия"),
        (("перелом", "позвоночник", "позвон"), "Травма / перелом позвоночника"),
        (("геморрой", "геморр"), "Геморрой"),
    ]
    for keys, title in mapping:
        if any(k in q_low for k in keys):
            return title
    return None


def _guess_mkb(q: str) -> str | None:
    m = re.search(r"\b([A-Z]\d{2}(?:\.\d+)?)\b", q.upper())
    if m:
        return m.group(1)
    q_low = q.lower()
    if any(k in q_low for k in ("инфаркт", "окс", "коронар")):
        return "I21"
    if "пневмония" in q_low:
        return "J18"
    if "гипертенз" in q_low or "гипертония" in q_low:
        return "I10"
    if any(k in q_low for k in ("перелом", "позвоночник", "позвон")):
        return "S22"
    if "геморрой" in q_low or "геморр" in q_low:
        return "K64"
    return None


def _extract_sections(content: str) -> list[dict[str, str]]:
    """Разбор markdown-секций. Без DOTALL на заголовках — иначе GigaChat-ответ ломается."""
    if not content.strip():
        return []

    # Режем по заголовкам # .. #### (GigaChat часто даёт #### внутри ###)
    parts = re.split(r"(?m)^(#{1,4}\s+.+)$", content)
    # parts: [preamble, heading1, body1, heading2, body2, ...]
    blocks: list[tuple[str, str]] = []
    if len(parts) >= 3:
        for i in range(1, len(parts) - 1, 2):
            heading = parts[i].strip()
            body = parts[i + 1] if i + 1 < len(parts) else ""
            body = re.sub(
                r"(?is)Информация носит справочный характер.*$",
                "",
                body,
            ).strip()
            blocks.append((heading, body))
    else:
        clean = re.sub(
            r"(?is)Информация носит справочный характер.*$",
            "",
            content,
        ).strip()
        return [{"key": "all", "title": "Ответ", "body": clean}]

    sections: list[dict[str, str]] = []
    used_heads: set[int] = set()
    for key, title, pat in SECTION_DEFS:
        for idx, (heading, body) in enumerate(blocks):
            if idx in used_heads:
                continue
            # Только текст заголовка (без переноса в body)
            head_text = re.sub(r"^#{1,4}\s+", "", heading)
            if re.search(pat, head_text, flags=re.I):
                # Не цепляем общий H2, где в одной строке перечислены все разделы
                if sum(1 for _, _, p in SECTION_DEFS if re.search(p, head_text, flags=re.I)) >= 3:
                    continue
                if len(body) >= 20:
                    sections.append({"key": key, "title": title, "body": body})
                    used_heads.add(idx)
                    break

    # Если парсер собрал только «Источник: …» — отдаём полный ответ
    if sections:
        useful = [s for s in sections if not _is_source_only_stub(s["body"])]
        if useful:
            return useful

    # Fallback: все осмысленные блоки по их заголовкам
    fallback = []
    for heading, body in blocks:
        if len(body) < 20 or _is_source_only_stub(body):
            continue
        head_text = re.sub(r"^#{1,4}\s+", "", heading)
        fallback.append({"key": "block", "title": head_text[:80], "body": body})
    if fallback:
        return fallback

    clean = re.sub(
        r"(?is)Информация носит справочный характер.*$",
        "",
        content,
    ).strip()
    return [{"key": "all", "title": "Ответ", "body": clean}]


def _is_source_only_stub(body: str) -> bool:
    text = (body or "").strip()
    if not text:
        return True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    source_lines = sum(
        1
        for ln in lines
        if re.match(r"(?i)^источник\b", ln) or re.match(r"(?i)^[-*•]\s*источник\b", ln)
    )
    fact_lines = [
        ln
        for ln in lines
        if not (
            re.match(r"(?i)^источник\b", ln) or re.match(r"(?i)^[-*•]\s*источник\b", ln)
        )
    ]
    # Заглушка = есть только «Источник: …», без фактов
    if fact_lines:
        return False
    return source_lines >= 1
