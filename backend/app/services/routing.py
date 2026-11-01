from __future__ import annotations

import re
from typing import Literal

Intent = Literal["greeting", "skills", "tech", "offtopic", "unsafe", "vague", "clinical"]

GREETING_RE = re.compile(
    r"^\s*(привет|здравствуй(те)?|добрый\s+(день|вечер|утро)|hi|hello|hey)\s*[!.]?$",
    re.I,
)
SKILLS_RE = re.compile(
    r"(что\s+ты\s+умеешь|какие\s+функц|что\s+можешь|skillset|возможности\s+систем)",
    re.I,
)
TECH_RE = re.compile(
    r"(как\s+работает\s+поиск|откуда\s+данные|кто\s+ты|что\s+такое\s+vitalis|как\s+ты\s+ишешь)",
    re.I,
)
TEST_RE = re.compile(r"^\s*(просто\s+)?тест\s*[!.]?$", re.I)
VAGUE_RE = re.compile(
    r"^\s*(как\s+жить|что\s+делать|помоги|ну\s+и|ок|хорошо|да|нет)\s*[!.]?$",
    re.I,
)
UNSAFE_RE = re.compile(
    r"(взлом|hack|suicid|самоповрежд|убить|наркотик|bomb|оружие|насилие)",
    re.I,
)
OFFTOPIC_RE = re.compile(
    r"(стих|анекдот|погод|политик|религ|wi-?fi|код\s+на\s+|python|javascript|любов|рецепт|кулинар|финанс|юридич)",
    re.I,
)
CLINICAL_HINT_RE = re.compile(
    r"(лечен|диагност|протокол|рекомендац|мкб|симптом|доз|терап|клинич|болезн|синдром|"
    r"геморр|пневмо|гипертен|инфаркт|окс|перелом|боль|анализ|шкал|пациент|нозолог|"
    r"туберкул|диабет|инсульт|онколог|астма|бронхит|что\s+такое|чем\s+лечить)",
    re.I,
)

# Известные нозологии / темы одним словом → всегда clinical
DISEASE_ONEWORD = {
    "туберкулез", "туберкулёз", "геморрой", "пневмония", "гипертония", "гипертензия",
    "диабет", "инсульт", "астма", "бронхит", "онкология", "грипп", "ангина",
}

GREETING_ANSWER = (
    "Здравствуйте, коллега! Я — медицинский ассистент Vitalis. "
    "Готов помочь с поиском актуальных клинических рекомендаций (РФ и международные), "
    "анализом симптомов или подбором терапии. Что вас интересует сегодня?"
)

SKILLS_ANSWER = (
    "Я могу:\n"
    "1. Найти действующие клинические рекомендации и проверить их статус в веб-источниках.\n"
    "2. Сравнить протоколы Минздрава РФ и международные (ESC, NICE и др.).\n"
    "3. Помочь с дифференциальной диагностикой.\n"
    "4. Сориентировать по шкалам и рискам (с опорой на источники).\n\n"
    "Просто задайте вопрос или укажите МКБ-10."
)

TECH_ANSWERS = {
    "search": (
        "Сначала я проверяю локальную базу знаний, затем по умолчанию делаю запрос "
        "в интернет для проверки актуальности данных. В API веб-поиск можно отключить; "
        "тогда ответ будет явно помечен как не прошедший веб-проверку."
    ),
    "sources": (
        "Из утверждённых клинических рекомендаций РФ и международных ассоциаций. "
        "Возраст документа сам по себе не отменяет его: система проверяет статус, версию "
        "и наличие заменяющего первоисточника."
    ),
    "who": (
        "Я — Система Vitalis, медицинский ассистент для врачей и студентов. "
        "Помогаю находить и сравнивать актуальные клинические рекомендации РФ и "
        "международные протоколы. Не ставлю диагноз конкретному пациенту — решение "
        "принимает лечащий врач."
    ),
}

OFFTOPIC_ANSWER = (
    "Я специализируюсь исключительно на медицинских данных и клинических рекомендациях. "
    "По другим темам я не могу дать квалифицированный ответ. "
    "Если у вас есть вопрос по диагностике, лечению или протоколам — я готов помочь."
)

UNSAFE_ANSWER = (
    "Данный запрос выходит за рамки моей компетенции как медицинского ассистента. "
    "Я могу предоставить информацию только на основе доказательной медицины и "
    "утверждённых клинических протоколов."
)

VAGUE_ANSWER = (
    "Уточните, пожалуйста, ваш запрос в медицинском контексте. Например: "
    "«Протокол лечения гипертонии», «Дифференциальная диагностика боли в груди» "
    "или «МКБ-10 код для…»."
)

TEST_ANSWER = (
    "Система работает исправно. Готов ответить на вопросы по клиническим рекомендациям "
    "РФ и международным протоколам."
)


def classify_intent(question: str) -> Intent:
    q = (question or "").strip()
    if not q:
        return "vague"
    if GREETING_RE.match(q):
        return "greeting"
    if TEST_RE.match(q):
        return "vague"  # handled as test template
    q_low = q.lower().strip(" ?!.")
    if q_low in DISEASE_ONEWORD:
        return "clinical"
    if UNSAFE_RE.search(q) and not CLINICAL_HINT_RE.search(q):
        return "unsafe"
    if SKILLS_RE.search(q):
        return "skills"
    if TECH_RE.search(q):
        return "tech"
    if OFFTOPIC_RE.search(q) and not CLINICAL_HINT_RE.search(q):
        return "offtopic"
    if CLINICAL_HINT_RE.search(q):
        return "clinical"
    if len(q) >= 12 and len(q.split()) >= 2:
        return "clinical"
    if len(q.split()) <= 2 and not CLINICAL_HINT_RE.search(q):
        return "vague"
    if OFFTOPIC_RE.search(q):
        return "offtopic"
    return "clinical"


def canned_answer(question: str, intent: Intent) -> str | None:
    q = (question or "").strip().lower()
    if intent == "greeting":
        return GREETING_ANSWER
    if intent == "skills":
        return SKILLS_ANSWER
    if intent == "tech":
        if "поиск" in q or "ищ" in q:
            return TECH_ANSWERS["search"]
        if "откуда" in q or "данн" in q:
            return TECH_ANSWERS["sources"]
        return TECH_ANSWERS["who"]
    if intent == "offtopic":
        return OFFTOPIC_ANSWER
    if intent == "unsafe":
        return UNSAFE_ANSWER
    if TEST_RE.match(question or ""):
        return TEST_ANSWER
    if intent == "vague":
        return VAGUE_ANSWER
    return None


def is_nonclinical(intent: Intent) -> bool:
    return intent in {"greeting", "skills", "tech", "offtopic", "unsafe", "vague"}
