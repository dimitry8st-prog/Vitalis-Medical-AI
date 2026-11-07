# Life-OS → Vitalis RAG

## Граница интеграции

Life-OS остаётся владельцем исходных заметок. Vitalis читает явный каталог Obsidian
только офлайн, формирует собственную каноническую копию и никогда автоматически не
изменяет Life-OS. Runtime Vitalis не получает доступ к базе SQLite или vault Life-OS.

Порядок доверия:

1. официальный полный текст КР Минздрава;
2. другие проверенные первичные источники;
3. одобренный человеком фрагмент Life-OS;
4. библиографическая карточка — только для обнаружения документа.

Заметка Life-OS не может отменить или переопределить КР.

## Контракт одобренной заметки

В заметке должен быть frontmatter:

```yaml
---
title: Проверенный обзор по теме
source_id: stable-neuro-note-001
source_url: https://example.org/primary-or-reviewed-source
source_version: v1
year: 2026
jurisdiction: RU
language: ru
specialty_tags: [neurology]
content_scope: reviewed_excerpt
vitalis_ingest: true
review_status: approved
reviewed_by: ФИО или роль ответственного врача
reviewed_at: 2026-11-07
---
```

Разрешены только профили `neurology` и `neurosurgery` и только scope
`reviewed_excerpt`. Обязательны стабильный ID, версия, публичный HTTPS-источник,
год, юрисдикция, рецензент и дата проверки.

Импорт отклоняет заметку, если она:

- не имеет явного одобрения;
- содержит prompt injection;
- содержит персональные или страховые идентификаторы;
- дублирует `source_id`;
- не имеет проверяемого происхождения;
- пытается объявить себя официальным полным текстом.

## Синхронизация

```bash
PYTHONPATH=backend python -m app.rag.lifeos_import \
  /absolute/path/to/life-os/data/vault vault
```

Канонические копии и manifest создаются в `vault/generated/lifeos/`. Если одобрение
отозвано или заметка удалена, соответствующая копия удаляется при следующей
синхронизации. После импорта перезапустите API или вызовите локальный rebuild RAG в
контролируемой среде.

## Проверка

```bash
PYTHONPATH=backend python -m unittest backend/tests/test_lifeos_import.py -v
```

До production проверяется manifest, контрольный клинический вопрос, отсутствие
конфликта с официальной КР и корректное отображение происхождения источника.
