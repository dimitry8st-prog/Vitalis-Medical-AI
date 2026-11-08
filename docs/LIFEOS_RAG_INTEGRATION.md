# Life-OS → Vitalis RAG

## Где находится фактический Life-OS

Рабочая база Life-OS находится в портале `st8dom-site`: runtime-БД (`LifeOSMaterial`,
`Article`, `ArticleSource`), `data/content-packages/` и `obsidian-vault/`. Репозиторий
`Life-Os/data/demo-vault` содержит только демонстрацию и не используется для вывода о
фактических материалах.

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

## Синхронизация фактической базы портала

На сервере `st8dom-site` создаётся read-only export:

```bash
python -m scripts.export_vitalis_lifeos \
  --output /private/path/lifeos-vitalis.json
```

Затем на стороне Vitalis:

```bash
PYTHONPATH=backend python -m app.rag.portal_lifeos_import \
  /private/path/lifeos-vitalis.json vault
```

В RAG попадают только опубликованные медицинские статьи по неврологии и
нейрохирургии с рецензентом, датой проверки и HTTPS-источниками. Сырые записи потока
`science-neuro` остаются в manifest как `needs_medical_review` и не индексируются.

## Синхронизация отдельного Obsidian-vault

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
PYTHONPATH=backend python -m unittest \
  backend/tests/test_lifeos_import.py \
  backend/tests/test_portal_lifeos_import.py -v
```

До production проверяется manifest, контрольный клинический вопрос, отсутствие
конфликта с официальной КР и корректное отображение происхождения источника.

Локальная контрольная проверка 2026-11-07: экспортировано и импортировано 10/10
опубликованных рецензированных статей, отклонено 0. Поиск и карточка проверены по
мигрени, реабилитации после ишемического инсульта и ботулинотерапии при
орофациальной боли. Это результат локальной заполненной базы портала; записи
фактической production-БД подтверждаются только отдельным read-only экспортом на VPS.

Production-проверка 2026-11-08 выполнена после успешного деплоя портала. Закрытый
export сохранён на VPS в `/srv/st8dom/private/lifeos-vitalis.json` с правами `600`:
10 записей `approved_records` и 10 сырых `candidates`. Кандидаты не содержат тела в
export, имеют статус `needs_medical_review` и не должны импортироваться до ручной
врачебной проверки и публикации через редакционный процесс портала.
