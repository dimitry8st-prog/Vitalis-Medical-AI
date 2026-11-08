# Vitalis project memory

Read this file before auditing, changing, testing, or deploying Vitalis.

## Authoritative repositories and data

- Vitalis application: `dimitry8st-prog/Vitalis-Medical-AI`.
- Portal and the actual current Life-OS workspace: `dimitry8st-prog/st8dom-site`.
- The portal runtime stores raw Life-OS records in `LifeOSMaterial` and portal
  publications in `Article` plus `ArticleSource`.
- Versioned portal material lives in `st8dom-site/data/content-packages/` and
  `st8dom-site/obsidian-vault/`.
- `dimitry8st-prog/Life-Os/data/demo-vault` is only a demonstration vault.

## Non-negotiable Vitalis RAG policy

- Owner decision dated 2026-11-08: Vitalis RAG accepts **only verified full texts of
  clinical guidelines**.
- An indexable file must have
  `document_type=clinical_guideline_full_text`,
  `content_scope=official_full_text_extracted`, and `full_text_checked=true`.
- Bibliographic cards may help discover a document, but are not indexed and are not
  medical evidence.
- Portal articles, scientific news, reviews, patient materials, Life-OS notes, and raw
  candidates are portal-only content. They never enter Vitalis RAG, even if published or
  medically reviewed.
- Search must work from a free clinical query, a diagnosis, an ICD/MKB code, or their
  combination. An explicit ICD/MKB code is included in local retrieval ranking.
- Do not delete or change the review status of Life-OS candidates unless Dmitry asks for
  that exact action. The 10 production candidates remain unchanged.

## Current facts

- Production portal export on 2026-11-08 previously reported 10 published portal
  articles and 10 raw `needs_medical_review` candidates.
- The portal export contract is being narrowed to schema 2: `approved_records` stays
  empty, portal articles are counted as excluded, and candidates remain inventory only.
- The repository still contains legacy generated portal-article Markdown files. They are
  preserved for traceability but index schema 2 excludes them from FAISS.
- The local workspace currently has 17 official Russian guideline metadata cards and
  **0/17 extracted full-text Markdown documents**. Therefore a rebuilt strict RAG is
  intentionally empty until the official PDF sync succeeds.
- Draft portal article
  `brain-options-decision-2026-11.json` is marked `workflow_status=review`,
  `portal_only=true`, and `vitalis_ingest_allowed=false`. It is not published.
- Vitalis production work remains in draft PR #17 on branch
  `codex/vitalis-lifeos-production`.

## Safety boundary

- Never write from Vitalis into the portal or Life-OS database.
- Never treat metadata, a search snippet, or a portal article as full guideline text.
- Exact doses require literal support in a retrieved, verified guideline excerpt.
- No automatic publication, candidate rejection, or ingestion from Life-OS.

## Resume checklist

1. Complete the official Russian clinical-guideline PDF sync to 17/17.
2. Rebuild the local RAG and confirm that only verified full guideline files were indexed.
3. Smoke-test searches by diagnosis, MKB code, and arbitrary clinical question, including
   page citations and strict blocking when no full text exists.
4. Run the full Python, Node, audit, and container checks.
5. Keep PR #17 draft until the clinical release gate and backup/restore drill pass.
