# Vitalis project memory

Read this file before auditing, changing, testing, or deploying Vitalis.

## Authoritative repositories and data

- Vitalis application: `dimitry8st-prog/Vitalis-Medical-AI`.
- Portal and the actual current Life-OS workspace: `dimitry8st-prog/st8dom-site`.
- The portal runtime stores Life-OS records in its configured SQLAlchemy database
  (`DATABASE_URL`; local fallback `instance/site.db`) in `LifeOSMaterial` and stores
  medically reviewed publications in `Article` plus `ArticleSource`.
- Versioned Life-OS/editorial material in the portal repository lives in
  `data/content-packages/` and `obsidian-vault/`.
- `dimitry8st-prog/Life-Os/data/demo-vault` is only a demonstration vault. Never use
  it to conclude that the real Life-OS database is empty or that today's materials
  were not added.

## Current facts

- Dmitry stated that medical materials were added to Life-OS on 2026-11-07.
- Before reporting import counts, inspect the actual portal database or a controlled
  export from it, plus versioned portal packages. Do not report only demo-vault counts.
- Vitalis production work is in draft PR #17 on branch
  `codex/vitalis-lifeos-production`.
- GitHub CI for PR #17 passed Python, Node, dependency audit, and container smoke.
- The local seeded portal database exports 10 reviewed articles; Vitalis imports all
  10 with no rejection and exposes them as secondary reviewed Life-OS sources.
- This local count does not prove which runtime records were added on 2026-11-07;
  verify those with the same read-only exporter against the VPS database.
- The private production release remains blocked until the official Russian guideline
  manifest is complete (17/17), the reviewed Life-OS export is imported, the clinical
  review report passes, and backup/restore is tested.

## Integration boundary

- Life-OS to Vitalis is one-way and offline: export -> staging -> validation -> RAG.
- Never write back to the portal database from Vitalis.
- Import only medically reviewed/published materials with provenance and no personal data.
- Official clinical guideline full text always outranks a Life-OS article or note.
- A Life-OS review cannot validate an exact dose unless the retrieved official primary
  source explicitly contains it.

## Resume checklist

1. Inspect `st8dom-site` versioned packages and the current portal DB/export.
2. Import only approved neurology/neurosurgery records into generated Vitalis RAG data.
3. Verify a normal `/ask` request, source provenance, page citations, and strict blocking.
4. Run all local tests and GitHub CI.
5. Keep the PR draft until every production gate passes.
