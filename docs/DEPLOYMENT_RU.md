# Vitalis RU: VPS deployment and operations

## Recorded deployment — 16 October 2026

Private pilot: https://vitalis.st8dom.ru/app. HTTPS and HTTP Basic authentication are enabled by host Caddy. The landing page is at `/`. Login is `dmitry`; the password is stored privately by the operator.

Application baseline: `9cfb784403e8dead895fbae3c6be6cb84fefbe3c` (main). This documentation/configuration commit does not change application logic, prompts, or the INT branch. The existing application still contains both region choices; deployment does not enforce RU-only selection.

Environment observed during the operator-guided session: Timeweb VPS, Ubuntu 26.04, Docker 29.7.2, Compose 5.5.0, Caddy 2.6.2; approximately 889 MiB RAM and a 1 GiB swap file. Checkout: `/opt/vitalis-ru`. Public DNS is managed at Beget, not in the empty Timeweb domains list.

## Layout and configuration

- `Dockerfile.node`: Node 22, Express landing/app/API proxy, 128 MiB memory cap.
- `Dockerfile.python`: Python 3.12, FastAPI, one worker, 384 MiB memory cap.
- `compose.yaml`: safe mock default; publishes Node only on `127.0.0.1:3080`. Python is accessible inside the Compose network on port 8000, not published to the host.
- Named volumes: `vitalis-ru_rag_data` at `/data` and `vitalis-ru_leads_data` at `/app/server/data`.
- JSON Docker logs rotate at 5 MiB, two files per service. Both services restart unless stopped. No Docker healthcheck is configured; `depends_on` does not guarantee API readiness.
- `deploy/compose.live.example.yaml`: reproduces the live override used on the VPS. Copy locally to `compose.override.yaml` only after creating `/root/.config/vitalis-ru/python.env` with the real key. This file is outside Git and has mode 600; parent directory mode 700. Do not print `docker compose config` without `--quiet` when secrets are loaded.
- GigaChat is disabled. DeepSeek is enabled in live mode.

The deployed Dockerfiles use base-image tags and `npm install`, matching the successful server build. They are not immutable/reproducible image pins. Before future updates, retain the actual working image IDs or local rollback tags.

## Recreate a new private instance

These instructions are for a prepared host, not a script to overwrite an existing deployment.

1. Clone the repository, choose the intended commit, and enter the checkout.
2. Inspect current containers and host ports. Keep other projects running.
3. Build: `docker compose build`; start mock services: `docker compose up -d --no-build`.
4. Verify local Node `/api/health`, Python `/api/ai/health`, and `/app` on `http://127.0.0.1:3080`.
5. Create the private env file using a hidden prompt, never pasting the key into command text:

```bash
mkdir -p -m 700 /root/.config/vitalis-ru
read -rsp "DeepSeek key: " VITALIS_DEEPSEEK_KEY; echo
(umask 077; printf 'DEEPSEEK_API_KEY=%s\n' "$VITALIS_DEEPSEEK_KEY" > /root/.config/vitalis-ru/python.env) && unset VITALIS_DEEPSEEK_KEY
cp deploy/compose.live.example.yaml compose.override.yaml
docker compose config --quiet && docker compose up -d --no-build --no-deps python
```

6. Configure one A record for the chosen hostname pointing to the VPS. Remove unintended duplicate A records and check AAAA records against the actual server. For this deployment the final authoritative A lookup returned only `72.56.76.107`.
7. Back up the host Caddyfile. Generate a private bcrypt hash:

```bash
(umask 077; caddy hash-password --algorithm bcrypt > /root/.config/vitalis-ru/access.hash)
```

8. Prepare a candidate Caddyfile preserving all existing sites. Add the site from `deploy/Caddyfile.vitalis.example` and substitute the hash locally without publishing it. The template is intentionally not ready for use until that placeholder is replaced. Caddy 2.6.2 uses `basicauth`; newer releases from 2.8 use `basic_auth`.
9. Validate the candidate with `caddy validate --config PATH --adapter caddyfile`. Only after success, install it at `/etc/caddy/Caddyfile` and run `systemctl reload caddy`. Check the service status and journal if reload fails. The real server retained `st8dom.ru` and `www.st8dom.ru` pointing to `127.0.0.1:8080`.
10. Confirm certificate issuance, authentication, UI, and a control answer through the hostname. No public opening of port 3080 or 8000 is required.

The site block authenticates all paths and blocks external RAG rebuilding, including for authenticated users. It removes the Basic Authorization header before forwarding to Node. Keep the SSH tunnel optional; normal browser use through the domain does not need it.

## Verification recorded during this session

Evidence is the operator's terminal outputs, screenshots, and explicit confirmations in the session; this is not an independent automated acceptance report.

| Check | Observed result |
| --- | --- |
| Docker images | Both images built; both services started |
| Local app | `/app`: HTTP 200, HTML |
| Python health after live override | `llm_mode=live`, `deepseek=true`, `keys_ready=true`, `gigachat=false`, `vault_chunks=13` |
| Live generation through SSH tunnel | A control answer was displayed |
| Authoritative DNS | One A record, the intended VPS |
| Caddy candidate | `Valid configuration`; existing forwarded-header/format warnings were non-blocking |
| Domain API without credentials | HTTP 401, confirmed by Dmitry at 23:01 MSK |
| Existing portal | HTTPS HTTP 200, confirmed at 23:01 MSK |
| Domain UI and answer after login | User reported working at 23:02 MSK |
| Authenticated RAG rebuild block | Configured, not separately exercised |
| Cold boot / restore from backup | Not tested |

A health response confirms configuration/key presence; successful generation provides additional live-path evidence. Neither establishes medical accuracy or source quality.

## Backup and rollback

The original `/etc/caddy/Caddyfile` was copied before editing (timestamped `Caddyfile.before-vitalis-*`), and another copy was made at `/etc/caddy/Caddyfile.before-vitalis-enable`. `/etc/fstab.before-vitalis-swap` preserves the prior swap configuration.

For future releases, before rebuilding record `docker compose images` and tag both currently working image IDs with a local rollback tag. Keep a private copy of the Compose files and private env settings. Application volume backups have not yet been created or restore-tested in this session.

For a consistent volume backup during a maintenance window, stop only the Vitalis services, archive both named volumes to an operator-controlled private location, then restart the services and recheck health. Include the exact application version and image IDs in backup metadata. Never run `docker compose down -v` as a rollback: it destroys named-volume data. Do not publish env files, user submissions, or volume backups.

To remove this deployment from public access while preserving other sites, validate the known pre-Vitalis Caddyfile with `caddy validate --config /etc/caddy/Caddyfile.before-vitalis-enable --adapter caddyfile`, restore it, and reload Caddy. Then, from `/opt/vitalis-ru`, run `docker compose stop` to stop only this project. This is a shutdown rollback; image/data restoration is a separate operation. Inspect the current file before restoring an older backup if other sites were changed meanwhile. Swap removal is not required to roll back the app.

## Operations and remaining work

```bash
cd /opt/vitalis-ru
docker compose ps
docker compose logs --tail=80 node python
curl -fsS http://127.0.0.1:3080/api/ai/health
curl -sS -o /dev/null -w '%{http_code}\n' https://vitalis.st8dom.ru/api/ai/health
curl -sS -o /dev/null -w '%{http_code}\n' https://st8dom.ru/
```

Expected public unauthenticated Vitalis status is 401; existing portal is 200. Inspect logs privately before sharing them.

Search quality improvements were postponed until after deployment. Irrelevant web results and sensitivity to the selected ICD code were observed; validation against primary Russian guidelines, citation quality, and regional isolation remain open. The 13 indexed chunks are a small initial corpus, not a complete medical knowledge base. A private deployment and HTTPS do not establish 152-FZ compliance or clinical readiness.

## Time record (MSK)

- 18:18 — preparation started.
- Around 20:18 — server command work underway after a possible gap; exact active/pause boundaries were not recorded.
- 22:05 — first observed live control response through the tunnel.
- 22:46 — final authoritative DNS result verified.
- 22:56 — Caddy candidate validation passed.
- 23:01 — user confirmed 401 / 200 checks.
- 23:02 — user confirmed the assistant works through the domain.

Elapsed preparation-to-confirmation window: approximately 4 h 44 min. This includes waiting and possible breaks; active labor time cannot be derived precisely.
