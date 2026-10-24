'use strict';

const path = require('path');
const fs = require('fs');
const http = require('http');
const https = require('https');
const express = require('express');
const dotenv = require('dotenv');

dotenv.config({ path: path.join(__dirname, '..', '.env') });

const formRouter = require('./routes/form');
const { injectMetrika } = require('./services/metrika');

const PORT = Number(process.env.PORT || 3000);
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://127.0.0.1:8000';
const PUBLIC_DIR = path.join(__dirname, '..', 'public');
const WEB_DIR = path.join(__dirname, '..', 'web');
const pythonTarget = new URL(PYTHON_API_URL);

if (!['http:', 'https:'].includes(pythonTarget.protocol)) {
  throw new Error('PYTHON_API_URL должен использовать http или https');
}

const app = express();
app.disable('x-powered-by');
app.use((_req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
  next();
});

app.get('/api/health', (_req, res) => {
  res.json({
    ok: true,
    service: 'vitalis-node',
    metrika: Boolean(process.env.YANDEX_METRIKA_ID),
    python: PYTHON_API_URL,
  });
});

const HOP_BY_HOP_HEADERS = new Set([
  'connection',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
]);

function withoutHopByHopHeaders(headers) {
  const nominated = new Set(String(headers.connection || '').toLowerCase().split(',').map(value => value.trim()));
  const filtered = Object.fromEntries(
    Object.entries(headers).filter(([name]) => !HOP_BY_HOP_HEADERS.has(name.toLowerCase()) && !nominated.has(name.toLowerCase()))
  );
  delete filtered.authorization;
  delete filtered.cookie;
  return filtered;
}

// Потоковый прокси стоит до json-parser: multipart и JSON передаются в FastAPI без буферизации.
app.use('/api/ai', (req, res) => {
  res.setHeader('Cache-Control', 'no-store');
  const transport = pythonTarget.protocol === 'https:' ? https : http;
  const basePath = pythonTarget.pathname.replace(/\/$/, '');
  const requestPath = req.url.startsWith('/') ? req.url : `/${req.url}`;
  const headers = withoutHopByHopHeaders(req.headers);
  headers.host = pythonTarget.host;

  const proxyReq = transport.request(
    {
      protocol: pythonTarget.protocol,
      hostname: pythonTarget.hostname,
      port: pythonTarget.port || undefined,
      method: req.method,
      path: `${basePath}${requestPath}`,
      headers,
    },
    (proxyRes) => {
      res.writeHead(
        proxyRes.statusCode || 502,
        { ...withoutHopByHopHeaders(proxyRes.headers), 'cache-control': 'no-store' }
      );
      proxyRes.on('error', () => res.destroy());
      proxyRes.pipe(res);
    }
  );

  // Search plus two sequential model calls may legitimately exceed one minute.
  const deadline = setTimeout(() => proxyReq.destroy(new Error('Python API deadline')), 240_000);
  proxyReq.setTimeout(240_000, () => {
    proxyReq.destroy(new Error('Python API timeout'));
  });
  proxyReq.on('error', (err) => {
    console.error(`Python proxy error: ${err.code || err.name || 'unknown'}`);
    if (!res.headersSent) {
      res.status(502).json({
        error: 'Python API недоступен',
        detail: 'Соединение с внутренним Python API не установлено',
        hint: 'Запустите Python API на настроенном PYTHON_API_URL',
      });
    } else {
      res.destroy(err);
    }
  });
  req.on('aborted', () => proxyReq.destroy());
  res.on('close', () => {
    clearTimeout(deadline);
    if (!res.writableFinished) proxyReq.destroy();
  });
  req.pipe(proxyReq);
});

app.use(express.json({ limit: '1mb' }));
app.use('/api/leads', formRouter);

function sendHtml(filePath, res, privatePage = false) {
  fs.readFile(filePath, 'utf8', (err, html) => {
    if (err) {
      res.status(404).send('Not found');
      return;
    }
    if (privatePage) {
      res.setHeader('Cache-Control', 'no-store');
      res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'");
    }
    res.type('html').send(privatePage ? html : injectMetrika(html));
  });
}

app.get('/', (_req, res) => sendHtml(path.join(PUBLIC_DIR, 'index.html'), res));
app.get('/app', (_req, res) => sendHtml(path.join(WEB_DIR, 'app.html'), res, true));
app.get('/app/', (_req, res) => sendHtml(path.join(WEB_DIR, 'app.html'), res, true));

app.use('/assets', express.static(path.join(WEB_DIR, 'assets')));
app.use(express.static(PUBLIC_DIR));

if (require.main === module) app.listen(PORT, () => {
  console.log(`Vitalis Node: http://localhost:${PORT}`);
  console.log(`Лендинг:      http://localhost:${PORT}/`);
  console.log(`Продукт:      http://localhost:${PORT}/app`);
  console.log(`AI proxy →    ${PYTHON_API_URL}`);
});

module.exports = { app };
