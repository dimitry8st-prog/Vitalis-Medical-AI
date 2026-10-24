'use strict';
const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');

let stub, server, base;
const requests = [];
before(async () => {
  stub = http.createServer((req, res) => {
    const chunks = [];
    req.on('data', chunk => chunks.push(chunk));
    req.on('end', () => {
      const record = { path: req.url, method: req.method, headers: req.headers, body: Buffer.concat(chunks).toString() };
      requests.push(record);
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Cache-Control', 'public, max-age=999');
      res.end(JSON.stringify(record));
    });
  });
  await new Promise(resolve => stub.listen(0, '127.0.0.1', resolve));
  process.env.PYTHON_API_URL = `http://127.0.0.1:${stub.address().port}`;
  process.env.YANDEX_METRIKA_ID = '123456';
  const { app } = require('../index');
  await new Promise(resolve => { server = app.listen(0, '127.0.0.1', resolve); });
  base = `http://127.0.0.1:${server.address().port}`;
});
after(async () => {
  await Promise.all([new Promise(resolve => server.close(resolve)), new Promise(resolve => stub.close(resolve))]);
});

test('cabinet has no analytics or third-party assets and forbids caching', async () => {
  for (const path of ['/app', '/app/']) {
    const response = await fetch(base + path);
    const text = await response.text();
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('cache-control'), 'no-store');
    assert.equal(response.headers.get('referrer-policy'), 'no-referrer');
    assert.match(response.headers.get('content-security-policy'), /connect-src 'self'/);
    assert.doesNotMatch(text, /mc\.yandex\.ru|fonts\.googleapis\.com|webvisor:true/);
  }
});

test('JSON API proxy preserves paths, methods and payloads', async () => {
  for (const path of ['/ask', '/mode', '/rag/rebuild', '/documents/token/questions']) {
    const response = await fetch(base + '/api/ai' + path, { method: 'POST', headers: {
      'Content-Type': 'application/json', Authorization: 'Bearer synthetic', Cookie: 'synthetic=value'
    }, body: JSON.stringify({ question: 'synthetic' }) });
    const record = await response.json();
    assert.equal(record.path, path);
    assert.equal(record.method, 'POST');
    assert.equal(record.body, '{"question":"synthetic"}');
    assert.equal(record.headers.authorization, undefined);
    assert.equal(record.headers.cookie, undefined);
    assert.equal(response.headers.get('cache-control'), 'no-store');
  }
});

test('multipart PDF and deletion reach the backend unchanged', async () => {
  const form = new FormData();
  form.append('file', new Blob(['%PDF-synthetic'], { type: 'application/pdf' }), 'synthetic.pdf');
  form.append('confirmed_deidentified', 'true');
  const response = await fetch(base + '/api/ai/documents/analyze', { method: 'POST', body: form });
  const record = await response.json();
  assert.equal(record.path, '/documents/analyze');
  assert.match(record.headers['content-type'], /multipart\/form-data/);
  assert.match(record.body, /%PDF-synthetic/);
  assert.match(record.body, /confirmed_deidentified/);
  const removed = await fetch(base + '/api/ai/documents/token', { method: 'DELETE' });
  assert.equal((await removed.json()).method, 'DELETE');
});

test('health and landing remain available', async () => {
  assert.equal((await fetch(base + '/api/health')).status, 200);
  assert.equal((await fetch(base + '/api/ai/health')).status, 200);
  const response = await fetch(base + '/');
  assert.equal(response.status, 200);
  assert.match(await response.text(), /mc\.yandex\.ru/);
});
