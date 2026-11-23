'use strict';

async function sendTelegram(lead) {
  const token = (process.env.TELEGRAM_BOT_TOKEN || '').trim();
  const chatId = (process.env.TELEGRAM_CHAT_ID || '').trim();
  if (!token || !chatId) {
    return { ok: false, skipped: true, reason: 'TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID не заданы' };
  }

  const text = [
    '🩺 <b>Новая заявка Vitalis</b>',
    '',
    `<b>Имя:</b> ${escapeHtml(lead.name)}`,
    `<b>Email:</b> ${escapeHtml(lead.email)}`,
    `<b>Роль:</b> ${escapeHtml(lead.role || '—')}`,
    `<b>Интерес:</b> ${escapeHtml(lead.tariff || '—')}`,
    lead.comment ? `<b>Комментарий:</b> ${escapeHtml(lead.comment)}` : null,
    '',
    `<i>${escapeHtml(lead.createdAt)}</i>`,
  ]
    .filter(Boolean)
    .join('\n');

  const url = `https://api.telegram.org/bot${token}/sendMessage`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      chat_id: chatId,
      text,
      parse_mode: 'HTML',
      disable_web_page_preview: true,
    }),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data.ok) {
    return { ok: false, skipped: false, reason: data.description || `HTTP ${res.status}` };
  }
  return { ok: true, skipped: false };
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

module.exports = { sendTelegram };
