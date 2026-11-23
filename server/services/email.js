'use strict';

const nodemailer = require('nodemailer');

async function sendEmail(lead) {
  const host = (process.env.SMTP_HOST || '').trim();
  const to = (process.env.LEADS_EMAIL_TO || '').trim();
  if (!host || !to) {
    return { ok: false, skipped: true, reason: 'SMTP_HOST / LEADS_EMAIL_TO не заданы' };
  }

  const port = Number(process.env.SMTP_PORT || 587);
  const secure = String(process.env.SMTP_SECURE || 'false') === 'true';
  const user = process.env.SMTP_USER || '';
  const pass = process.env.SMTP_PASS || '';

  const transporter = nodemailer.createTransport({
    host,
    port,
    secure,
    auth: user ? { user, pass } : undefined,
  });

  const subject = `Заявка Vitalis: ${lead.name} (${lead.role || 'без роли'})`;
  const text = [
    `Имя: ${lead.name}`,
    `Email: ${lead.email}`,
    `Роль: ${lead.role || '—'}`,
    `Интерес: ${lead.tariff || '—'}`,
    `Комментарий: ${lead.comment || '—'}`,
    `Время: ${lead.createdAt}`,
    `ID: ${lead.id}`,
  ].join('\n');

  await transporter.sendMail({
    from: process.env.SMTP_FROM || 'Vitalis <noreply@vitalis.local>',
    to,
    replyTo: lead.email,
    subject,
    text,
  });

  return { ok: true, skipped: false };
}

module.exports = { sendEmail };
