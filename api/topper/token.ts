import { randomUUID } from 'node:crypto';
import { importJWK, SignJWT } from 'jose';

const ALLOWED_ORIGINS = new Set([
  'https://btc2g.com',
  'https://www.btc2g.com',
  'https://vitalydz.github.io',
]);

const TOPPER_APP_URLS = {
  sandbox: 'https://app.sandbox.topperpay.com/',
  production: 'https://app.topperpay.com/',
};

function setCorsHeaders(req: any, res: any) {
  const origin = req.headers?.origin;

  if (typeof origin === 'string' && ALLOWED_ORIGINS.has(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
  }

  res.setHeader('Vary', 'Origin');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  res.setHeader('Content-Type', 'application/json');
}

function sendJson(res: any, status: number, body: Record<string, unknown>) {
  return res.status(status).json(body);
}

function parseBody(req: any) {
  if (!req.body) {
    return {};
  }

  if (typeof req.body === 'string') {
    try {
      return JSON.parse(req.body);
    } catch {
      return {};
    }
  }

  return req.body;
}

function readString(value: unknown, fallback: string) {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed || fallback;
  }

  if (typeof value === 'number' && Number.isFinite(value)) {
    return String(value);
  }

  return fallback;
}

function getTopperEnvironment() {
  return process.env.TOPPER_ENV === 'production' ? 'production' : 'sandbox';
}

function getPrivateJwk() {
  const rawJwk = process.env.TOPPER_PRIVATE_JWK;

  if (!rawJwk) {
    throw new Error('Missing private JWK');
  }

  return JSON.parse(rawJwk);
}

export default async function handler(req: any, res: any) {
  setCorsHeaders(req, res);

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  if (req.method !== 'POST') {
    return sendJson(res, 405, { error: 'Method not allowed. Use POST.' });
  }

  const widgetId = process.env.TOPPER_WIDGET_ID;
  const keyId = process.env.TOPPER_KEY_ID;

  if (!widgetId || !keyId || !process.env.TOPPER_PRIVATE_JWK) {
    return sendJson(res, 500, { error: 'Topper server configuration is incomplete.' });
  }

  const body = parseBody(req);
  const address = readString(body.address, '');

  if (!address) {
    return sendJson(res, 400, { error: 'BTC address is required.' });
  }

  const amount = readString(body.amount, '50.00');
  const fiat = readString(body.fiat, 'USD').toUpperCase();
  const asset = readString(body.asset, 'BTC').toUpperCase();
  const network = readString(body.network, 'bitcoin').toLowerCase();
  const label = readString(body.label, 'BTC2G wallet');
  const environment = getTopperEnvironment();

  try {
    const privateJwk = getPrivateJwk();
    const privateKey = await importJWK(privateJwk, 'ES256');
    const iat = Math.floor(Date.now() / 1000);

    const bt = await new SignJWT({
      iat,
      jti: randomUUID(),
      sub: widgetId,
      source: {
        amount,
        asset: fiat,
      },
      target: {
        address,
        asset,
        network,
        label,
      },
    })
      .setProtectedHeader({
        typ: 'JWT',
        alg: 'ES256',
        kid: keyId,
      })
      .sign(privateKey);

    return sendJson(res, 200, {
      bt,
      appUrl: TOPPER_APP_URLS[environment],
    });
  } catch {
    return sendJson(res, 500, { error: 'Unable to create Topper bootstrap token.' });
  }
}
