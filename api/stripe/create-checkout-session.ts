import Stripe from 'stripe';

const ALLOWED_ORIGINS = new Set([
  'https://btc2g.com',
  'https://www.btc2g.com',
  'https://vitalydz.github.io',
]);

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

function readEmail(value: unknown) {
  if (typeof value !== 'string') {
    return '';
  }

  return value.trim();
}

function isValidEmail(email: string) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function getRequiredEnv(name: string) {
  const value = process.env[name];

  if (!value) {
    throw new Error(`Missing ${name}`);
  }

  return value;
}

export default async function handler(req: any, res: any) {
  setCorsHeaders(req, res);

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  if (req.method !== 'POST') {
    return sendJson(res, 405, { error: 'Method not allowed. Use POST.' });
  }

  try {
    const secretKey = getRequiredEnv('STRIPE_SECRET_KEY');
    const priceId = getRequiredEnv('STRIPE_PRICE_ID');
    const siteUrl = getRequiredEnv('SITE_URL').replace(/\/$/, '');
    const body = parseBody(req);
    const email = readEmail(body.email);

    if (email && !isValidEmail(email)) {
      return sendJson(res, 400, { error: 'A valid email address is required.' });
    }

    const stripe = new Stripe(secretKey);
    const session = await stripe.checkout.sessions.create({
      mode: 'subscription',
      line_items: [
        {
          price: priceId,
          quantity: 1,
        },
      ],
      success_url: `${siteUrl}/premium-success.html?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${siteUrl}/buy.html?premium=cancelled`,
      customer_email: email || undefined,
      allow_promotion_codes: true,
    });

    if (!session.url) {
      return sendJson(res, 500, { error: 'Unable to create Stripe Checkout session.' });
    }

    return sendJson(res, 200, { url: session.url });
  } catch {
    return sendJson(res, 500, { error: 'Stripe checkout is unavailable right now.' });
  }
}
