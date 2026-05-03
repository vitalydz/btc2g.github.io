import Stripe from 'stripe';

export const config = {
  api: {
    bodyParser: false,
  },
};

const HANDLED_EVENTS = new Set([
  'checkout.session.completed',
  'customer.subscription.created',
  'customer.subscription.updated',
  'customer.subscription.deleted',
  'invoice.payment_succeeded',
  'invoice.payment_failed',
]);

function sendJson(res: any, status: number, body: Record<string, unknown>) {
  res.setHeader('Content-Type', 'application/json');
  return res.status(status).json(body);
}

function readRawBody(req: any): Promise<Buffer> {
  if (Buffer.isBuffer(req.body)) {
    return Promise.resolve(req.body);
  }

  if (typeof req.body === 'string') {
    return Promise.resolve(Buffer.from(req.body));
  }

  if (req.body && typeof req.body === 'object') {
    // Vercel should preserve the raw stream because bodyParser is disabled.
    // This fallback avoids hanging if a local runtime pre-parses JSON anyway.
    return Promise.resolve(Buffer.from(JSON.stringify(req.body)));
  }

  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];

    req.on('data', (chunk: Buffer | string) => {
      chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    });

    req.on('end', () => {
      resolve(Buffer.concat(chunks));
    });

    req.on('error', reject);
  });
}

export default async function handler(req: any, res: any) {
  if (req.method !== 'POST') {
    return sendJson(res, 405, { error: 'Method not allowed. Use POST.' });
  }

  const secretKey = process.env.STRIPE_SECRET_KEY;
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
  const signature = req.headers['stripe-signature'];

  if (!secretKey || !webhookSecret) {
    return sendJson(res, 500, { error: 'Stripe webhook configuration is incomplete.' });
  }

  if (typeof signature !== 'string') {
    return sendJson(res, 400, { error: 'Missing Stripe signature.' });
  }

  try {
    const stripe = new Stripe(secretKey);
    const rawBody = await readRawBody(req);
    const event = stripe.webhooks.constructEvent(rawBody, signature, webhookSecret);

    if (HANDLED_EVENTS.has(event.type)) {
      console.log(`Stripe webhook received: ${event.type}`);
      // TODO: Grant, update, or revoke BTC2G Premium access when user storage exists.
      // TODO: Store subscription status by Stripe customer or checkout session email.
    } else {
      console.log(`Stripe webhook ignored: ${event.type}`);
    }

    return sendJson(res, 200, { received: true });
  } catch {
    return sendJson(res, 400, { error: 'Invalid Stripe webhook signature.' });
  }
}
