"""Standalone USD checkout; deploy with persistent SQLite storage."""
from pathlib import Path
import stripe
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, StrictBool
import vip_settings as settings
from vip_product import PRICE_USD_CENTS
from vip_store import Store
from vip_service import require_sales, process_stripe_event

app = FastAPI(docs_url=None, redoc_url=None)


class Checkout(BaseModel):
    code: str = Field(min_length=30, max_length=100)
    accepted_terms: StrictBool


@app.get('/vip')
def vip_page():
    return FileResponse(Path(__file__).parent / 'web' / 'vip.html')


@app.get('/offer')
def offer():
    try:
        require_sales(settings, 'stripe')
        available = True
    except (ValueError, OSError, KeyError):
        available = False
    return {'available': available, 'usd_cents': PRICE_USD_CENTS, 'stars': settings.STARS_PRICE}


@app.post('/checkout')
def checkout(body: Checkout):
    if not body.accepted_terms:
        raise HTTPException(400, 'Accept the terms before paying')
    try:
        require_sales(settings, 'stripe')
    except (ValueError, OSError, KeyError):
        raise HTTPException(503, 'Sales are temporarily unavailable') from None
    try:
        order = Store(settings.DATABASE).from_token(body.code)
    except ValueError:
        raise HTTPException(400, 'Invalid, expired or used account code') from None
    try:
        session = stripe.checkout.Session.create(
            api_key=settings.STRIPE_SECRET, idempotency_key='vip-' + order['id'],
            mode='payment', payment_method_types=['card'],
            line_items=[{'price_data': {'currency': 'usd', 'unit_amount': PRICE_USD_CENTS,
                         'product_data': {'name': 'BTC2G VIP — 30 days'}}, 'quantity': 1}],
            metadata={'vip_order_id': order['id']},
            success_url=settings.PUBLIC_URL + '/vip?result=success',
            cancel_url=settings.PUBLIC_URL + '/vip?result=cancelled',
        )
        return {'url': session.url}
    except stripe.StripeError:
        raise HTTPException(502, 'Checkout is temporarily unavailable; retry with the same code') from None


@app.post('/stripe/webhook')
async def webhook(request: Request):
    if not settings.STRIPE_WEBHOOK_SECRET:
        raise HTTPException(503, 'Webhook is not configured')
    raw = await request.body()
    if len(raw) > 262144:
        raise HTTPException(413, 'Request too large')
    try:
        event = stripe.Webhook.construct_event(raw, request.headers.get('stripe-signature', ''), settings.STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.SignatureVerificationError):
        raise HTTPException(400, 'Invalid webhook signature') from None
    store = Store(settings.DATABASE)
    try:
        if event['type'] == 'charge.refunded':
            charge = event['data']['object']
            if charge.get('refunded'):
                store.revoke('stripe', charge['payment_intent'])
        else:
            process_stripe_event(store, event)
    except (ValueError, KeyError):
        raise HTTPException(400, 'Payment reconciliation required') from None
    return {'received': True}
