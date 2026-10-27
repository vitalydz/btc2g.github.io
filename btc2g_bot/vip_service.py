"""Product availability and verified payment handling without network side effects."""
import json
from datetime import datetime, timezone
from vip_product import validate_forecast, PRICE_USD_CENTS


def current_forecast(path):
    return validate_forecast(json.loads(path.read_text(encoding='utf-8')), datetime.now(timezone.utc).date())


def require_sales(settings, provider):
    if not settings.SALES_ENABLED or not settings.SUPPORT:
        raise ValueError('Sales are not configured')
    current_forecast(settings.FORECAST)
    if provider == 'stars' and settings.STARS_PRICE <= 0:
        raise ValueError('Stars price is not configured')
    if provider == 'stripe' and (not settings.STRIPE_SECRET or not settings.STRIPE_WEBHOOK_SECRET
                                 or not settings.PUBLIC_URL.startswith('https://')):
        raise ValueError('Website payments are not configured')


def process_stripe_event(store, event):
    if event['type'] not in {'checkout.session.completed', 'checkout.session.async_payment_succeeded'}:
        return False
    session = event['data']['object']
    if session.get('payment_status') != 'paid':
        return False
    if session.get('mode') != 'payment' or session.get('amount_total') != PRICE_USD_CENTS or session.get('currency') != 'usd':
        raise ValueError('Unexpected paid product')
    order_id = session.get('metadata', {}).get('vip_order_id', '')
    order = store.order(order_id)
    if not order:
        raise ValueError('Unknown order')
    return store.grant(order_id, 'stripe', order['user_id'], PRICE_USD_CENTS, 'USD', session['payment_intent'])
