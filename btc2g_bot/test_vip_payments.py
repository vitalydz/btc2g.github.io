import asyncio
import hashlib
import hmac
import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import bot
import web_app
import vip_settings as settings
from vip_store import Store
from vip_service import process_stripe_event


class PaymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.db = root / 'vip.sqlite3'
        self.store = Store(self.db)
        forecast = root / 'forecast.json'
        today = datetime.now(timezone.utc).date()
        forecast.write_text(json.dumps(dict(issued_on=str(today), valid_until=str(today+timedelta(days=7)),
                                            horizon_days=7, text='Test fixture only.')), encoding='utf-8')
        for key, value in dict(DATABASE=self.db, FORECAST=forecast, SALES_ENABLED=True,
                               STARS_PRICE=200, SUPPORT='test support', STRIPE_SECRET='sk_test_fixture',
                               STRIPE_WEBHOOK_SECRET='whsec_fixture', PUBLIC_URL='https://example.com').items():
            p = patch.object(settings, key, value)
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(web_app.app)

    def make_event(self, order, **values):
        obj = dict(mode='payment', payment_status='paid', amount_total=495, currency='usd',
                   payment_intent='pi_fixture', metadata={'vip_order_id': order})
        obj.update(values)
        return dict(type='checkout.session.completed', data={'object': obj})

    def signed(self, event):
        raw = json.dumps(event).encode()
        timestamp = int(time.time())
        signature = hmac.new(b'whsec_fixture', str(timestamp).encode()+b'.'+raw, hashlib.sha256).hexdigest()
        return self.client.post('/stripe/webhook', content=raw,
                                headers={'stripe-signature':f't={timestamp},v1={signature}'})

    def test_both_providers_extend_same_account(self):
        for provider, amount, currency in [('stripe',495,'USD'),('stars',200,'XTR')]:
            order, _ = self.store.create(7, provider, amount, currency)
            self.store.grant(order, provider, 7, amount, currency, provider+'-charge', now=1000)
        self.assertEqual(self.store.expires(7), 1000+60*86400)

    def test_duplicate_charge_survives_restart_and_concurrency(self):
        order, _ = self.store.create(7,'stars',200,'XTR')
        def deliver(_):
            return Store(self.db).grant(order,'stars',7,200,'XTR','charge',now=1000)
        with ThreadPoolExecutor(max_workers=4) as executor:
            self.assertEqual(sum(executor.map(deliver, range(8))), 1)
        self.assertEqual(self.store.expires(7),1000+30*86400)

    def test_mismatched_payment_rejected(self):
        order, _ = self.store.create(7,'stars',200,'XTR')
        for user, amount, currency in [(8,200,'XTR'),(7,199,'XTR'),(7,200,'USD')]:
            with self.assertRaises(ValueError): self.store.grant(order,'stars',user,amount,currency,'charge')
        self.assertEqual(self.store.expires(7),0)

    def test_refund_before_and_after_payment(self):
        order, _ = self.store.create(7,'stars',200,'XTR')
        self.store.revoke('stars','charge',now=1000)
        self.assertFalse(self.store.grant(order,'stars',7,200,'XTR','charge',now=1000))
        self.assertEqual(self.store.expires(7),0)
        self.store.grant(order,'stars',7,200,'XTR','another-charge',now=1000)
        self.assertTrue(self.store.revoke('stars','another-charge',now=1100))
        self.assertFalse(self.store.revoke('stars','another-charge',now=1100))
        self.assertEqual(self.store.expires(7),1100)

    def test_token_expiry_and_consumption(self):
        order, token = self.store.create(7,'stripe',495,'USD',now=1000,with_token=True)
        self.assertEqual(self.store.from_token(token,now=1001)['user_id'],7)
        with self.assertRaises(ValueError): self.store.from_token(token,now=87400)
        self.store.grant(order,'stripe',7,495,'USD','pi_token',now=2000)
        with self.assertRaises(ValueError): self.store.from_token(token,now=2001)

    def test_checkout_retries_use_same_key_and_fixed_price(self):
        _, token = self.store.create(7,'stripe',495,'USD',with_token=True)
        with patch.object(web_app.stripe.checkout.Session,'create', return_value=SimpleNamespace(url='https://checkout.stripe.com/fixture')) as create:
            for _ in range(2):
                self.assertEqual(self.client.post('/checkout', json={'code':token,'accepted_terms':True}).status_code,200)
            self.assertEqual(create.call_args_list[0].kwargs['idempotency_key'],create.call_args_list[1].kwargs['idempotency_key'])
            self.assertEqual(create.call_args.kwargs['line_items'][0]['price_data']['unit_amount'],495)
            self.assertEqual(create.call_args.kwargs['mode'],'payment')
        self.assertEqual(self.store.expires(7),0)

    def test_consent_and_sales_gate(self):
        _, token = self.store.create(7,'stripe',495,'USD',with_token=True)
        self.assertEqual(self.client.post('/checkout',json={'code':token,'accepted_terms':False}).status_code,400)
        with patch.object(settings,'SALES_ENABLED',False):
            self.assertEqual(self.client.post('/checkout',json={'code':token,'accepted_terms':True}).status_code,503)
        settings.FORECAST.unlink()
        self.assertEqual(self.client.post('/checkout',json={'code':token,'accepted_terms':True}).status_code,503)

    def test_verified_webhook_only_and_replay(self):
        order, _ = self.store.create(7,'stripe',495,'USD')
        event = self.make_event(order)
        self.assertEqual(self.client.post('/stripe/webhook',json=event).status_code,400)
        self.assertEqual(self.store.expires(7),0)
        self.assertEqual(self.signed(event).status_code,200)
        expiry=self.store.expires(7)
        self.assertEqual(self.signed(event).status_code,200)
        self.assertEqual(self.store.expires(7),expiry)

    def test_unpaid_wrong_amount_and_return_page_do_not_grant(self):
        order, _ = self.store.create(7,'stripe',495,'USD')
        self.assertFalse(process_stripe_event(self.store,self.make_event(order,payment_status='unpaid')))
        self.assertEqual(self.signed(self.make_event(order,amount_total=1)).status_code,400)
        self.assertEqual(self.client.get('/vip?result=success').status_code,200)
        self.assertEqual(self.store.expires(7),0)

    def test_stars_precheckout_and_payment(self):
        order, _ = self.store.create(7,'stars',200,'XTR')
        query = SimpleNamespace(invoice_payload=order,from_user=SimpleNamespace(id=7),total_amount=200,currency='XTR',answer=AsyncMock())
        context = SimpleNamespace(application=SimpleNamespace(bot_data={'ledger':self.store}))
        asyncio.run(bot.precheckout(SimpleNamespace(pre_checkout_query=query),context))
        query.answer.assert_awaited_once_with(ok=True)
        self.assertEqual(self.store.expires(7),0)
        payment=SimpleNamespace(invoice_payload=order,total_amount=200,currency='XTR',telegram_payment_charge_id='tg_charge')
        message=SimpleNamespace(successful_payment=payment,reply_text=AsyncMock())
        update=SimpleNamespace(message=message,effective_user=SimpleNamespace(id=7),effective_chat=SimpleNamespace(type='private'))
        asyncio.run(bot.paid(update,context))
        expiry=self.store.expires(7)
        asyncio.run(bot.paid(update,context))
        self.assertEqual(self.store.expires(7),expiry)

    def test_forecast_expiry_and_private_chat(self):
        reply=AsyncMock()
        update=SimpleNamespace(message=SimpleNamespace(reply_text=reply),effective_user=SimpleNamespace(id=7),effective_chat=SimpleNamespace(type='private'))
        context=SimpleNamespace(application=SimpleNamespace(bot_data={'ledger':self.store}))
        asyncio.run(bot.forecast(update,context))
        self.assertIn('Нужен',reply.call_args.args[0])
        order,_=self.store.create(7,'stars',200,'XTR')
        self.store.grant(order,'stars',7,200,'XTR','forecast_charge')
        asyncio.run(bot.forecast(update,context))
        self.assertIn('Test fixture',reply.call_args.args[0])
        reply.reset_mock()
        update.effective_chat.type='group'
        asyncio.run(bot.forecast(update,context))
        reply.assert_not_awaited()

    def test_offer_disabled_without_forecast(self):
        self.assertTrue(self.client.get('/offer').json()['available'])
        settings.FORECAST.unlink()
        self.assertFalse(self.client.get('/offer').json()['available'])

    def test_paid_callback_still_processed_while_sales_disabled(self):
        order,_=self.store.create(7,'stripe',495,'USD')
        with patch.object(settings,'SALES_ENABLED',False):
            self.assertEqual(self.signed(self.make_event(order)).status_code,200)
        self.assertGreater(self.store.expires(7),time.time())

    def test_bot_handlers_build_without_contacting_telegram(self):
        app=bot.build_application('123456789:TEST_ONLY_NOT_A_REAL_TOKEN')
        self.assertIsInstance(app.bot_data['ledger'], Store)
        self.assertGreaterEqual(len(app.handlers[0]),12)


if __name__ == '__main__':
    unittest.main()
