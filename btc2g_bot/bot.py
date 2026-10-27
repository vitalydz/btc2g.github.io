#!/usr/bin/env python3
"""BTC2G VIP delivery through a shared payment ledger."""
import logging
import os
import time
from datetime import datetime, timezone
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, PreCheckoutQueryHandler, MessageHandler, filters
import vip_settings as settings
from vip_product import PRICE_USD_CENTS
from vip_store import Store
from vip_service import current_forecast, require_sales

logging.basicConfig(level=logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


def ledger(context):
    return context.application.bot_data['ledger']


def private(update):
    return update.effective_chat is not None and update.effective_chat.type == 'private'


def terms_text():
    return ('BTC2G VIP: 30 дней доступа к прогнозам BTC/Gold на ближайшие 7 дней. '
            f'Цена в Telegram: {settings.STARS_PRICE} Stars. Автосписаний нет. '
            'Прогноз не гарантирует прибыль. Материалы: /forecast; срок: /status. '
            'Поддержка и вопросы возврата: /paysupport.')


async def start(update, context):
    if not private(update):
        return
    text = terms_text()
    buttons = []
    try:
        require_sales(settings, 'stars')
        buttons = [[InlineKeyboardButton(f'Принимаю условия — {settings.STARS_PRICE} Stars', callback_data='buy_stars')]]
    except (ValueError, OSError, KeyError):
        text += '\n\nПродажи временно недоступны. Оплату отправлять не нужно.'
    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(buttons) if buttons else None)


async def buy_stars(update, context):
    query = update.callback_query
    await query.answer()
    if not private(update):
        return
    try:
        require_sales(settings, 'stars')
    except (ValueError, OSError, KeyError):
        await query.message.reply_text('Оплата временно недоступна. Попробуй позже.')
        return
    order, _ = ledger(context).create(update.effective_user.id, 'stars', settings.STARS_PRICE, 'XTR')
    await context.bot.send_invoice(chat_id=update.effective_chat.id, title='BTC2G VIP — 30 дней',
        description='Прогнозы BTC/Gold на 7 дней вперёд. Без автосписаний.', payload=order,
        provider_token='', currency='XTR', prices=[LabeledPrice('VIP на 30 дней', settings.STARS_PRICE)], start_parameter='vip')


async def precheckout(update, context):
    query = update.pre_checkout_query
    try:
        require_sales(settings, 'stars')
        order = ledger(context).check(query.invoice_payload, 'stars', query.from_user.id, query.total_amount, query.currency)
        if order['paid'] is not None or order['created'] + 86400 <= time.time():
            raise ValueError('Invoice expired or paid')
    except (ValueError, OSError, KeyError):
        await query.answer(ok=False, error_message='Счёт недоступен. Открой /vip для нового счёта.')
        return
    await query.answer(ok=True)


async def paid(update, context):
    payment = update.message.successful_payment
    try:
        ledger(context).grant(payment.invoice_payload, 'stars', update.effective_user.id,
            payment.total_amount, payment.currency, payment.telegram_payment_charge_id)
    except ValueError:
        logging.error('Payment validation failed; reconciliation required')
        await update.message.reply_text('Платёж требует проверки. Обратись в /paysupport.')
        return
    await update.message.reply_text('Платёж обработан. /forecast — прогноз, /status — срок доступа.')
    await status(update, context)


async def refunded(update, context):
    ledger(context).revoke('stars', update.message.refunded_payment.telegram_payment_charge_id)


async def status(update, context):
    if not private(update):
        return
    expiry = ledger(context).expires(update.effective_user.id)
    text = ('VIP активен до ' + datetime.fromtimestamp(expiry, timezone.utc).strftime('%d.%m.%Y %H:%M UTC')
            if expiry > time.time() else 'Активного VIP-доступа нет. /vip')
    await update.message.reply_text(text)


async def forecast(update, context):
    if not private(update):
        return
    if ledger(context).expires(update.effective_user.id) <= time.time():
        await update.message.reply_text('Нужен действующий VIP-доступ. /vip')
        return
    try:
        data = current_forecast(settings.FORECAST)
    except (ValueError, OSError, KeyError):
        await update.message.reply_text('Новый прогноз пока не готов. Обратись в /support.')
        return
    content = f"Прогноз от {data['issued_on']}, до {data['valid_until']} (UTC)\n\n{data['text']}\n\nПрогноз не гарантирует прибыль."
    for offset in range(0, len(content), 3500):
        await update.message.reply_text(content[offset:offset+3500], protect_content=True)


async def connect(update, context):
    # Codes bind website purchases to the authenticated Telegram user.
    if not private(update):
        return
    try:
        require_sales(settings, 'stripe')
    except (ValueError, OSError, KeyError):
        await update.message.reply_text('Привязка к сайту пока недоступна.')
        return
    _, token = ledger(context).create(update.effective_user.id, 'stripe', PRICE_USD_CENTS, 'USD', with_token=True)
    await update.message.reply_text('Код привязки аккаунта к сайту (24 часа). Не передавай другим:\n' + token)


async def terms(update, context):
    if private(update):
        await update.message.reply_text(terms_text())


async def support(update, context):
    if private(update):
        await update.message.reply_text('Поддержка BTC2G: ' + (settings.SUPPORT or 'Контакт ещё не настроен; продажи закрыты.'))


def build_application(token):
    app = Application.builder().token(token).build()
    app.bot_data['ledger'] = Store(settings.DATABASE)
    for name, handler in {'start': start, 'vip': start, 'status': status, 'forecast': forecast,
                         'connect': connect, 'terms': terms, 'support': support, 'paysupport': support}.items():
        app.add_handler(CommandHandler(name, handler))
    app.add_handler(CallbackQueryHandler(buy_stars, pattern='^buy_stars$'))
    app.add_handler(PreCheckoutQueryHandler(precheckout))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, paid))
    app.add_handler(MessageHandler(filters.StatusUpdate.REFUNDED_PAYMENT, refunded))
    return app


def main():
    token = os.getenv('BOT_TOKEN')
    if not token or 'ВСТАВЬ' in token:
        raise SystemExit('Configure BOT_TOKEN in the environment')
    build_application(token).run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
