"""Shared VIP product definition; billing duration is configured separately."""

from decimal import Decimal

PRICE_USD = Decimal('4.95')
PRICE_USD_CENTS = 495
FORECAST_HORIZON_DAYS = 7
ACCESS_DAYS = 30
PRODUCT_CODE = 'btc2g-vip'
PRODUCT_NAME = 'BTC2G VIP'


def offer_text():
    return (
        f'{PRODUCT_NAME}: ${PRICE_USD:.2f} на сайте за {ACCESS_DAYS} дней доступа. '
        f'Прогноз BTC/Gold на {FORECAST_HORIZON_DAYS} дней. '
        'Прогноз не гарантирует прибыль. '
        'В Telegram оплата в Stars по отдельной цене. Автосписаний нет.'
    )


def validate_forecast(forecast, today):
    """Reject missing or expired weekly content before enabling a paid offer."""
    from datetime import date
    if not isinstance(forecast, dict):
        raise ValueError('Weekly forecast must be an object')
    if type(forecast.get('horizon_days')) is not int or forecast['horizon_days'] != FORECAST_HORIZON_DAYS:
        raise ValueError('A seven-day forecast is required')
    if not all(isinstance(forecast.get(key), str) for key in ('issued_on', 'valid_until')):
        raise ValueError('Forecast dates are required')
    issued = date.fromisoformat(forecast['issued_on'])
    valid_until = date.fromisoformat(forecast['valid_until'])
    if (valid_until - issued).days != FORECAST_HORIZON_DAYS:
        raise ValueError('Forecast validity must span seven days')
    if not issued <= today < valid_until:
        raise ValueError('Weekly forecast is not current')
    if not isinstance(forecast.get('text'), str) or not forecast['text'].strip():
        raise ValueError('Weekly forecast content is required')
    return forecast
