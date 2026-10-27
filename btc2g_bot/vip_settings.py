"""Environment configuration shared by the bot and the web payment service."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
DATABASE = Path(os.getenv('VIP_DATABASE', str(ROOT / 'private' / 'vip.sqlite3')))
FORECAST = Path(os.getenv('VIP_FORECAST_FILE', str(ROOT / 'private' / 'weekly_forecast.json')))
SALES_ENABLED = os.getenv('VIP_SALES_ENABLED', 'false').lower() == 'true'
STARS_PRICE = int(os.getenv('VIP_STARS_PRICE', '200'))
SUPPORT = os.getenv('VIP_SUPPORT_CONTACT', '').strip()
STRIPE_SECRET = os.getenv('STRIPE_SECRET_KEY', '')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET', '')
PUBLIC_URL = os.getenv('VIP_PUBLIC_URL', '').rstrip('/')
