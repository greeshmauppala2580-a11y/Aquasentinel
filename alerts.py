import os
import requests


def telegram_ready():
    return bool(os.getenv("TELEGRAM_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))


def send_telegram(text):
    if not telegram_ready():
        return False
    try:
        url = f"https://api.telegram.org/bot{os.environ['TELEGRAM_TOKEN']}/sendMessage"
        requests.post(url, data={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text}, timeout=5)
        return True
    except Exception:
        return False
