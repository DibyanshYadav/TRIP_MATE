# tools/currency_tool.py
import os
import threading
import time

import requests
from dotenv import load_dotenv

load_dotenv()

FREECURRENCY_URL = "https://api.freecurrencyapi.com/v1/latest"
FALLBACK_URL = "https://open.er-api.com/v6/latest/{base}"

CACHE_TTL_SECONDS = 6 * 60 * 60     # refresh rates every 6 hours

_cache: dict = {}                   # base -> (fetched_at, rates)
_lock = threading.Lock()


def _fetch_freecurrency(base: str, api_key: str) -> dict:
    r = requests.get(
        FREECURRENCY_URL,
        params={"apikey": api_key, "base_currency": base},
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("data", {})


def _fetch_fallback(base: str) -> dict:
    """No key needed, supports many more currencies."""
    r = requests.get(FALLBACK_URL.format(base=base), timeout=10)
    r.raise_for_status()
    j = r.json()
    if j.get("result") != "success":
        raise ValueError("rate API returned an error")
    return j["rates"]


def _store(base: str, rates: dict):
    with _lock:
        _cache[base] = (time.time(), rates)


def _rates(base: str) -> dict:
    """All rates for a base currency. Cached for a few hours, so a server that
    runs for days still picks up fresh rates."""
    with _lock:
        hit = _cache.get(base)
        if hit and time.time() - hit[0] < CACHE_TTL_SECONDS:
            return hit[1]

    rates = {}
    api_key = os.getenv("CURRENCY_API_KEY")

    # 1) freecurrencyapi (your key)
    if api_key:
        try:
            rates = _fetch_freecurrency(base, api_key)
        except Exception:
            rates = {}   # fall through to the backup API

    # 2) backup
    if not rates:
        rates = _fetch_fallback(base)

    _store(base, rates)
    return rates


def convert_currency(amount: float, frm: str, to: str) -> str:
    frm, to = frm.upper().strip(), to.upper().strip()
    rates = _rates(frm)

    if to not in rates:
        # free tier of freecurrencyapi has a limited currency list, so try the backup
        fallback = _fetch_fallback(frm)
        if to not in fallback:
            return f"Currency {to} not supported."
        rates = fallback
        _store(frm, rates)   # the backup list is bigger, so keep it for next time

    rate = rates[to]
    return f"{amount:,.2f} {frm} ≈ {amount * rate:,.2f} {to} (rate {rate:.4f}, approximate)"