import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Optional
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")
print(f"🔑 ENTSO-E key loaded: {ENTSOE_API_KEY[:8] if ENTSOE_API_KEY else 'MISSING'}...")
ENTSOE_BASE = "https://web-api.tp.entsoe.eu/api"

AREA_CODES = {
    "DE": "10Y1001A1001A82H",  # Germany (correct bidding zone)
    "NL": "10YNL----------L",  # Netherlands (APX day-ahead)
    "FR": "10YFR-RTE------C",  # France RTE
    "ES": "10YES-REE------0",  # Spain REE
}

_cache: dict = {}
CACHE_TTL = 3600

def fetch_day_ahead_prices(market: str = "DE") -> Optional[list]:
    area = AREA_CODES.get(market)
    if not area or not ENTSOE_API_KEY:
        return None

    cached = _cache.get(market)
    if cached and (time.time() - cached["fetched_at"]) < CACHE_TTL:
        print(f"✅ ENTSO-E cache hit for {market}")
        return cached["prices"]

    try:
        now = datetime.utcnow()
        # Go back 3 days to ensure we always have published data
        period_start = (now - timedelta(days=3)).strftime("%Y%m%d0000")
        period_end   = (now + timedelta(days=1)).strftime("%Y%m%d2300")

        params = {
            "securityToken": ENTSOE_API_KEY,
            "documentType":  "A44",
            "in_Domain":     area,
            "out_Domain":    area,
            "periodStart":   period_start,
            "periodEnd":     period_end,
        }

        print(f"🌐 Fetching ENTSO-E prices for {market} ({period_start} → {period_end})...")
        response = requests.get(ENTSOE_BASE, params=params, timeout=15)

        if response.status_code != 200:
            print(f"❌ ENTSO-E HTTP {response.status_code}")
            return None

        root = ET.fromstring(response.content)

        # Auto-detect namespace
        tag = root.tag
        ns_uri = ""
        if tag.startswith("{"):
            ns_uri = tag[1:tag.index("}")]
        ns = {"ns": ns_uri} if ns_uri else {}

        def find(el, path):
            if ns_uri:
                path = "/".join(f"ns:{p}" for p in path.split("/"))
                return el.find(path, ns)
            return el.find(path)

        def findall(el, path):
            if ns_uri:
                path = "/".join(f"ns:{p}" for p in path.split("/"))
                return el.findall(path, ns)
            return el.findall(path)

        # Check for acknowledgement (error) document
        if "Acknowledgement" in tag:
            reason = find(root, "Reason/text")
            print(f"❌ ENTSO-E returned Acknowledgement: {reason.text if reason is not None else 'unknown reason'}")
            return None

        prices = []
        for ts in findall(root, "TimeSeries"):
            period = find(ts, "Period")
            if period is None:
                continue
            start_el = find(period, "timeInterval/start")
            if start_el is None:
                continue
            try:
                start_dt = datetime.strptime(start_el.text.strip(), "%Y-%m-%dT%H:%MZ")
            except:
                continue
            for pt in findall(period, "Point"):
                pos_el   = find(pt, "position")
                price_el = find(pt, "price.amount")
                if pos_el is None or price_el is None:
                    continue
                pos   = int(pos_el.text)
                price = float(price_el.text)
                ts_dt = start_dt + timedelta(hours=pos - 1)
                prices.append({
                    "timestamp":      ts_dt.strftime("%Y-%m-%d %H:%M"),
                    "price_eur_mwh":  round(price, 2),
                    "source":         "ENTSO-E"
                })

        if not prices:
            print(f"⚠️ No price points parsed for {market} — XML may use different structure")
            print(f"   Root tag: {root.tag}")
            print(f"   Children: {[c.tag for c in list(root)[:5]]}")
            return None

        prices.sort(key=lambda x: x["timestamp"])
        prices = prices[-48:]

        _cache[market] = {"prices": prices, "fetched_at": time.time()}
        print(f"✅ ENTSO-E: {len(prices)} real prices for {market}. Latest: €{prices[-1]['price_eur_mwh']}/MWh at {prices[-1]['timestamp']}")
        return prices

    except Exception as e:
        print(f"❌ ENTSO-E exception for {market}: {e}")
        return None


def get_latest_price(market: str = "DE") -> Optional[float]:
    prices = fetch_day_ahead_prices(market)
    if prices:
        return prices[-1]["price_eur_mwh"]
    return None


def get_price_summary(market: str = "DE") -> dict:
    prices = fetch_day_ahead_prices(market)
    if not prices:
        return {}
    vals    = [p["price_eur_mwh"] for p in prices]
    last24  = vals[-24:] if len(vals) >= 24 else vals
    current = vals[-1]
    prev24  = vals[-25] if len(vals) >= 25 else vals[0]
    mean_24h   = round(float(np.mean(last24)), 2)
    volatility = round(float(np.std(last24)), 2)
    pct_change = round((current - prev24) / prev24 * 100, 2) if prev24 else 0
    trend      = "bullish" if current > mean_24h else "bearish"
    return {
        "current_price":    current,
        "mean_24h":         mean_24h,
        "volatility_24h":   volatility,
        "pct_change_24h":   pct_change,
        "trend":            trend,
        "anomalies_detected": 0,
        "source":           "ENTSO-E"
    }