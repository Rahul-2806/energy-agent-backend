import os, json, time, threading, requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq
from entsoe import fetch_day_ahead_prices, get_price_summary, get_latest_price

load_dotenv()
NEWSAPI_KEY  = os.getenv("NEWSAPI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
BREVO_API_KEY = os.getenv("BREVO_API_KEY")
GMAIL_USER        = os.getenv("GMAIL_USER")
GMAIL_PASSWORD    = os.getenv("GMAIL_APP_PASSWORD")
TWILIO_SID        = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN      = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WA_FROM    = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")

groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="EnergyAgent API", version="2.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HISTORY_FILE     = "history.json"
SUBSCRIBERS_FILE = "subscribers.json"
ALERTS_FILE      = "alert_config.json"

def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path) as f: return json.load(f)
    except: pass
    return default

def save_json(path, data):
    with open(path, "w") as f: json.dump(data, f, indent=2)

analysis_history     = load_json(HISTORY_FILE, [])
subscribers          = load_json(SUBSCRIBERS_FILE, [])
WA_SUBSCRIBERS_FILE  = "wa_subscribers.json"
wa_subscribers       = load_json(WA_SUBSCRIBERS_FILE, [])
alert_config         = load_json(ALERTS_FILE, {"price_above": 150, "price_below": 30, "volatility_above": 25})
last_news_fetch      = {"time": 0, "data": None}
last_analysis_result = {"data": None}
last_email_sent = {"time": 0}  # Rate limit: max 1 alert email per 2 hours
EMAIL_COOLDOWN  = 7200  # 2 hours in seconds

MARKET_META = {
    "DE": {"label": "Germany",        "base": 86},
    "NL": {"label": "Netherlands",    "base": 90},
    "FR": {"label": "France",         "base": 78},
    "ES": {"label": "Spain",          "base": 70},
}

# ── BREVO EMAIL ───────────────────────────────────────────────────────────────
def send_brevo_email(to_email: str, subject: str, html_content: str):
    """Send email via Brevo REST API — works on Railway (HTTPS, no SMTP)."""
    try:
        r = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json"},
            json={
                "sender":     {"name": os.getenv("BREVO_SENDER_NAME","EnergyAgent"), "email": os.getenv("BREVO_SENDER_EMAIL","rahulsr2806@gmail.com")},
                "to":         [{"email": to_email}],
                "subject":    subject,
                "htmlContent": html_content,
            },
            timeout=15
        )
        if r.status_code in [200, 201]:
            print(f"✅ Brevo email sent to {to_email}")
            return True
        else:
            print(f"❌ Brevo error {r.status_code}: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Brevo exception: {e}")
        return False

def send_whatsapp(to_number: str, message: str):
    """Send WhatsApp message via Twilio."""
    try:
        r = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
            auth=(TWILIO_SID, TWILIO_TOKEN),
            data={
                "From": TWILIO_WA_FROM,
                "To":   f"whatsapp:{to_number}",
                "Body": message,
            },
            timeout=10
        )
        if r.status_code == 201:
            print(f"✅ WhatsApp sent to {to_number}")
            return True
        else:
            print(f"❌ WhatsApp error {r.status_code}: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ WhatsApp exception: {e}")
        return False

def send_whatsapp_alert(alert_data, price_data, reasoning):
    """Send WhatsApp alert to all WA subscribers."""
    if not wa_subscribers:
        return
    s   = price_data.get("summary", {})
    rec = reasoning.get("recommendation", "HOLD")
    con = reasoning.get("confidence", "LOW")
    src = price_data.get("data_source", "Unknown")
    mkt = price_data.get("market_label", "Europe")
    rec_emoji = "🟢" if rec == "BUY" else "🔴" if rec == "SELL" else "🟡"
    high_alerts = [a for a in alert_data if a["severity"] == "HIGH"]
    alert_text = "\n".join([f"⚠️ {a['message']}" for a in high_alerts[:3]])
    msg = f"""⚡ *EnergyAgent Alert*
━━━━━━━━━━━━━━━
{rec_emoji} *{rec}* | Confidence: {con}
📍 Market: {mkt}
💶 Price: €{s.get('current_price')}/MWh
📈 24h Change: {s.get('pct_change_24h',0):+.1f}%
📊 Source: {src}
━━━━━━━━━━━━━━━
{alert_text}
━━━━━━━━━━━━━━━
_{reasoning.get('reasoning_summary','')[:200]}_

View dashboard: https://energy-agent-gilt.vercel.app"""
    for number in wa_subscribers:
        send_whatsapp(number, msg)

def build_alert_email_html(alert_data, price_data, reasoning, market_label):
    s   = price_data.get("summary", {})
    rec = reasoning.get("recommendation", "HOLD")
    con = reasoning.get("confidence", "LOW")
    src = price_data.get("data_source", "Unknown")
    rec_color = "#5ac8fa" if rec == "BUY" else "#ff6b6b" if rec == "SELL" else "#ffd60a"
    alerts_html = "".join([
        f'<div style="background:rgba(255,255,255,0.04);border-left:3px solid {"#ff6b6b" if a["severity"]=="HIGH" else "#ffd60a"};padding:10px 14px;margin:6px 0;border-radius:6px;">'
        f'<strong style="color:{"#ff6b6b" if a["severity"]=="HIGH" else "#ffd60a"};font-size:11px;letter-spacing:1px;">{a["type"]}</strong><br>'
        f'<span style="color:#c0d4e8;font-size:13px">{a["message"]}</span><br>'
        f'<span style="color:#5a7a9a;font-size:11px">→ {a["action"]}</span></div>'
        for a in alert_data if a.get("severity") in ["HIGH", "MEDIUM"]
    ])
    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#060c14;font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display',sans-serif;color:#e8f4ff;">
<div style="max-width:580px;margin:0 auto;padding:32px 24px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid rgba(255,255,255,0.07);">
    <div style="width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#5ac8fa,#0a84ff);display:flex;align-items:center;justify-content:center;font-size:16px;">⚡</div>
    <div>
      <div style="font-size:16px;font-weight:700;letter-spacing:-0.3px;color:#fff;">EnergyAgent</div>
      <div style="font-size:10px;color:#4a6a8a;letter-spacing:1px;">MARKET ALERT · {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</div>
    </div>
  </div>
  <div style="background:rgba(90,200,250,0.06);border:1px solid rgba(90,200,250,0.15);border-radius:16px;padding:20px 22px;margin-bottom:20px;">
    <div style="font-size:10px;color:#4a7a9a;letter-spacing:2px;margin-bottom:6px;">AI SIGNAL · {market_label.upper()}</div>
    <div style="font-size:44px;font-weight:800;color:{rec_color};letter-spacing:-2px;line-height:1;">{rec}</div>
    <div style="font-size:12px;color:#6a9ab0;margin-top:4px;">Confidence: {con} &nbsp;·&nbsp; Source: {src}</div>
    <div style="font-size:13px;color:#a0bdd0;margin-top:12px;line-height:1.7;">{reasoning.get('reasoning_summary','')}</div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:20px;">
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:14px;">
      <div style="font-size:10px;color:#4a6a8a;letter-spacing:1px;margin-bottom:4px;">CURRENT PRICE</div>
      <div style="font-size:26px;font-weight:700;color:#5ac8fa;">€{s.get('current_price')}</div>
      <div style="font-size:11px;color:#4a6a8a;">/MWh</div>
    </div>
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:14px;">
      <div style="font-size:10px;color:#4a6a8a;letter-spacing:1px;margin-bottom:4px;">24H CHANGE</div>
      <div style="font-size:26px;font-weight:700;color:{'#34c759' if s.get('pct_change_24h',0)>0 else '#ff6b6b'};">{s.get('pct_change_24h',0):+.1f}%</div>
      <div style="font-size:11px;color:#4a6a8a;">vs yesterday</div>
    </div>
  </div>
  {f'<div style="margin-bottom:20px;"><div style="font-size:10px;color:#4a6a8a;letter-spacing:1px;margin-bottom:8px;">ACTIVE ALERTS</div>{alerts_html}</div>' if alerts_html else ''}
  <div style="text-align:center;padding-top:20px;border-top:1px solid rgba(255,255,255,0.06);">
    <div style="font-size:10px;color:#2a4a6a;letter-spacing:1px;">ENERGYAGENT · AUTONOMOUS MARKET INTELLIGENCE</div>
  </div>
</div></body></html>"""

def send_alert_email(alert_data, price_data, reasoning, extra_recipients=[]):
    if not subscribers and not extra_recipients: return
    market_label = price_data.get("market_label", "Europe")
    s = price_data.get("summary", {})
    rec = reasoning.get("recommendation", "HOLD")
    subject = f"⚡ EnergyAgent — {rec} Signal | €{s.get('current_price')}/MWh | {market_label}"
    html = build_alert_email_html(alert_data, price_data, reasoning, market_label)
    for email in list(set(subscribers + extra_recipients)):
        send_brevo_email(email, subject, html)

def send_welcome_email(to_email: str):
    html = f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#060c14;font-family:-apple-system,BlinkMacSystemFont,'SF Pro Display',sans-serif;color:#e8f4ff;">
<div style="max-width:520px;margin:0 auto;padding:40px 24px;text-align:center;">
  <div style="width:56px;height:56px;border-radius:16px;background:linear-gradient(135deg,#5ac8fa,#0a84ff);display:flex;align-items:center;justify-content:center;font-size:24px;margin:0 auto 20px;">⚡</div>
  <h1 style="font-size:24px;font-weight:800;color:#fff;letter-spacing:-0.5px;margin:0 0 8px;">Welcome to EnergyAgent</h1>
  <p style="font-size:14px;color:#6a9ab0;line-height:1.7;margin:0 0 24px;">You're now subscribed to real-time European energy market alerts. You'll receive notifications when high-severity trading signals are detected.</p>
  <div style="background:rgba(90,200,250,0.06);border:1px solid rgba(90,200,250,0.15);border-radius:12px;padding:16px;margin-bottom:24px;">
    <div style="font-size:12px;color:#5ac8fa;">Powered by ENTSO-E Real Prices + Groq LLaMA 3.3 70B</div>
  </div>
  <div style="font-size:11px;color:#2a4a6a;letter-spacing:1px;">ENERGYAGENT · AUTONOMOUS MARKET INTELLIGENCE</div>
</div></body></html>"""
    send_brevo_email(to_email, "⚡ Welcome to EnergyAgent Alerts", html)

# ── TOOLS ─────────────────────────────────────────────────────────────────────
def tool_fetch_energy_news(query="energy market electricity gas prices Europe", max_articles=10):
    now = time.time()
    if last_news_fetch["data"] and (now - last_news_fetch["time"]) < 1800:
        return last_news_fetch["data"]
    try:
        r = requests.get("https://newsapi.org/v2/everything", params={
            "q": query, "apiKey": NEWSAPI_KEY, "language": "en",
            "sortBy": "publishedAt", "pageSize": max_articles,
            "from": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        }, timeout=10)
        data = r.json()
        if data.get("status") != "ok": return last_news_fetch["data"] or {"articles": [], "total": 0}
        articles = [{"title": a.get("title",""), "description": a.get("description",""),
                     "source": a.get("source",{}).get("name",""), "publishedAt": a.get("publishedAt",""),
                     "url": a.get("url","")} for a in data.get("articles",[])]
        result = {"articles": articles, "total": len(articles)}
        last_news_fetch.update({"time": now, "data": result})
        return result
    except Exception as e:
        return last_news_fetch["data"] or {"articles": [], "total": 0}

def tool_analyze_prices(market="DE"):
    meta = MARKET_META.get(market, MARKET_META["DE"])
    real_prices = fetch_day_ahead_prices(market)
    real_summary = get_price_summary(market) if real_prices else None
    if real_prices and real_summary:
        price_points = []
        for p in real_prices[-48:]:
            price_points.append({"timestamp": p["timestamp"], "electricity_eur_mwh": p["price_eur_mwh"],
                "gas_eur_mwh": round(p["price_eur_mwh"] * 0.42 + np.random.randn() * 1.5, 2),
                "renewable_pct": round(np.clip(35 + 20 * np.sin(len(price_points) * 0.3) + np.random.randn() * 3, 5, 85), 1),
                "source": "ENTSO-E"})
        vals = [p["electricity_eur_mwh"] for p in price_points]
        mean_all = np.mean(vals); std_all = np.std(vals)
        anomalies = [{"timestamp": price_points[i]["timestamp"], "price": vals[i]}
                     for i in range(len(vals)) if abs(vals[i] - mean_all) > 2 * std_all]
        return {"market": market, "market_label": meta["label"], "prices": price_points,
                "summary": {**real_summary, "anomalies_detected": len(anomalies), "anomalies": anomalies[:3], "data_source": "ENTSO-E"},
                "data_source": "ENTSO-E"}
    # Fallback simulated
    np.random.seed(int(time.time()) % 1000)
    hours = 48
    timestamps = [(datetime.now() - timedelta(hours=hours-i)).strftime("%Y-%m-%d %H:00") for i in range(hours)]
    base = meta["base"]
    prices = np.clip(base + np.cumsum(np.random.randn(hours)*2) + 15*np.sin(np.linspace(0,4*np.pi,hours)) + np.random.randn(hours)*3, 20, 300)
    gas = prices * 0.4 + np.random.randn(hours) * 2
    renewables = np.clip(30 + 20*np.sin(np.linspace(0,6*np.pi,hours)) + np.random.randn(hours)*5, 5, 80)
    price_points = [{"timestamp": timestamps[i], "electricity_eur_mwh": round(float(prices[i]),2),
                     "gas_eur_mwh": round(float(gas[i]),2), "renewable_pct": round(float(renewables[i]),1), "source": "Simulated"} for i in range(hours)]
    current = float(prices[-1]); mean_24h = float(prices[-24:].mean()); volatility = float(prices[-24:].std())
    pct_change = float((prices[-1]-prices[-24])/prices[-24]*100)
    anomalies = [{"timestamp": timestamps[i], "price": round(float(prices[i]),2)} for i in range(len(prices)) if abs(prices[i]-prices.mean()) > 2*prices.std()]
    return {"market": market, "market_label": meta["label"], "prices": price_points,
            "summary": {"current_price": round(current,2), "mean_24h": round(mean_24h,2), "volatility_24h": round(volatility,2),
                        "trend": "bullish" if current > mean_24h else "bearish", "pct_change_24h": round(pct_change,2),
                        "anomalies_detected": len(anomalies), "anomalies": anomalies[:3], "data_source": "Simulated"},
            "data_source": "Simulated"}

def tool_market_reasoner(news_data, price_data):
    try:
        articles_text = "".join([f"- {a['title']}: {a['description']}\n" for a in news_data.get("articles",[])[:5]])
        s = price_data.get("summary", {}); src = price_data.get("data_source","Unknown")
        prompt = f"""You are an expert energy market analyst. Analyze step by step.
PRICE DATA ({price_data.get('market_label','Europe')}) — Source: {src}:
- Current: €{s.get('current_price')} /MWh | 24h avg: €{s.get('mean_24h')} /MWh
- Volatility: €{s.get('volatility_24h')} /MWh | Trend: {s.get('trend')} | Change: {s.get('pct_change_24h')}%
NEWS:\n{articles_text}
Return ONLY JSON: {{"price_analysis":"...","news_signals":"...","risk_assessment":"...","market_outlook":"...","recommendation":"BUY|SELL|HOLD","confidence":"HIGH|MEDIUM|LOW","reasoning_summary":"...","key_factors":["f1","f2","f3"]}}"""
        r = groq_client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role":"user","content":prompt}], temperature=0.3, max_tokens=1000)
        raw = r.choices[0].message.content.strip()
        if "```json" in raw: raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw: raw = raw.split("```")[1].split("```")[0].strip()
        return {"success": True, "reasoning": json.loads(raw)}
    except Exception as e:
        print(f"❌ Groq market_reasoner error: {e}")
        return {"success": False, "reasoning": {"recommendation":"HOLD","confidence":"LOW","reasoning_summary":"Analysis unavailable.","key_factors":[]}}

def tool_summarize_news(articles):
    try:
        if not articles: return {"overall_sentiment":"NEUTRAL","sentiment_score":0.0,"key_events":[],"supply_signals":"N/A","demand_signals":"N/A","geopolitical_risks":"N/A","market_moving_news":"N/A"}
        text = "\n".join([f"[{a['source']}] {a['title']}: {a['description']}" for a in articles[:8]])
        r = groq_client.chat.completions.create(model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":f'Analyze energy news. Return ONLY JSON: {{"overall_sentiment":"BULLISH|BEARISH|NEUTRAL","sentiment_score":<-1 to 1>,"key_events":["e1","e2","e3"],"supply_signals":"...","demand_signals":"...","geopolitical_risks":"...","market_moving_news":"..."}}\nARTICLES:\n{text}'}],
            temperature=0.2, max_tokens=600)
        raw = r.choices[0].message.content.strip()
        if "```json" in raw: raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw: raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw)
    except Exception as e:
        print(f"❌ Groq summarize_news error: {e}")
        return {"overall_sentiment":"NEUTRAL","sentiment_score":0.0,"key_events":[],"supply_signals":"N/A","demand_signals":"N/A","geopolitical_risks":"N/A","market_moving_news":"N/A"}

def tool_generate_alerts(price_data, reasoning):
    alerts = []; s = price_data.get("summary",{}); cfg = alert_config
    vol = s.get("volatility_24h",0); chg = s.get("pct_change_24h",0); cur = s.get("current_price",0)
    ano = s.get("anomalies_detected",0); rec = reasoning.get("recommendation","HOLD"); con = reasoning.get("confidence","LOW")
    if cur >= cfg.get("price_above",150): alerts.append({"type":"PRICE_THRESHOLD_HIGH","severity":"HIGH","message":f"Price €{cur}/MWh exceeded threshold €{cfg['price_above']}","action":"Review long positions immediately"})
    if cur <= cfg.get("price_below",30):  alerts.append({"type":"PRICE_THRESHOLD_LOW","severity":"HIGH","message":f"Price €{cur}/MWh dropped below threshold €{cfg['price_below']}","action":"Consider buying opportunity"})
    if vol > cfg.get("volatility_above",25): alerts.append({"type":"HIGH_VOLATILITY","severity":"HIGH","message":f"Extreme volatility: €{vol:.1f}/MWh std dev","action":"Reduce position size"})
    elif vol > 10: alerts.append({"type":"ELEVATED_VOLATILITY","severity":"MEDIUM","message":f"Elevated volatility: €{vol:.1f}/MWh std dev","action":"Monitor closely, consider hedging"})
    if abs(chg) > 15: alerts.append({"type":"PRICE_SPIKE","severity":"HIGH","message":f"Price {'surge' if chg>0 else 'crash'}: {chg:+.1f}% in 24h","action":"Immediate review of open positions"})
    elif abs(chg) > 8: alerts.append({"type":"SIGNIFICANT_MOVE","severity":"MEDIUM","message":f"Significant move: {chg:+.1f}% in 24h","action":"Review portfolio exposure"})
    if ano > 0: alerts.append({"type":"ANOMALY_DETECTED","severity":"MEDIUM","message":f"{ano} price anomaly/anomalies detected","action":"Investigate before trading"})
    if rec in ["BUY","SELL"] and con == "HIGH": alerts.append({"type":"TRADING_SIGNAL","severity":"INFO","message":f"Strong {rec} signal with HIGH confidence","action":f"Consider {rec} position with proper risk management"})
    if not alerts: alerts.append({"type":"MARKET_STABLE","severity":"LOW","message":"No significant alerts. Market conditions stable.","action":"Continue monitoring"})
    return alerts

def run_full_analysis(market="DE", notify=False):
    start = time.time(); agent_log = []
    agent_log.append({"step":1,"tool":"news_fetcher","status":"running","message":"Fetching energy market news..."})
    news_data = tool_fetch_energy_news()
    agent_log[-1].update({"status":"done","result":f"Fetched {news_data.get('total',0)} articles"})
    agent_log.append({"step":2,"tool":"price_analyzer","status":"running","message":"Fetching real ENTSO-E prices..."})
    price_data = tool_analyze_prices(market)
    src = price_data.get("data_source","Unknown")
    agent_log[-1].update({"status":"done","result":f"€{price_data['summary']['current_price']}/MWh | {price_data['summary']['trend']} | {src}"})
    agent_log.append({"step":3,"tool":"news_summarizer","status":"running","message":"Extracting news signals..."})
    intel = tool_summarize_news(news_data.get("articles",[]))
    agent_log[-1].update({"status":"done","result":f"Sentiment: {intel.get('overall_sentiment')} ({intel.get('sentiment_score')})"})
    agent_log.append({"step":4,"tool":"market_reasoner","status":"running","message":"Running multi-step AI reasoning..."})
    reasoning_result = tool_market_reasoner(news_data, price_data)
    reasoning = reasoning_result.get("reasoning", {})
    agent_log[-1].update({"status":"done","result":f"{reasoning.get('recommendation')} | {reasoning.get('confidence')}"})
    agent_log.append({"step":5,"tool":"alert_generator","status":"running","message":"Generating alerts..."})
    alerts = tool_generate_alerts(price_data, reasoning)
    agent_log[-1].update({"status":"done","result":f"{len(alerts)} alert(s)"})
    result = {"timestamp": datetime.now().isoformat(), "elapsed_seconds": round(time.time()-start,2),
              "market": market, "data_source": src, "agent_log": agent_log,
              "prices": price_data, "news": {"articles": news_data.get("articles",[])[:6], "intelligence": intel},
              "reasoning": reasoning, "alerts": alerts,
              "recommendation": {"action": reasoning.get("recommendation","HOLD"), "confidence": reasoning.get("confidence","LOW"),
                                 "summary": reasoning.get("reasoning_summary",""), "key_factors": reasoning.get("key_factors",[])}}
    analysis_history.append({"timestamp": result["timestamp"], "market": market,
        "price": price_data["summary"]["current_price"], "trend": price_data["summary"]["trend"],
        "recommendation": reasoning.get("recommendation","HOLD"), "confidence": reasoning.get("confidence","LOW"),
        "pct_change": price_data["summary"]["pct_change_24h"], "volatility": price_data["summary"]["volatility_24h"],
        "data_source": src})
    if len(analysis_history) > 144: analysis_history.pop(0)
    save_json(HISTORY_FILE, analysis_history)
    if notify:
        high_alerts = [a for a in alerts if a["severity"] == "HIGH"]
        now = time.time()
        if high_alerts and (now - last_email_sent["time"]) > EMAIL_COOLDOWN:
            send_alert_email(alerts, price_data, reasoning)
            last_email_sent["time"] = now
            print(f"📧 Alert email sent — next earliest in 2 hours")
        elif high_alerts:
            remaining = int((EMAIL_COOLDOWN - (now - last_email_sent["time"])) / 60)
            print(f"⏳ Alert suppressed — cooldown active ({remaining} min remaining)")
        # WhatsApp alerts — also rate limited
        if high_alerts and (now - last_email_sent["time"]) > EMAIL_COOLDOWN:
            send_whatsapp_alert(alerts, price_data, reasoning)
    return result

def scheduler_loop():
    print("🕐 Scheduler started — running every 30 minutes")
    while True:
        try:
            print(f"⚡ Auto-analysis at {datetime.now().strftime('%H:%M:%S')}")
            result = run_full_analysis(notify=True)
            last_analysis_result["data"] = result
            print(f"✅ Done: {result['recommendation']['action']} | {result['recommendation']['confidence']} | {result['data_source']}")
        except Exception as e:
            print(f"❌ Scheduler error: {e}")
        time.sleep(1800)

@app.on_event("startup")
async def startup():
    threading.Thread(target=scheduler_loop, daemon=True).start()
    print("✅ EnergyAgent v2.2 started — Brevo email, ENTSO-E real data")

@app.get("/")
def root(): return {"status":"online","version":"2.2.0","data_source":"ENTSO-E","email":"Brevo","subscribers":len(subscribers)}
@app.get("/health")
def health(): return {"status":"healthy","timestamp":datetime.now().isoformat(),"subscribers":len(subscribers)}

@app.get("/api/analyze")
def full_analysis(market: str = "DE"):
    if last_analysis_result["data"] and last_analysis_result["data"].get("market") == market:
        age = (datetime.now() - datetime.fromisoformat(last_analysis_result["data"]["timestamp"])).seconds
        if age < 300: return JSONResponse(last_analysis_result["data"])
    result = run_full_analysis(market=market, notify=False)
    last_analysis_result["data"] = result
    return JSONResponse(result)

@app.get("/api/prices")
def get_prices(market: str = "DE"): return JSONResponse(tool_analyze_prices(market))

@app.get("/api/prices/live")
def get_live_price(market: str = "DE"):
    price = get_latest_price(market)
    return JSONResponse({"market": market, "current_price": price or MARKET_META.get(market,{}).get("base",80),
                         "source": "ENTSO-E" if price else "Simulated", "timestamp": datetime.now().isoformat()})

@app.get("/api/news")
def get_news(): return JSONResponse(tool_fetch_energy_news())
@app.get("/api/history")
def get_history(market: str = "DE"):
    filtered = [h for h in analysis_history if h.get("market","DE") == market]
    return JSONResponse({"history": filtered[-48:], "total": len(filtered)})
@app.get("/api/markets")
def get_markets(): return JSONResponse({"markets": [{"code":"DE","label":"Germany","flag":"🇩🇪"},{"code":"UK","label":"United Kingdom","flag":"🇬🇧"},{"code":"FR","label":"France","flag":"🇫🇷"},{"code":"ES","label":"Spain","flag":"🇪🇸"}]})

class SubscribeRequest(BaseModel):
    email: str

@app.post("/api/subscribe")
def subscribe(req: SubscribeRequest):
    email = req.email.strip().lower()
    if not email or "@" not in email: raise HTTPException(status_code=400, detail="Invalid email")
    if email in subscribers: return JSONResponse({"message":"Already subscribed!","subscribed":True})
    subscribers.append(email)
    save_json(SUBSCRIBERS_FILE, subscribers)
    send_welcome_email(email)
    return JSONResponse({"message":"Subscribed! Check your inbox for a welcome email.","subscribed":True})

@app.post("/api/unsubscribe")
def unsubscribe(req: SubscribeRequest):
    email = req.email.strip().lower()
    if email in subscribers:
        subscribers.remove(email); save_json(SUBSCRIBERS_FILE, subscribers)
    return JSONResponse({"message":"Unsubscribed successfully.","subscribed":False})

class AlertConfig(BaseModel):
    price_above: Optional[float] = None
    price_below: Optional[float] = None
    volatility_above: Optional[float] = None

class WASubscribeRequest(BaseModel):
    phone: str

@app.post("/api/wa/subscribe")
def wa_subscribe(req: WASubscribeRequest):
    phone = req.phone.strip().replace(" ","").replace("-","")
    if not phone.startswith("+"): phone = "+" + phone
    if phone in wa_subscribers:
        return JSONResponse({"message":"Already subscribed!","subscribed":True})
    wa_subscribers.append(phone)
    save_json(WA_SUBSCRIBERS_FILE, wa_subscribers)
    # Send welcome WhatsApp
    send_whatsapp(phone, """⚡ *Welcome to EnergyAgent!*
━━━━━━━━━━━━━━━
You're now subscribed to real-time European energy market alerts via WhatsApp!

You'll receive alerts when high-severity trading signals are detected.

🌐 Dashboard: https://energy-agent-gilt.vercel.app
━━━━━━━━━━━━━━━
_Powered by ENTSO-E Real Prices + Groq AI_""")
    return JSONResponse({"message":"Subscribed! Check WhatsApp for welcome message.","subscribed":True})

@app.post("/api/wa/unsubscribe")
def wa_unsubscribe(req: WASubscribeRequest):
    phone = req.phone.strip()
    if phone in wa_subscribers:
        wa_subscribers.remove(phone)
        save_json(WA_SUBSCRIBERS_FILE, wa_subscribers)
    return JSONResponse({"message":"Unsubscribed from WhatsApp alerts.","subscribed":False})

@app.post("/api/alerts/config")
def set_alert_config(cfg: AlertConfig):
    if cfg.price_above is not None: alert_config["price_above"] = cfg.price_above
    if cfg.price_below is not None: alert_config["price_below"] = cfg.price_below
    if cfg.volatility_above is not None: alert_config["volatility_above"] = cfg.volatility_above
    save_json(ALERTS_FILE, alert_config)
    return JSONResponse({"message":"Alert config updated","config":alert_config})

@app.get("/api/alerts/config")
def get_alert_config(): return JSONResponse(alert_config)

@app.post("/api/ask")
async def ask_agent(request: dict):
    question = request.get("question",""); market = request.get("market","DE")
    if not question: raise HTTPException(status_code=400, detail="Question required")
    try:
        price_data = tool_analyze_prices(market); s = price_data.get("summary",{}); src = price_data.get("data_source","Unknown")
        r = groq_client.chat.completions.create(model="llama-3.1-8b-instant",
            messages=[{"role":"user","content":f"You are EnergyAgent, expert AI for European energy trading.\nMarket ({price_data.get('market_label','Europe')}) — {src}: €{s.get('current_price')}/MWh | {s.get('trend')} | {s.get('pct_change_24h')}% | Vol €{s.get('volatility_24h')}/MWh\nQuestion: {question}\nAnswer concisely and professionally."}],
            temperature=0.4, max_tokens=500)
        return JSONResponse({"answer":r.choices[0].message.content.strip(),"timestamp":datetime.now().isoformat(),"data_source":src})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/test-email")
def test_email(request: dict):
    email = request.get("email","")
    if not email: raise HTTPException(status_code=400, detail="Email required")
    result = run_full_analysis(notify=False)
    send_alert_email(result["alerts"], result["prices"], result["reasoning"], [email])
    return JSONResponse({"message":f"Test email sent to {email}"})