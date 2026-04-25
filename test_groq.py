# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import os, json
from groq import Groq

load_dotenv(dotenv_path='.env')
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

r = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role":"user","content":'Return ONLY this JSON with no extra text: {"recommendation":"BUY","confidence":"HIGH","reasoning_summary":"test summary","key_factors":["factor1"],"price_analysis":"price analysis","news_signals":"news signals","risk_assessment":"risk assessment","market_outlook":"market outlook"}'}],
    temperature=0.3,
    max_tokens=500
)
raw = r.choices[0].message.content.strip()
print("RAW:", raw[:300])
try:
    parsed = json.loads(raw)
    print("PARSED OK:", parsed["recommendation"])
except Exception as e:
    print("PARSE ERROR:", e)
