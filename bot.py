"""
NSE Trading Screener Bot for Discord
Scans NSE stocks daily and posts swing / intraday / options candidates
with charts, entry, target, stop-loss and risk-reward.

DISCLAIMER: This is a screener, not financial advice. No trade is 100%.
Always use position sizing and stop-losses.
"""

import os
import io
import asyncio
import datetime as dt

import discord
from discord.ext import tasks
import pandas as pd
import yfinance as yf
import mplfinance as mpf

# ---------------- CONFIG ----------------
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
CHANNEL_ID = int(os.environ["CHANNEL_ID"])

# NIFTY 50 + liquid midcaps — edit this list as you like (.NS = NSE on Yahoo)
WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "BAJFINANCE.NS", "HCLTECH.NS", "WIPRO.NS", "ADANIENT.NS",
    "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS", "JSWSTEEL.NS",
    "HINDALCO.NS", "M&M.NS", "TECHM.NS", "DRREDDY.NS", "CIPLA.NS",
]

SCAN_HOUR_IST = 8   # 8:30 AM IST pre-market scan
SCAN_MIN_IST = 30
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


# ---------------- INDICATORS ----------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def analyze(symbol: str):
    """Return a signal dict or None."""
    df = yf.download(symbol, period="6mo", interval="1d",
                     progress=False, auto_adjust=True)
    if df is None or len(df) < 60:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["RSI"] = rsi(df["Close"])
    df["ATR"] = atr(df)
    df["VolAvg"] = df["Volume"].rolling(20).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = float(last["Close"])
    a = float(last["ATR"])

    signal = None

    # --- SWING: EMA trend + pullback bounce ---
    if (last["EMA20"] > last["EMA50"]
            and 40 <= last["RSI"] <= 60
            and last["Close"] > last["EMA20"] >= prev["Low"]):
        entry = price
        sl = round(price - 1.5 * a, 1)
        tgt = round(price + 3 * a, 1)
        signal = ("SWING", entry, sl, tgt,
                  "Uptrend (EMA20>EMA50), pullback to EMA20 bounce, RSI neutral")

    # --- INTRADAY/MOMENTUM: volume breakout ---
    elif (last["Volume"] > 2 * last["VolAvg"]
          and last["Close"] > df["High"].iloc[-21:-1].max()):
        entry = price
        sl = round(price - 1 * a, 1)
        tgt = round(price + 2 * a, 1)
        signal = ("INTRADAY/MOMENTUM", entry, sl, tgt,
                  "20-day high breakout on 2x volume")

    # --- OPTIONS idea: strong trend + high RSI momentum ---
    elif last["RSI"] > 65 and last["EMA20"] > last["EMA50"] and price > prev["Close"]:
        entry = price
        sl = round(price - 1 * a, 1)
        tgt = round(price + 2.5 * a, 1)
        signal = ("OPTIONS (ATM CE idea)", entry, sl, tgt,
                  "Strong momentum, RSI>65 — consider ATM call, strict SL")

    if not signal:
        return None

    kind, entry, sl, tgt, reason = signal
    rr = round((tgt - entry) / max(entry - sl, 0.01), 2)
    if rr < 1.5:
        return None  # skip poor risk-reward

    return {
        "symbol": symbol.replace(".NS", ""),
        "type": kind, "price": round(entry, 1),
        "sl": sl, "target": tgt, "rr": rr,
        "rsi": round(float(last["RSI"]), 1),
        "reason": reason, "df": df.tail(90),
    }


def make_chart(sig) -> io.BytesIO:
    buf = io.BytesIO()
    df = sig["df"]
    ap = [
        mpf.make_addplot(df["EMA20"], color="orange", width=1),
        mpf.make_addplot(df["EMA50"], color="blue", width=1),
    ]
    mpf.plot(df, type="candle", style="yahoo", addplot=ap, volume=True,
             title=f"{sig['symbol']} — {sig['type']}",
             hlines=dict(hlines=[sig["sl"], sig["target"]],
                         colors=["red", "green"], linestyle="--"),
             savefig=dict(fname=buf, dpi=110, bbox_inches="tight"))
    buf.seek(0)
    return buf


# ---------------- DISCORD ----------------
intents = discord.Intents.default()
client = discord.Client(intents=intents)


async def run_scan():
    channel = client.get_channel(CHANNEL_ID)
    if channel is None:
        return
    today = dt.datetime.now(IST).strftime("%d %b %Y")
    await channel.send(
        f"📊 **Daily Scan — {today}**\n"
        f"⚠️ Screener output, not guaranteed calls. Risk max 1-2% per trade.")

    found = 0
    for sym in WATCHLIST:
        try:
            sig = await asyncio.to_thread(analyze, sym)
        except Exception as e:
            print(f"{sym}: {e}")
            continue
        if not sig:
            continue
        found += 1
        tv = f"https://www.tradingview.com/chart/?symbol=NSE:{sig['symbol']}"
        embed = discord.Embed(
            title=f"{sig['symbol']} — {sig['type']}",
            description=sig["reason"], color=0x2ecc71,
            url=tv)
        embed.add_field(name="Entry (CMP)", value=f"₹{sig['price']}")
        embed.add_field(name="Target", value=f"₹{sig['target']}")
        embed.add_field(name="Stop-Loss", value=f"₹{sig['sl']}")
        embed.add_field(name="Risk:Reward", value=f"1:{sig['rr']}")
        embed.add_field(name="RSI", value=str(sig["rsi"]))
        embed.set_footer(text="Data: Yahoo Finance (~15 min delay). Verify before trading.")
        chart = await asyncio.to_thread(make_chart, sig)
        await channel.send(embed=embed,
                           file=discord.File(chart, f"{sig['symbol']}.png"))
        await asyncio.sleep(2)

    if found == 0:
        await channel.send("😐 No high-quality setups today. No trade is also a trade.")


@tasks.loop(minutes=1)
async def scheduler():
    now = dt.datetime.now(IST)
    if (now.hour == SCAN_HOUR_IST and now.minute == SCAN_MIN_IST
            and now.weekday() < 5):
        await run_scan()


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    if not scheduler.is_running():
        scheduler.start()


@client.event
async def on_message(msg):
    if msg.author.bot:
        return
    if msg.content.strip().lower() == "!scan":
        await run_scan()


client.run(DISCORD_TOKEN)
