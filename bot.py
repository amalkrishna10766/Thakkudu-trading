"""
NSE Trading Screener Bot for Discord — v2
- 8:30 AM IST: Swing scan (daily setups)
- 9:20 AM IST: Intraday scan (gap + opening range) + NIFTY options view
- !scan  : manual swing scan
- !intra : manual intraday scan
- !nifty : manual NIFTY options view

DISCLAIMER: Screener output, not financial advice. No trade is 100%.
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

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
CHANNEL_ID = int(os.environ["CHANNEL_ID"])
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "BAJFINANCE.NS", "HCLTECH.NS", "WIPRO.NS", "ADANIENT.NS",
    "NTPC.NS", "POWERGRID.NS", "ONGC.NS", "COALINDIA.NS", "JSWSTEEL.NS",
    "HINDALCO.NS", "M&M.NS", "TECHM.NS", "DRREDDY.NS", "CIPLA.NS",
]


# ---------------- HELPERS ----------------
def _flat(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def rsi(s, period=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + g / l))


def atr(df, period=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()


def make_chart(sig):
    buf = io.BytesIO()
    df = sig["df"]
    ap = []
    if "EMA20" in df:
        ap = [mpf.make_addplot(df["EMA20"], color="orange", width=1),
              mpf.make_addplot(df["EMA50"], color="blue", width=1)]
    mpf.plot(df, type="candle", style="yahoo", addplot=ap, volume=True,
             title=f"{sig['symbol']} — {sig['type']}",
             hlines=dict(hlines=[sig["sl"], sig["target"]],
                         colors=["red", "green"], linestyle="--"),
             savefig=dict(fname=buf, dpi=110, bbox_inches="tight"))
    buf.seek(0)
    return buf


# ---------------- SWING (daily) ----------------
def analyze_swing(symbol):
    df = yf.download(symbol, period="6mo", interval="1d",
                     progress=False, auto_adjust=True)
    if df is None or len(df) < 60:
        return None
    df = _flat(df)
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["RSI"] = rsi(df["Close"])
    df["ATR"] = atr(df)
    df["VolAvg"] = df["Volume"].rolling(20).mean()

    last, prev = df.iloc[-1], df.iloc[-2]
    price, a = float(last["Close"]), float(last["ATR"])
    sig = None

    if (last["EMA20"] > last["EMA50"] and 40 <= last["RSI"] <= 60
            and last["Close"] > last["EMA20"] >= prev["Low"]):
        sig = ("SWING", price, round(price - 1.5 * a, 1), round(price + 3 * a, 1),
               "Uptrend (EMA20>EMA50), pullback to EMA20 bounce, RSI neutral")
    elif (last["Volume"] > 2 * last["VolAvg"]
          and last["Close"] > df["High"].iloc[-21:-1].max()):
        sig = ("MOMENTUM", price, round(price - a, 1), round(price + 2 * a, 1),
               "20-day high breakout on 2x volume")

    if not sig:
        return None
    kind, entry, sl, tgt, reason = sig
    rr = round((tgt - entry) / max(entry - sl, 0.01), 2)
    if rr < 1.5:
        return None
    return {"symbol": symbol.replace(".NS", ""), "type": kind,
            "price": round(entry, 1), "sl": sl, "target": tgt, "rr": rr,
            "rsi": round(float(last["RSI"]), 1), "reason": reason,
            "df": df.tail(90)}


# ---------------- INTRADAY (5-min, run after 9:20 IST) ----------------
def analyze_intraday(symbol):
    intra = yf.download(symbol, period="5d", interval="5m",
                        progress=False, auto_adjust=True)
    daily = yf.download(symbol, period="1mo", interval="1d",
                        progress=False, auto_adjust=True)
    if intra is None or daily is None or len(intra) < 5 or len(daily) < 5:
        return None
    intra, daily = _flat(intra), _flat(daily)

    today = intra.index[-1].date()
    tday = intra[intra.index.date == today]
    if len(tday) < 2:
        return None

    prev_close = float(daily["Close"].iloc[-2]) if daily.index[-1].date() == today \
        else float(daily["Close"].iloc[-1])
    open_px = float(tday["Open"].iloc[0])
    last_px = float(tday["Close"].iloc[-1])
    or_high = float(tday["High"].iloc[:3].max())   # first 15 min range
    or_low = float(tday["Low"].iloc[:3].min())
    gap_pct = (open_px - prev_close) / prev_close * 100
    avg_vol = intra["Volume"].rolling(20).mean().iloc[-1]
    cur_vol = tday["Volume"].iloc[-1]

    sig = None
    # Gap-up hold + ORB long
    if gap_pct > 0.5 and last_px > or_high and cur_vol > 1.5 * avg_vol:
        sl = round(or_low, 1)
        tgt = round(last_px + 2 * (last_px - sl), 1)
        sig = ("INTRADAY LONG (ORB)", last_px, sl, tgt,
               f"Gap-up {gap_pct:.1f}%, holding above opening range high, volume surge")
    # Gap-down + ORB short
    elif gap_pct < -0.5 and last_px < or_low and cur_vol > 1.5 * avg_vol:
        sl = round(or_high, 1)
        tgt = round(last_px - 2 * (sl - last_px), 1)
        sig = ("INTRADAY SHORT (ORB)", last_px, sl, tgt,
               f"Gap-down {gap_pct:.1f}%, breaking opening range low, volume surge")

    if not sig:
        return None
    kind, entry, sl, tgt, reason = sig
    rr = round(abs(tgt - entry) / max(abs(entry - sl), 0.01), 2)
    return {"symbol": symbol.replace(".NS", ""), "type": kind,
            "price": round(entry, 1), "sl": sl, "target": tgt, "rr": rr,
            "rsi": 0, "reason": reason, "df": tday.tail(75)}


# ---------------- NIFTY OPTIONS VIEW ----------------
def nifty_options_view():
    df = yf.download("^NSEI", period="3mo", interval="1d",
                     progress=False, auto_adjust=True)
    intra = yf.download("^NSEI", period="2d", interval="5m",
                        progress=False, auto_adjust=True)
    if df is None or len(df) < 30:
        return None
    df, intra = _flat(df), _flat(intra)
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["RSI"] = rsi(df["Close"])
    df["ATR"] = atr(df)

    spot = float(intra["Close"].iloc[-1]) if intra is not None and len(intra) else \
        float(df["Close"].iloc[-1])
    a = float(df["ATR"].iloc[-1])
    r = float(df["RSI"].iloc[-1])
    ema = float(df["EMA20"].iloc[-1])
    atm = round(spot / 50) * 50  # NIFTY strikes in 50s

    prev_close = float(df["Close"].iloc[-2])
    chg = (spot - prev_close) / prev_close * 100

    if spot > ema and r > 55:
        bias, idea = "BULLISH 🟢", f"Consider **{atm} CE / {atm + 50} CE**"
    elif spot < ema and r < 45:
        bias, idea = "BEARISH 🔴", f"Consider **{atm} PE / {atm - 50} PE**"
    else:
        bias, idea = "SIDEWAYS 🟡", "No clear direction — avoid buying options today (theta decay)"

    return {"spot": round(spot, 1), "chg": round(chg, 2), "atm": atm,
            "bias": bias, "idea": idea, "rsi": round(r, 1),
            "sl_pts": round(0.5 * a), "tgt_pts": round(a)}


# ---------------- DISCORD ----------------
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


async def post_signals(channel, signals, header):
    await channel.send(header)
    if not signals:
        await channel.send("😐 No high-quality setups. No trade is also a trade.")
        return
    for sig in signals:
        tv = f"https://www.tradingview.com/chart/?symbol=NSE:{sig['symbol']}"
        embed = discord.Embed(title=f"{sig['symbol']} — {sig['type']}",
                              description=sig["reason"], color=0x2ecc71, url=tv)
        embed.add_field(name="Entry (CMP)", value=f"₹{sig['price']}")
        embed.add_field(name="Target", value=f"₹{sig['target']}")
        embed.add_field(name="Stop-Loss", value=f"₹{sig['sl']}")
        embed.add_field(name="Risk:Reward", value=f"1:{sig['rr']}")
        if sig["rsi"]:
            embed.add_field(name="RSI", value=str(sig["rsi"]))
        embed.set_footer(text="Yahoo data (~15 min delay). Verify price before entry.")
        try:
            chart = await asyncio.to_thread(make_chart, sig)
            await channel.send(embed=embed,
                               file=discord.File(chart, f"{sig['symbol']}.png"))
        except Exception:
            await channel.send(embed=embed)
        await asyncio.sleep(2)


async def run_swing(channel):
    today = dt.datetime.now(IST).strftime("%d %b %Y")
    sigs = []
    for s in WATCHLIST:
        try:
            r = await asyncio.to_thread(analyze_swing, s)
            if r:
                sigs.append(r)
        except Exception as e:
            print(f"{s}: {e}")
    await post_signals(channel, sigs,
        f"📊 **Swing Scan — {today}**\n⚠️ Screener output, not guaranteed. Risk 1-2% max per trade.")


async def run_intraday(channel):
    now = dt.datetime.now(IST).strftime("%d %b %Y %H:%M")
    sigs = []
    for s in WATCHLIST:
        try:
            r = await asyncio.to_thread(analyze_intraday, s)
            if r:
                sigs.append(r)
        except Exception as e:
            print(f"{s}: {e}")
    await post_signals(channel, sigs,
        f"⚡ **Intraday Scan — {now} IST** (gap + opening range)\n"
        f"⚠️ Intraday = strict SL. Exit all positions by 3:15 PM.")


async def run_nifty(channel):
    try:
        v = await asyncio.to_thread(nifty_options_view)
    except Exception as e:
        print(f"nifty: {e}")
        v = None
    if not v:
        await channel.send("NIFTY data fetch failed, try again later.")
        return
    embed = discord.Embed(title="🎯 NIFTY 50 Options View", color=0x3498db,
        url="https://www.tradingview.com/chart/?symbol=NSE:NIFTY")
    embed.add_field(name="Spot", value=f"{v['spot']} ({v['chg']:+.2f}%)")
    embed.add_field(name="Bias", value=v["bias"])
    embed.add_field(name="ATM Strike", value=str(v["atm"]))
    embed.add_field(name="Idea", value=v["idea"], inline=False)
    embed.add_field(name="SL (index pts)", value=f"~{v['sl_pts']} pts against you")
    embed.add_field(name="Target (index pts)", value=f"~{v['tgt_pts']} pts")
    embed.add_field(name="RSI (daily)", value=str(v["rsi"]))
    embed.set_footer(text="Premium/OI data not included — check option chain in your "
                          "broker app before entry. Options can go to zero; size small.")
    await channel.send(embed=embed)


@tasks.loop(minutes=1)
async def scheduler():
    now = dt.datetime.now(IST)
    if now.weekday() >= 5:
        return
    ch = client.get_channel(CHANNEL_ID)
    if ch is None:
        return
    if now.hour == 8 and now.minute == 30:      # pre-market swing
        await run_swing(ch)
    elif now.hour == 9 and now.minute == 20:    # post-open intraday + nifty
        await run_nifty(ch)
        await run_intraday(ch)


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    if not scheduler.is_running():
        scheduler.start()


@client.event
async def on_message(msg):
    if msg.author.bot:
        return
    cmd = msg.content.strip().lower()
    if cmd == "!scan":
        await run_swing(msg.channel)
    elif cmd == "!intra":
        await run_intraday(msg.channel)
    elif cmd == "!nifty":
        await run_nifty(msg.channel)
    elif cmd == "!help":
        await msg.channel.send(
            "**Commands:** `!scan` swing • `!intra` intraday • `!nifty` NIFTY options view\n"
            "**Auto:** 8:30 AM swing scan, 9:20 AM intraday + NIFTY (Mon-Fri)")


client.run(DISCORD_TOKEN)
