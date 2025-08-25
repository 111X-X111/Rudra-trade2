#!/usr/bin/env python3
# streamlit_test2.py — Streamlit UI for OI tracker (access-token flow) with target expiry selection
#
# Install:
#   pip install streamlit kiteconnect pandas pytz
#
# Run:
#   streamlit run streamlit_test2.py
#
# Notes:
# - Uses ACCESS TOKEN (no request-token). Enter via sidebar or env var KITE_ACCESS_TOKEN.
# - Lets you choose a target expiry from available expiries.
# - Shows CALL/PUT OI tables around ATM with % change over 5/10/15/30 minutes.
# - Auto-refresh toggle.

import os
import time
from datetime import datetime, date, timedelta, timezone

import pandas as pd
import pytz
import streamlit as st
from kiteconnect import KiteConnect


def check_password():
    def password_entered():
        if st.session_state["password"] == "Rudraj@911":   # <-- Change this password
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        st.error("Password incorrect ❌")
        return False
    else:
        return True

# If password wrong → stop app
if not check_password():
    st.stop()
# ====== Config ======
HISTORICAL_DATA_MINUTES = 40
OI_CHANGE_INTERVALS_MIN = (5, 10, 15, 30)
PCT_CHANGE_THRESHOLDS = {5: 8.0, 10: 10.0, 15: 15.0, 30: 25.0}
IST = pytz.timezone("Asia/Kolkata")

EXCHANGE_NFO = KiteConnect.EXCHANGE_NFO
EXCHANGE_LTP = KiteConnect.EXCHANGE_NSE

# ====== Helpers ======
def get_atm_strike(kite_obj: KiteConnect, underlying_sym: str, exch_for_ltp: str, strike_diff: int):
    try:
        ltp_inst = f"{exch_for_ltp}:{underlying_sym}"
        ltp_data = kite_obj.ltp(ltp_inst)
        if not ltp_data or ltp_inst not in ltp_data or "last_price" not in ltp_data[ltp_inst]:
            return None, None
        ltp = ltp_data[ltp_inst]["last_price"]
        atm = round(ltp / strike_diff) * strike_diff
        return atm, ltp
    except Exception:
        return None, None

def list_expiries(instruments: list, underlying_prefix: str):
    """Return sorted unique future expiries (date objects) for the given underlying in NFO-OPT."""
    today = date.today()
    expiries = set()
    for inst in instruments:
        if inst.get("name") == underlying_prefix and inst.get("exchange") == EXCHANGE_NFO:
            ex = inst.get("expiry")
            if isinstance(ex, date) and ex >= today:
                expiries.add(ex)
    return sorted(list(expiries))

def symbol_prefix_for_expiry(instruments: list, underlying_prefix: str, expiry_dt: date):
    """Pick a representative option's tradingsymbol to infer symbol prefix for the given expiry."""
    for inst in instruments:
        if (inst.get("name") == underlying_prefix and
            inst.get("exchange") == EXCHANGE_NFO and
            inst.get("expiry") == expiry_dt):
            ts = inst.get("tradingsymbol", "")
            return ts[:len(underlying_prefix)+5] if ts else ""
    return ""

def get_relevant_option_details(instruments: list, atm: float, expiry_dt: date,
                                strike_diff: int, levels: int,
                                underlying_prefix: str, symbol_prefix: str):
    out = {}
    if not expiry_dt or atm is None:
        return out
    for i in range(-levels, levels+1):
        strike = atm + i*strike_diff
        ce_found, pe_found = None, None
        for inst in instruments:
            if (inst.get("name") == underlying_prefix and
                inst.get("strike") == strike and
                inst.get("expiry") == expiry_dt and
                inst.get("exchange") == EXCHANGE_NFO):
                if inst.get("instrument_type") == "CE" and symbol_prefix in inst.get("tradingsymbol",""):
                    ce_found = inst
                if inst.get("instrument_type") == "PE" and symbol_prefix in inst.get("tradingsymbol",""):
                    pe_found = inst
            if ce_found and pe_found:
                break
        suffix = "atm" if i==0 else (f"itm{-i}" if i<0 else f"otm{i}")
        if ce_found:
            out[f"{suffix}_ce"] = {"tradingsymbol": ce_found["tradingsymbol"],
                                   "instrument_token": ce_found["instrument_token"],
                                   "strike": strike}
        if pe_found:
            out[f"{suffix}_pe"] = {"tradingsymbol": pe_found["tradingsymbol"],
                                   "instrument_token": pe_found["instrument_token"],
                                   "strike": strike}
    return out

def fetch_historical_oi(kite: KiteConnect, option_details: dict, lookback_min: int = HISTORICAL_DATA_MINUTES):
    store = {}
    to_dt = datetime.now()
    from_dt = to_dt - timedelta(minutes=lookback_min)
    from_str = from_dt.strftime("%Y-%m-%d %H:%M:00")
    to_str = to_dt.strftime("%Y-%m-%d %H:%M:00")
    for k, d in option_details.items():
        tok = d.get("instrument_token")
        if not tok:
            store[k] = []
            continue
        try:
            data = kite.historical_data(tok, from_str, to_str, interval="minute", oi=True)
            store[k] = data
        except Exception:
            store[k] = []
    return store

def find_oi_at(candles: list, target_time: datetime, latest_pair: tuple | None):
    if not candles:
        return None
    for c in reversed(candles):
        ct = c["date"]
        if ct <= target_time:
            if latest_pair and ct > latest_pair[1]:
                continue
            return c.get("oi")
    return None

def compute_oi_diffs(raw: dict, intervals=(5,10,15,30)):
    out = {}
    now_utc = datetime.now(timezone.utc)
    for key, candles in raw.items():
        latest_oi = None
        latest_ts = None
        if candles:
            latest = candles[-1]
            latest_oi = latest.get("oi")
            latest_ts = latest.get("date")
        out[key] = {"latest_oi": latest_oi, "latest_oi_timestamp": latest_ts}
        if latest_oi is None:
            for m in intervals:
                out[key][f"pct_diff_{m}m"] = None
            continue
        for m in intervals:
            past_time = now_utc - timedelta(minutes=m)
            past_oi = find_oi_at(candles, past_time, (latest_oi, latest_ts))
            pct = None
            if past_oi not in (None, 0):
                pct = (latest_oi - past_oi) / past_oi * 100.0
            out[key][f"pct_diff_{m}m"] = pct
    return out

def build_table(report: dict, details: dict, atm: float, step: int, levels: int, intervals=(5,10,15,30), is_call=True):
    rows = []
    for i in range(-levels, levels+1):
        strike = atm + i*step
        suffix = "atm" if i==0 else ("itm{}".format(-i) if i<0 else "otm{}".format(i))
        key = f"{suffix}_{'ce' if is_call else 'pe'}"
        d = details.get(key, {})
        r = report.get(key, {})
        row = {
            "Strike": int(d.get("strike", strike)),
            "Symbol": d.get("tradingsymbol", "N/A"),
            "Latest OI": r.get("latest_oi"),
            "OI Time (IST)": r.get("latest_oi_timestamp").astimezone(IST).strftime("%H:%M:%S") if r.get("latest_oi_timestamp") else None
        }
        for m in intervals:
            row[f"OI %Chg ({m}m)"] = r.get(f"pct_diff_{m}m")
        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def safe_fmt(val, fmt):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    try:
        return fmt.format(val)
    except Exception:
        return str(val) if val is not None else ""

# ====== Streamlit UI ======
st.set_page_config(page_title="OI Tracker (Target Expiry)", layout="wide")
st.title("📊 OI Tracker — Access Token Flow (Target Expiry)")

with st.sidebar:
    st.header("Broker Session")
    api_key = st.text_input("Kite API Key", value=os.getenv("KITE_API_KEY", ""))
    access_token = st.text_input("Access Token", value=os.getenv("KITE_ACCESS_TOKEN", ""), type="password")

    st.header("Market & Filters")
    underlying = st.selectbox("Underlying", ("NIFTY 50", "NIFTY BANK"), index=0)
    strike_diff = 50 if underlying == "NIFTY 50" else 100
    levels = st.slider("Strikes each side of ATM", 1, 5, 2)

    refresh = st.number_input("Refresh seconds", min_value=10, max_value=300, value=60, step=5)
    autorun = st.checkbox("Auto-refresh", value=False)
    go = st.button("Run / Refresh")

# Auto-refresh
if autorun:
    last = st.session_state.get("last_refresh", 0.0)
    now = time.time()
    if now - last > refresh:
        st.session_state["last_refresh"] = now
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()

if not api_key or not access_token:
    st.info("Enter API Key and Access Token in the sidebar to begin.")
    st.stop()

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Verify login
try:
    prof = kite.profile()
    st.success(f"Connected: {prof.get('user_id')} — {prof.get('user_name')}")
except Exception as e:
    st.error(f"Login failed: {e}")
    st.stop()

# Load instruments
@st.cache_data(ttl=600)
def load_instruments(nfo_exchange: str):
    return kite.instruments(nfo_exchange)

instruments = load_instruments(EXCHANGE_NFO)
under_prefix = underlying.split(" ")[0].upper()

# List available expiries and ask user to pick one
expiries = list_expiries(instruments, under_prefix)
if not expiries:
    st.error("No future expiries found.")
    st.stop()

# Default to nearest (index 0). Let user pick any target expiry.
exp_strs = [ex.strftime("%d-%b-%Y (%a)") for ex in expiries]
sel_idx = st.sidebar.selectbox("Target Expiry", range(len(expiries)), format_func=lambda i: exp_strs[i], index=0)
target_expiry = expiries[sel_idx]

symbol_prefix = symbol_prefix_for_expiry(instruments, under_prefix, target_expiry)

if go or autorun:
    atm, ltp = get_atm_strike(kite, underlying, EXCHANGE_LTP, strike_diff)
    if atm is None:
        st.error("Unable to compute ATM — try again.")
        st.stop()

    st.subheader(f"{underlying} — Spot: {ltp:.2f} | ATM: {int(atm)} | Expiry: {target_expiry.strftime('%d-%b-%Y')}")
    st.caption(f"Updated: {datetime.now(IST).strftime('%d-%b-%Y %H:%M:%S')} IST")

    details = get_relevant_option_details(instruments, atm, target_expiry, strike_diff, levels, under_prefix, symbol_prefix)
    if not details:
        st.warning("No matching option contracts found around ATM for selected expiry.")
        st.stop()

    raw = fetch_historical_oi(kite, details)
    report = compute_oi_diffs(raw)

    calls_df = build_table(report, details, atm, strike_diff, levels, is_call=True)
    puts_df  = build_table(report, details, atm, strike_diff, levels, is_call=False)

    # Safe formatting to avoid NoneType format errors
    fmt_cols = {
        "Latest OI": lambda v: safe_fmt(v, "{:,.0f}")
    }
    for m in OI_CHANGE_INTERVALS_MIN:
        col = f"OI %Chg ({m}m)"
        fmt_cols[col] = (lambda f: (lambda v: safe_fmt(v, f)))("{:+.2f}%")

    st.markdown("### Calls (ΔOI %)")
    st.dataframe(calls_df.style.format(fmt_cols))

    st.markdown("### Puts (ΔOI %)")
    st.dataframe(puts_df.style.format(fmt_cols))


    # Breach summary
    def breach_count(df):
        count = 0
        total = 0
        for m in OI_CHANGE_INTERVALS_MIN:
            col = f"OI %Chg ({m}m)"
            thr = PCT_CHANGE_THRESHOLDS[m]
            vals = df[col].dropna()
            total += len(vals)
            count += (vals.abs() > thr).sum()
        return count, total

    bc, bt = breach_count(calls_df)
    pc, pt = breach_count(puts_df)
    st.info(f"Threshold breaches — Calls: {bc}/{bt}, Puts: {pc}/{pt}")
else:
    st.info("Select a **Target Expiry** and click **Run / Refresh**.")
