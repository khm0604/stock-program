# -*- coding: utf-8 -*-
# Streamlit 주식 분석 웹앱 (모바일/PC 공용)
from __future__ import annotations

import os, json, math, time, requests
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import yfinance as yf
import streamlit as st
from rapidfuzz import process, fuzz
import plotly.graph_objects as go

BASE = Path(__file__).parent
CFG_PATH = BASE / "config.json"
WATCH_PATH = BASE / "watchlist.json"

# 기본 설정 파일 없으면 생성
DEFAULT_CFG = {
    "period": "1y",
    "interval": "1d",
    "benchmark": "^KS11",
    "universe": ["005930.KS","000660.KS","035420.KS","035720.KS","AAPL","MSFT","NVDA","AMZN","TSLA","^GSPC","^IXIC"],
}
if not CFG_PATH.exists():
    CFG_PATH.write_text(json.dumps(DEFAULT_CFG, ensure_ascii=False, indent=2), encoding="utf-8")
CFG = json.loads(CFG_PATH.read_text(encoding="utf-8"))

# ---------- 한글→티커/영문 별칭 ----------
KOR_TO_TICKER = {
    "엔비디아":"NVDA","nvidia":"NVDA","엔비디어":"NVDA",
    "삼성전자":"005930.KS","삼전":"005930.KS","삼성":"005930.KS",
    "하이닉스":"000660.KS","sk하이닉스":"000660.KS","sk하닉":"000660.KS",
    "카카오":"035720.KS","카톡":"035720.KS",
    "네이버":"035420.KS","라인":"035420.KS",
    "애플":"AAPL","마이크로소프트":"MSFT","ms":"MSFT","아마존":"AMZN","테슬라":"TSLA",
}

# ---------- 야후 파이낸스 검색 API ----------
YF_SEARCH = "https://query1.finance.yahoo.com/v1/finance/search"

def yahoo_search(query: str, count: int = 35) -> List[Dict]:
    try:
        r = requests.get(YF_SEARCH, params={"q":query,"quotesCount":count,"newsCount":0,"listsCount":0}, timeout=7)
        r.raise_for_status()
        out = []
        for q in r.json().get("quotes", []):
            out.append({
                "symbol": q.get("symbol"),
                "name": q.get("shortname") or q.get("longname") or "",
                "exch": q.get("exchDisp") or "",
                "type": q.get("typeDisp") or "",
            })
        return out
    except Exception:
        # 실패 시 로컬 유니버스 대체
        return [{"symbol": s, "name": ""} for s in CFG["universe"]]

def fuzzy_pick(query: str, candidates: List[Dict], limit=20) -> List[Dict]:
    strings = [f"{c['symbol']} {c.get('name','')}" for c in candidates]
    matches = process.extract(query, strings, scorer=fuzz.WRatio, limit=limit)
    picked, seen = [], set()
    for _, score, idx in matches:
        if score >= 60:
            sym = candidates[idx]["symbol"]
            if sym not in seen:
                picked.append(candidates[idx])
                seen.add(sym)
    return picked

# ---------- 캐시 ----------
@st.cache_data(show_spinner=False)
def get_history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    return df.dropna()

@st.cache_data(show_spinner=False)
def get_meta(symbols: List[str]) -> pd.DataFrame:
    rows = []
    for s in symbols:
        info = {}
        try:
            t = yf.Ticker(s)
            # get_info() 새버전/구버전 호환
            try:
                info = t.get_info()
            except Exception:
                info = t.info
        except Exception:
            info = {}
        rows.append({
            "symbol": s,
            "name": info.get("shortName") or info.get("longName") or s,
            "sector": info.get("sector") or "",
            "industry": info.get("industry") or "",
            "marketCap": info.get("marketCap"),
            "avgVolume": info.get("averageVolume") or info.get("averageDailyVolume10Day"),
        })
    return pd.DataFrame(rows)

# ---------- 랭킹 지표 ----------
def score_popularity(meta: pd.DataFrame) -> pd.Series:
    # 시총 + 평균거래량을 정규화하여 합산
    mc = (meta["marketCap"].fillna(0).astype("float")).copy()
    vol = (meta["avgVolume"].fillna(0).astype("float")).copy()
    mc_s = (mc - mc.min()) / (mc.max() - mc.min() + 1e-9)
    vol_s = (vol - vol.min()) / (vol.max() - vol.min() + 1e-9)
    return 0.6*mc_s + 0.4*vol_s

def score_spike(hist_map: Dict[str, pd.DataFrame]) -> pd.Series:
    # 최근 1개월 수익률 + 최근 5일 변동성
    out = {}
    for s, df in hist_map.items():
        if df.empty or "Close" not in df.columns: 
            out[s] = 0.0; continue
        close = df["Close"]
        # 1M 수익률
        ret_m = (close.iloc[-1] / close.iloc[max(0, len(close)-21)] - 1.0) if len(close) > 1 else 0.0
        # 5D 변동성
        vol5 = close.pct_change().iloc[-5:].std() if len(close) > 5 else 0.0
        out[s] = 0.7*ret_m + 0.3*vol5
    ser = pd.Series(out)
    # 정규화
    return (ser - ser.min()) / (ser.max() - ser.min() + 1e-9)

def score_stability(hist_map: Dict[str, pd.DataFrame]) -> pd.Series:
    # 변동성 낮을수록 점수 높음 (최근 60D)
    out = {}
    for s, df in hist_map.items():
        if df.empty or "Close" not in df.columns:
            out[s] = 0.0; continue
        ret = df["Close"].pct_change().iloc[-60:]
        vol = ret.std() if len(ret)>5 else 0.0
        out[s] = 1.0 / (vol + 1e-9)  # 역수
    ser = pd.Series(out)
    return (ser - ser.min()) / (ser.max() - ser.min() + 1e-9)

# ---------- 차트 ----------
def plot_price(df: pd.DataFrame, symbol: str, use_candle: bool):
    fig = go.Figure()
    if use_candle and {"Open","High","Low","Close"}.issubset(df.columns):
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
        ))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    if len(df) > 20:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(20).mean(), name="MA20"))
    if len(df) > 60:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(60).mean(), name="MA60"))
    fig.update_layout(height=520, xaxis_rangeslider_visible=True, template="plotly_white", title=symbol)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# ---------- 성장 기대치 근거 ----------
def growth_rationale(symbol: str) -> str:
    t = yf.Ticker(symbol)
    lines = []
    try:
        fin = t.financials
        if fin is not None and not fin.empty and "Total Revenue" in fin.index:
            rev = fin.loc["Total Revenue"].dropna().sort_index()
            if len(rev) >= 2:
                yoy = (rev.iloc[-1]/rev.iloc[-2]-1)*100
                lines.append(f"- 최근 연매출 성장률: **{yoy:.1f}%**")
    except Exception:
        pass
    try:
        qf = t.quarterly_financials
        if qf is not None and not qf.empty and "Total Revenue" in qf.index:
            q = qf.loc["Total Revenue"].dropna().sort_index()
            if len(q) >= 8:
                ttm = (q.iloc[-4:].sum()/q.iloc[-8:-4].sum()-1)*100
                lines.append(f"- TTM(최근4분기) 매출 성장률: **{ttm:.1f}%**")
    except Exception:
        pass
    try:
        earn = t.earnings
        if earn is not None and not earn.empty and "Earnings" in earn.columns:
            gr = earn["Earnings"].pct_change().iloc[-1]*100
            if pd.notna(gr):
                lines.append(f"- 연간 순이익 성장률: **{gr:.1f}%**")
    except Exception:
        pass
    if not lines:
        lines.append("• 재무 데이터가 부족합니다(야후 제한).")
    return "\n".join(lines)

# ---------- 기간 선택 ----------
def range_to_period_interval(rng: str) -> Tuple[str,str]:
    m = {
        "1D":("5d","5m"), "5D":("10d","15m"), "1M":("1mo","30m"), "3M":("3mo","1d"),
        "6M":("6mo","1d"), "1Y":("1y","1d"), "2Y":("2y","1d"), "5Y":("5y","1wk"),
        "10Y":("10y","1wk"), "MAX":("max","1mo"),
    }
    return m.get(rng, ("1y","1d"))

# ---------- 워치리스트 ----------
def load_watch()->List[str]:
    if WATCH_PATH.exists():
        try: return json.loads(WATCH_PATH.read_text(encoding="utf-8"))
        except Exception: return []
    return []
def save_watch(lst: List[str]):
    WATCH_PATH.write_text(json.dumps(lst, ensure_ascii=False, indent=2), encoding="utf-8")

# ===================== UI =====================
st.set_page_config(page_title="주식 분석 프로그램 (Web)", layout="wide")

if "watch" not in st.session_state: st.session_state.watch = load_watch()
if "results" not in st.session_state: st.session_state.results = []

st.title("📈 주식 분석 프로그램 (Web)")

col0, col1, col2 = st.columns([3,1.5,2])
with col0:
    q = st.text_input("🔎 종목 검색 (한글/영문/오타 허용)", placeholder="예) 엔비디아 / 삼성전자 / AAPL / NVDA ...")
with col1:
    rng = st.selectbox("차트 기간", ["1D","5D","1M","3M","6M","1Y","2Y","5Y","10Y","MAX"], index=5)
with col2:
    do_search = st.button("검색")

# ---- 검색 처리 ----
if do_search and q.strip():
    query = q.strip().lower()
    base = KOR_TO_TICKER.get(query, None)
    if base is None:
        # 별칭 부분 포함
        for k, v in KOR_TO_TICKER.items():
            if k in query:
                base = v
                break
    base = base or query
    raw = yahoo_search(base, 40)
    st.session_state.results = fuzzy_pick(base, raw, limit=25)

left, right = st.columns([1.2, 2])

with left:
    st.subheader("검색 결과")
    labels, sel_symbol = [], None
    if st.session_state.results:
        for r in st.session_state.results:
            labels.append(f"{r['symbol']} — {r.get('name','')}")
        picked = st.radio("클릭해서 선택:", labels, index=0, label_visibility="collapsed")
        sel_symbol = picked.split("—")[0].strip()
    else:
        st.info("검색 결과가 여기에 표시됩니다.")

    st.divider()
    st.subheader("워치리스트")
    if st.session_state.watch:
        st.write(", ".join(st.session_state.watch))
    if sel_symbol and st.button("현재 선택 종목 추가"):
        if sel_symbol not in st.session_state.watch:
            st.session_state.watch.append(sel_symbol)
            save_watch(st.session_state.watch)
            st.success(f"{sel_symbol} 추가됨")

    st.divider()
    st.subheader("카테고리(섹터/산업) 검색")
    sec_kw = st.text_input("Sector (예: energy / technology / financial)")
    ind_kw = st.text_input("Industry (예: semiconductors / software / automobiles)")
    if st.button("카테고리 검색 실행"):
        # 후보군: 검색결과 또는 워치리스트 또는 universe
        candidates = [r["symbol"] for r in st.session_state.results] or st.session_state.watch or CFG["universe"]
        meta = get_meta(list(dict.fromkeys(candidates))[:80])
        df = meta.copy()
        if sec_kw: df = df[df["sector"].str.lower().str.contains(sec_kw.strip().lower(), na=False)]
        if ind_kw: df = df[df["industry"].str.lower().str.contains(ind_kw.strip().lower(), na=False)]
        if df.empty:
            st.warning("조건에 맞는 종목이 없습니다.")
        else:
            st.dataframe(df[["symbol","name","sector","industry","marketCap","avgVolume"]], use_container_width=True)
            st.session_state.results = [{"symbol": s, "name": n} for s, n in zip(df["symbol"], df["name"])]

with right:
    st.subheader("차트 · 성장 근거 · 정렬")

    # 어떤 종목을 그릴지
    symbol = None
    if st.session_state.results:
        symbol = st.session_state.results[0]["symbol"]
    elif st.session_state.watch:
        symbol = st.session_state.watch[0]

    # 정렬 기준
    sort_mode = st.segmented_control("정렬", ["인기순","급등 가능성","안정성"], default="인기순")

    # 후보군 만들기 (리스트 표시에 사용)
    candidates = [r["symbol"] for r in st.session_state.results] or st.session_state.watch or CFG["universe"]
    uniq = list(dict.fromkeys(candidates))[:50]

    # 메타 및 히스토리 수집
    meta = get_meta(uniq)
    period, interval = range_to_period_interval(rng)
    hist_map = {s: get_history(s, period, interval) for s in uniq}

    # 스코어 계산
    if sort_mode == "인기순":
        meta["score"] = score_popularity(meta)
    elif sort_mode == "급등 가능성":
        meta["score"] = score_spike(hist_map)
    else:
        meta["score"] = score_stability(hist_map)

    top = meta.sort_values("score", ascending=False).head(15)

    st.markdown("**랭킹 (상위 15)**")
    st.dataframe(top[["symbol","name","sector","industry","marketCap","avgVolume","score"]], use_container_width=True)

    # 메인 차트
    show_candle = st.toggle("캔들차트", value=True)
    if symbol:
        df = hist_map.get(symbol, pd.DataFrame())
        if df is None or df.empty:
            st.error(f"{symbol} 데이터 없음")
        else:
            plot_price(df, symbol, show_candle)
            st.markdown("**성장 기대치 근거**")
            st.markdown(growth_rationale(symbol))
    else:
        st.info("좌측에서 종목을 선택하거나 검색하세요.")
