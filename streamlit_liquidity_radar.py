# streamlit_liquidity_radar.py  —  Worker 版（最終）
# 功能：統一排序 + 方向優先 + 多人快取 + 雲端容錯 + 透過 Cloudflare Worker 代理 Binance
# 依賴：streamlit pandas requests python-dateutil numpy

import time, random
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import tz
import streamlit as st

# ===================== 必改：把這行換成你的 Workers 網址 =====================
WORKER_BASE = "https://binance-proxy.tonytien1980.workers.dev/"   # 例： https://binance-proxy-xxxxx.workers.dev
# =============================================================================

# 透過 Worker 走 fapi 與 /futures/data/ 端點
FAPI_BASES = [WORKER_BASE]         # 我們統一打到 Worker，讓 Worker 轉發並快取
DATA_BASE  = WORKER_BASE
UA = {"User-Agent": "liquidity-radar/streamlit-1.0"}

# 若 fapi 24h/klines 暫時不可用，改用白名單維持功能（20 檔）
FALLBACK_SYMBOLS = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","ADAUSDT","LINKUSDT","LTCUSDT","TRXUSDT",
    "BCHUSDT","DOTUSDT","AVAXUSDT","MATICUSDT","UNIUSDT","ATOMUSDT","FILUSDT","APTUSDT","SUIUSDT","NEARUSDT"
]

TZ_LOCAL = tz.gettz("Asia/Taipei")
st.set_page_config(page_title="USDTⓈ-M 流動性雷達（5m）", layout="wide")

# ---------- 色帶工具（無需 matplotlib） ----------
def _hex(rgb): return "#{:02x}{:02x}{:02x}".format(*rgb)
def _interp(c1, c2, t): return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))
def _norm_series(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.isna().all(): return pd.Series([np.nan]*len(s), index=s.index)
    vmin, vmax = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax: return pd.Series([np.nan]*len(s), index=s.index)
    return (s - vmin) / (vmax - vmin)
def make_bg_styles(series: pd.Series, start_rgb, end_rgb):
    t = _norm_series(series); styles=[]
    for val in t:
        if pd.isna(val): styles.append("")
        else:
            rgb = _interp(start_rgb, end_rgb, float(val))
            styles.append(f"background-color: {_hex(rgb)}")
    return styles
COLORS = {
    "spike": ((255, 245, 157), (244, 67, 54)),   # 淡黃->紅（放量）
    "netbuy": ((240, 249, 232), (56, 142, 60)),  # 淡綠->綠（買氣）
    "oi": ((236, 231, 242), (3, 136, 166)),      # 淡紫->藍（OI變化）
    "vol": ((232, 245, 233), (27, 94, 32)),      # 綠（成交額）
}

# ---------- HTTP + 快取（多人友善：ttl=120，帶抖動與容錯） ----------
@st.cache_data(ttl=120)
def http_get_json(url, params=None, timeout=12, tries=3, sleep_between=0.6):
    """一般 GET，回傳 json；失敗會 raise。"""
    time.sleep(random.uniform(0.05, 0.15))  # 避免同秒齊發
    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=UA)
            # Workers 在 upstream 非 200 時會回 502 + JSON，我們也一起解析
            if r.status_code != 200:
                # 若是 Worker 的 JSON 錯誤格式，直接丟例外讓上層降級
                raise requests.HTTPError(f"status={r.status_code}, body={r.text[:200]}")
            return r.json()
        except Exception as e:
            last_err = e
            if i == tries - 1:
                raise
            time.sleep(sleep_between + random.uniform(0, 0.25))
    if last_err:
        raise last_err

@st.cache_data(ttl=120)
def fapi_get_json(path, params=None, timeout=12, safe=False):
    """
    走 Worker 的 fapi 端點。
    safe=True：失敗時回 None（不 raise），用於雲端降級。
    """
    url = f"{WORKER_BASE}{path}"
    try:
        return http_get_json(url, params=params, timeout=timeout)
    except Exception:
        if safe:
            return None
        raise

# ---------- 資料來源 ----------
@st.cache_data(ttl=120)
def try_get_usdtm_tickers_df():
    """
    取 24h 成交額（/fapi/v1/ticker/24hr）。
    有時雲端會被擋；失敗回 None，不拋錯。
    """
    try:
        data = fapi_get_json("/fapi/v1/ticker/24hr", params=None, timeout=15, safe=True)
        if not data:
            return None
        rows = []
        for d in data:
            sym = d.get("symbol", "")
            if sym.endswith("USDT"):
                try: qv = float(d.get("quoteVolume", 0.0))
                except Exception: qv = 0.0
                rows.append({"symbol": sym, "quoteVolume24h": qv})
        if not rows:
            return None
        df = pd.DataFrame(rows).sort_values("quoteVolume24h", ascending=False)
        return df
    except Exception:
        return None

@st.cache_data(ttl=120)
def get_usdtm_top_symbols(n=20):
    """
    正常：用 24h 成交額挑 Top N。
    降級：用白名單（最少也能跑 20 檔）。
    """
    df = try_get_usdtm_tickers_df()
    degraded = df is None
    if not degraded:
        syms = df.head(n)["symbol"].tolist()
        qv_map = df.set_index("symbol")["quoteVolume24h"].to_dict()
        return syms, qv_map, degraded
    else:
        syms = FALLBACK_SYMBOLS[:n]
        qv_map = {}  # 沒有 24h 成交額
        return syms, qv_map, degraded

@st.cache_data(ttl=120)
def get_taker_buy_sell(symbol, period="5m", limit=12):
    url = f"{DATA_BASE}/futures/data/takerlongshortRatio"
    arr = http_get_json(url, {"symbol": symbol, "period": period, "limit": limit}, timeout=15)
    if not isinstance(arr, list) or len(arr) == 0: return pd.DataFrame()
    df = pd.DataFrame(arr)
    df["ts"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True)
    for col in ("buyVol", "sellVol", "buySellRatio"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    df["buySellDiff"] = df["buyVol"] - df["sellVol"]
    return df

@st.cache_data(ttl=120)
def get_oi_hist(symbol, period="5m", limit=2):
    url = f"{DATA_BASE}/futures/data/openInterestHist"
    arr = http_get_json(url, {"symbol": symbol, "period": period, "limit": limit}, timeout=15)
    if not isinstance(arr, list) or len(arr) == 0: return pd.DataFrame()
    df = pd.DataFrame(arr)
    df["ts"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="ms", utc=True)
    df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
    return df.dropna(subset=["ts", "sumOpenInterest"]).sort_values("ts")

@st.cache_data(ttl=120)
def get_last_1h_quote_volume(symbol):
    # 用 5m K 線合計近 1 小時報價成交額；safe=True 以便雲端被擋時回 NaN
    arr = fapi_get_json("/fapi/v1/klines", {"symbol": symbol, "interval": "5m", "limit": 12}, timeout=15, safe=True)
    if not isinstance(arr, list) or len(arr) == 0: return np.nan
    try:
        qvols = [float(k[7]) for k in arr]  # quoteAssetVolume
        return float(np.nansum(qvols))
    except Exception:
        return np.nan

# ---------- 摘要（單一標的） ----------
def summarize_symbol(symbol, qv24h_map):
    try:
        taker = get_taker_buy_sell(symbol, "5m", 12)
        oi = get_oi_hist(symbol, "5m", 2)
        qv1h = get_last_1h_quote_volume(symbol)

        if taker.empty:
            return {"symbol": symbol, "error": "no taker data"}

        last = taker.iloc[-1]
        ma = taker[["buyVol", "sellVol", "buySellDiff"]].rolling(6, min_periods=1).mean().iloc[-1]

        oiNow = oi["sumOpenInterest"].iloc[-1] if not oi.empty else np.nan
        oiPrev = oi["sumOpenInterest"].iloc[-2] if len(oi) >= 2 else np.nan
        oiChgPct = (oiNow - oiPrev) / oiPrev * 100.0 if (pd.notna(oiNow) and pd.notna(oiPrev) and oiPrev != 0) else np.nan

        takerSpikeX = float(last["buyVol"]) / float(ma["buyVol"]) if (pd.notna(ma["buyVol"]) and ma["buyVol"] > 0) else np.nan
        qv24h = float(qv24h_map.get(symbol, np.nan)) if qv24h_map else np.nan

        verdict = "中性"
        if pd.notna(last["buySellDiff"]) and pd.notna(oiChgPct):
            if last["buySellDiff"] > 0 and oiChgPct > 0: verdict = "多方新單可能"
            elif last["buySellDiff"] > 0 and oiChgPct < 0: verdict = "可能平空/擠壓"
            elif last["buySellDiff"] < 0 and oiChgPct > 0: verdict = "空方新單可能"
            elif last["buySellDiff"] < 0 and oiChgPct < 0: verdict = "空方走弱/減倉"

        return {
            "symbol": symbol,
            "24h 成交額(USDT)": qv24h,
            "近1小時成交額(USDT)": float(qv1h) if not pd.isna(qv1h) else np.nan,
            "主動買量（5m）": float(last["buyVol"]) if pd.notna(last["buyVol"]) else np.nan,
            "主動賣量（5m）": float(last["sellVol"]) if pd.notna(last["sellVol"]) else np.nan,
            "淨主動買量（5m）": float(last["buySellDiff"]) if pd.notna(last["buySellDiff"]) else np.nan,
            "買賣力道比（買/賣）": float(last["buySellRatio"]) if pd.notna(last["buySellRatio"]) else np.nan,
            "近30分鐘買量均值": float(ma["buyVol"]) if pd.notna(ma["buyVol"]) else np.nan,
            "買量放大量倍數（vs近30m均值）": float(takerSpikeX) if not pd.isna(takerSpikeX) else np.nan,
            "未平倉量 OI（現值）": float(oiNow) if not pd.isna(oiNow) else np.nan,
            "OI 變化%（相對前一根5m）": float(oiChgPct) if not pd.isna(oiChgPct) else np.nan,
            "判讀": verdict,
            "error": None
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

# ---------- UI ----------
st.title("USDTⓈ-M 流動性雷達（5分鐘）")
now_local = datetime.now(tz=TZ_LOCAL).strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"台北時間：{now_local}｜多人友善：所有資料以 120 秒快取合併請求。")

with st.sidebar:
    st.header("設定")
    top_n = st.slider("追蹤檔數（依 24h 成交額 / 或降級白名單）", 20, 50, 20, step=1)
    view_mode = st.radio("顯示模式", ["簡化模式（建議）", "進階模式"], index=0)
    direction_priority = st.selectbox("方向優先", ["全部", "多方新單優先", "空方新單優先"], index=0)
    hide_no_data = st.checkbox("隱藏無資料（no taker data）", value=True)
    auto_refresh = st.checkbox("自動每 5 分鐘更新", value=False)
    if auto_refresh:
        st.markdown("<meta http-equiv='refresh' content='300'>", unsafe_allow_html=True)
    if st.button("立即更新"):
        try: st.rerun()
        except Exception: pass

# 取得 TopN（可能降級）
symbols, qv24h_map, degraded = get_usdtm_top_symbols(top_n)
if degraded:
    st.warning("⚠️ 雲端目前無法取得 24h 成交額端點，已啟用『降級模式』：\n"
               "- 使用預設白名單品種作追蹤清單\n"
               "- 24h 成交額欄位可能為空（NaN）\n"
               "- 近1h 成交額 / 主動買賣 / OI 變化 依然正常，可照統一排序研判熱區與方向", icon="⚠️")

st.write(f"**追蹤標的（{len(symbols)} 檔）**：", ", ".join(symbols))

# 摘要
rows = [summarize_symbol(sym, qv24h_map) for sym in symbols]
df = pd.DataFrame(rows)
err_df = df[df["error"].notna()][["symbol","error"]]
if not err_df.empty:
    st.warning("有些標的抓取失敗（稍後可再試）：")
    st.dataframe(err_df, use_container_width=True, height=180)

ok = df[df["error"].isna()].drop(columns=["error"]).copy()
# 只有在欄位存在時才 dropna，避免 KeyError
if "淨主動買量（5m）" in ok.columns and hide_no_data:
    ok = ok.dropna(subset=["淨主動買量（5m）"], how="any")

# ---------- 統一排序（與本機一致） ----------
def unified_sort(df_in: pd.DataFrame):
    if df_in.empty: return df_in
    # 方向優先區塊
    if ("淨主動買量（5m）" in df_in.columns) and ("OI 變化%（相對前一根5m）" in df_in.columns):
        if direction_priority == "多方新單優先":
            pri = df_in[(df_in["淨主動買量（5m）"] > 0) & (df_in["OI 變化%（相對前一根5m）"] >= 0)]
            rest = df_in.drop(pri.index); blocks = [pri, rest]
        elif direction_priority == "空方新單優先":
            pri = df_in[(df_in["淨主動買量（5m）"] < 0) & (df_in["OI 變化%（相對前一根5m）"] >= 0)]
            rest = df_in.drop(pri.index); blocks = [pri, rest]
        else:
            blocks = [df_in]
    else:
        blocks = [df_in]

    outs = []
    for b in blocks:
        if b.empty: outs.append(b); continue
        b = b.copy()
        if "淨主動買量（5m）" in b.columns:
            b["_absNetBuy"] = b["淨主動買量（5m）"].abs()
        else:
            b["_absNetBuy"] = np.nan
        # 缺欄位也不會炸，NaN 會自動排後
        sort_cols = ["近1小時成交額(USDT)", "買量放大量倍數（vs近30m均值）", "_absNetBuy", "OI 變化%（相對前一根5m）", "24h 成交額(USDT)"]
        for c in sort_cols:
            if c not in b.columns: b[c] = np.nan
        b.sort_values(by=sort_cols, ascending=[False, False, False, False, False], inplace=True, na_position="last")
        b.drop(columns=["_absNetBuy"], inplace=True)
        outs.append(b)
    return pd.concat(outs, axis=0)

ok_sorted = unified_sort(ok)

# ---------- 顯示 ----------
TABLE_HEIGHT = 900
if ok_sorted.empty:
    st.info("目前沒有成功回傳資料的標的。請稍後再試或點『立即更新』。")
else:
    if view_mode.startswith("簡化"):
        cols = ["symbol","24h 成交額(USDT)","近1小時成交額(USDT)","淨主動買量（5m）","OI 變化%（相對前一根5m）","判讀"]
        for c in cols:
            if c not in ok_sorted.columns: ok_sorted[c] = np.nan
        view = ok_sorted[cols].copy()
        def style_simple(df_in: pd.DataFrame):
            styler = df_in.style
            styler = styler.apply(lambda s: make_bg_styles(s, (232,245,233), (27,94,32)), subset=["24h 成交額(USDT)"])
            styler = styler.apply(lambda s: make_bg_styles(s, (232,245,233), (27,94,32)), subset=["近1小時成交額(USDT)"])
            styler = styler.apply(lambda s: make_bg_styles(s, COLORS["netbuy"][0], COLORS["netbuy"][1]), subset=["淨主動買量（5m）"])
            styler = styler.apply(lambda s: make_bg_styles(s, COLORS["oi"][0], COLORS["oi"][1]), subset=["OI 變化%（相對前一根5m）"])
            styler = styler.format({
                "24h 成交額(USDT)": "{:,.0f}",
                "近1小時成交額(USDT)": "{:,.0f}",
                "淨主動買量（5m）": "{:,.2f}",
                "OI 變化%（相對前一根5m）": "{:,.3f}",
            })
            return styler
        st.subheader("📊 簡化模式（統一排序）")
        st.dataframe(style_simple(view), use_container_width=True, height=TABLE_HEIGHT)
    else:
        view = ok_sorted.copy()
        num_cols = [c for c in view.columns if c not in ["symbol","判讀"]]
        view[num_cols] = view[num_cols].apply(pd.to_numeric, errors="coerce")
        def style_full(df_in: pd.DataFrame):
            styler = df_in.style
            styler = styler.apply(lambda s: make_bg_styles(s, (232,245,233), (27,94,32)), subset=["24h 成交額(USDT)"])
            styler = styler.apply(lambda s: make_bg_styles(s, (232,245,233), (27,94,32)), subset=["近1小時成交額(USDT)"])
            if "買量放大量倍數（vs近30m均值）" in df_in.columns:
                styler = styler.apply(lambda s: make_bg_styles(s, COLORS["spike"][0], COLORS["spike"][1]), subset=["買量放大量倍數（vs近30m均值）"])
            styler = styler.apply(lambda s: make_bg_styles(s, COLORS["netbuy"][0], COLORS["netbuy"][1]), subset=["淨主動買量（5m）"])
            styler = styler.apply(lambda s: make_bg_styles(s, COLORS["oi"][0], COLORS["oi"][1]), subset=["OI 變化%（相對前一根5m）"])
            styler = styler.format({
                "24h 成交額(USDT)": "{:,.0f}",
                "近1小時成交額(USDT)": "{:,.0f}",
                "主動買量（5m）": "{:,.2f}",
                "主動賣量（5m）": "{:,.2f}",
                "淨主動買量（5m）": "{:,.2f}",
                "買賣力道比（買/賣）": "{:,.3f}",
                "近30分鐘買量均值": "{:,.2f}",
                "買量放大量倍數（vs近30m均值）": "{:,.3f}",
                "未平倉量 OI（現值）": "{:,.0f}",
                "OI 變化%（相對前一根5m）": "{:,.3f}",
            })
            return styler
        st.subheader("🧪 進階模式（統一排序）")
        st.dataframe(style_full(view), use_container_width=True, height=TABLE_HEIGHT)

st.caption("資料來源：Binance USDTⓈ-M Futures（透過 Cloudflare Worker 代理與快取 120s）。時區：台北。")
