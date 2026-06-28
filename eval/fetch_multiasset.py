#!/usr/bin/env python3
"""
fetch_multiasset.py — E4(B1)多资产同构对取数:为跨市场预训练抓取【股指↔本国10Y国债】日频对。

动机(research/E0_results.md 结论):E0/E0b 证强计量先验作为独立生成器被反驳(C2ST 0.94 vs DiT 0.60),
floor 是【数据稀缺】下界。唯一治本 = 注入【真·独立宏观窗】。本脚本取 US 之外的同构对(股指日 log
收益 + 本国 10Y 日差分,与 SP500/DGS10 同口径),把独立窗 ~29(L512)抬到 ~60-80 → 预训 DiT 学
"跨市场共享的 stylized 律",再用 SP500↔DGS10 微调。

★ 诚实/防泄露铁律(eval/docs 计划):
  - 外部数据【绝不入主 CSV】train_sp500_us10y.csv,单独落 data/multiasset/。
  - per-market z-score(在 dataset 侧),本脚本只存【原始量级 日收益/日差分】。
  - 不取与 SP500 近重复的近因子;选独立经济体(JP/UK/euro)。
  - 微调后须用 memorization 对【原 SP500 bank】单测复制率未升(防跨市场泄露)。

数据源(均已实测可达, 2026-06-28):
  股指(Yahoo chart API, period1=0 全历史):^N225 / ^FTSE / ^FCHI(价格指数, 非TR)。
  10Y 国债(各国官方日频):
    JP  MOF JGB 历史 CSV(1974+, 10Y 列)
    UK  BoE IADB IUDMNZC(10Y 名义零息收益, 日频)
    EU  ECB SDW 欧元区 AAA 基准 10Y(2004+, 日频)
用法: conda run -n ts_diffusion python eval/fetch_multiasset.py
产出: data/multiasset/{jp,uk,eu}.csv(列 date, eq_ret, y10_diff)+ 控制台 stylized/独立性体检。
"""
import io
import json
import os
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

OUT_DIR = "/home/u00134/data/multiasset"
UA = {"User-Agent": "Mozilla/5.0 (research data fetch)"}


def _get(url, timeout=40):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


# ──────────────────────────────── 股指(Yahoo) ────────────────────────────────
def fetch_yahoo_index(symbol):
    """symbol 如 '^N225' → DataFrame(date, close)。period1=0 取全历史日线。"""
    sym = symbol.replace("^", "%5E")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1=0&period2=9999999999&interval=1d")
    d = json.loads(_get(url))
    r = d["chart"]["result"][0]
    ts = r["timestamp"]
    close = r["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s").normalize(), "close": close})
    return df.dropna()


# ──────────────────────────────── 10Y 国债 ────────────────────────────────
def fetch_jp_10y():
    """MOF 历史 JGB CSV → DataFrame(date, y10)。前 2 行表头, Date=YYYY/M/D, 10Y 在第 10 列(idx 10)。"""
    raw = _get("https://www.mof.go.jp/english/policy/jgbs/reference/interest_rate/"
               "historical/jgbcme_all.csv").decode("utf-8", "ignore")
    df = pd.read_csv(io.StringIO(raw), skiprows=1)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], format="%Y/%m/%d", errors="coerce")
    df["y10"] = pd.to_numeric(df["10Y"], errors="coerce")
    return df[["date", "y10"]].dropna()


def fetch_uk_10y():
    """BoE IADB IUDMNZC(10Y 名义零息)→ DataFrame(date, y10)。"""
    url = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?"
           "csv.x=yes&Datefrom=01/Jan/1979&Dateto=now&SeriesCodes=IUDMNZC&CSVF=TT&UsingCodes=Y&VPD=Y&VFD=N")
    raw = _get(url).decode("utf-8", "ignore")
    # 前导有 SERIES,DESCRIPTION + 空行, 真表头是 DATE,IUDMNZC
    lines = raw.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.upper().startswith("DATE,"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df.columns = ["date", "y10"]
    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y", errors="coerce")
    df["y10"] = pd.to_numeric(df["y10"], errors="coerce")
    return df.dropna()


def fetch_eu_10y():
    """ECB SDW 欧元区 AAA 基准 10Y(日频)→ DataFrame(date, y10)。"""
    url = ("https://data-api.ecb.europa.eu/service/data/YC/"
           "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y?format=csvdata")
    raw = _get(url).decode("utf-8", "ignore")
    df = pd.read_csv(io.StringIO(raw))
    df = df.rename(columns={"TIME_PERIOD": "date", "OBS_VALUE": "y10"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["y10"] = pd.to_numeric(df["y10"], errors="coerce")
    return df[["date", "y10"]].dropna()


# ──────────────────────────────── 对齐 + 同口径 ────────────────────────────────
def build_pair(eq_df, y_df, label):
    """股指 close + 10Y yield → 对齐到共同交易日 → (eq 日 log 收益, y10 日差分)。"""
    m = pd.merge(eq_df, y_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    m["eq_ret"] = np.log(m["close"]).diff()
    m["y10_diff"] = m["y10"].diff()
    out = m[["date", "eq_ret", "y10_diff"]].dropna().reset_index(drop=True)
    return out


def diag(df, label, ref_ret=None):
    r = df["eq_ret"].values; y = df["y10_diff"].values
    s = (f"  {label:6} n={len(df):6d}  {df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()}  "
         f"eq[std={r.std():.4f} kurt={kurtosis(r):.1f}]  y10[std={y.std():.4f} kurt={kurtosis(y):.1f}]  "
         f"x-corr={np.corrcoef(r, y)[0,1]:+.3f}")
    return s


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    specs = [("jp", "^N225", fetch_jp_10y), ("uk", "^FTSE", fetch_uk_10y), ("eu", "^FCHI", fetch_eu_10y)]
    built = {}
    for mkt, sym, yfn in specs:
        try:
            print(f"[fetch] {mkt}: equity {sym} + 10Y ...", flush=True)
            eq = fetch_yahoo_index(sym)
            y = yfn()
            pair = build_pair(eq, y, mkt)
            path = os.path.join(OUT_DIR, f"{mkt}.csv")
            pair.to_csv(path, index=False)
            built[mkt] = pair
            print(f"        → {path}  ({len(pair)} 天)")
        except Exception as e:
            print(f"  [!] {mkt} 失败: {type(e).__name__}: {str(e)[:120]}")

    # US 参照(已有主 CSV)
    us = pd.read_csv("/home/u00134/data/train_sp500_us10y.csv")
    print("\n========== per-market stylized + 独立性体检 ==========")
    print(f"  {'US(ref)':6} n={len(us):6d}  eq[std={us['sp500'].std():.4f} kurt={kurtosis(us['sp500'].values):.1f}]"
          f"  y10[std={us['DGS10'].std():.4f} kurt={kurtosis(us['DGS10'].values):.1f}]")
    for mkt, pair in built.items():
        print(diag(pair, mkt))
    # 跨市场 eq 收益相关(看独立性; 全球危机致正相关但 <0.9 即非近重复)
    if built:
        print("\n  跨市场 eq 日收益相关(重叠期, <0.9=非近重复, 有独立信号):")
        keys = list(built.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = built[keys[i]].set_index("date")["eq_ret"]
                b = built[keys[j]].set_index("date")["eq_ret"]
                c = pd.concat([a, b], axis=1, join="inner").corr().iloc[0, 1]
                print(f"    {keys[i]}-{keys[j]}: {c:+.3f}")
    print(f"\n  独立 L=512 窗增量估算: " +
          " + ".join(f"{m}~{len(p)//512}" for m, p in built.items()) +
          f"  (现 US ~{len(us)//512}); 详见 dataset 多源加载。")


if __name__ == "__main__":
    main()
