#!/usr/bin/env python3
"""
eval/fetch_fred_treasury.py — 拉取美债相关 FRED 序列, 对齐项目交易日, 试跑取数。

序列(均日频, FRED, 按 basis point 0.01 量化):
  DGS2   = 2Y 国债名义收益率(短端 → 配 DGS10 构成 2s10s 斜率, 衰退指标)
  DGS30  = 30Y 国债名义收益率(长端; ⚠️ 2002-2006 停发有缺口)
  DFII10 = 10Y TIPS 实际收益率(2003 起)
  T10YIE = 10Y 盈亏平衡通胀 = DGS10 - DFII10(2003 起, 通胀预期, 与名义部分正交)

取数: 优先 fredapi(需 FRED_API_KEY); 否则走 FRED 公共 CSV 端点(免 key, 仅需 requests)。
对齐: 以项目 train_sp500_us10y.csv 的交易日为基准, ffill 补节假日; 报各序列覆盖范围(关键:
  DFII10/T10YIE 仅 2003 起 → 若作必需通道会把训练窗砍到 2003+ = 更少独立窗, 故只宜作辅助/子模型)。
输出: /home/u00134/data/fred_treasury.csv(独立文件, 不入主 CSV)。
"""
import io
import os
import sys

import numpy as np
import pandas as pd
import requests

SERIES = ["DGS2", "DGS30", "DFII10", "T10YIE"]
PROJ_REAL = "/home/u00134/data/train_sp500_us10y.csv"
OUT = "/home/u00134/data/fred_treasury.csv"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def fetch_fred(series_id, api_key=None):
    """返回 pd.Series(index=日期, value=float)。优先 fredapi, 否则公共 CSV 端点(免 key)。"""
    if api_key:
        try:
            from fredapi import Fred
            return Fred(api_key=api_key).get_series(series_id)
        except Exception as e:
            print(f"  [fredapi 失败, 回退公共CSV] {e}")
    r = requests.get(FRED_CSV.format(series_id), timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "value"})  # 按位置取
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
    return df.dropna(subset=["date"]).set_index("date")["value"]


def main():
    key = os.environ.get("FRED_API_KEY")
    print(f"[fred] 取数方式: {'fredapi(有key)' if key else 'FRED 公共CSV端点(免key)'}")

    # 项目基准交易日
    proj = pd.read_csv(PROJ_REAL, index_col=0, parse_dates=True)
    proj_dates = proj.index
    print(f"[fred] 项目基准: {PROJ_REAL} {proj_dates.min().date()}~{proj_dates.max().date()} ({len(proj_dates)} 交易日)\n")

    raw = {}
    print(f"{'序列':8s} {'起始':>11s} {'结束':>11s} {'有效点':>7s} {'与项目重叠':>9s} {'量化@0.01':>9s}")
    print("-" * 64)
    for sid in SERIES:
        try:
            s = fetch_fred(sid, key).dropna()
            raw[sid] = s
            ov = s.index.intersection(proj_dates)
            on_grid = float(np.mean(np.abs(s.values * 100 - np.round(s.values * 100)) < 1e-6))
            print(f"{sid:8s} {str(s.index.min().date()):>11s} {str(s.index.max().date()):>11s} "
                  f"{len(s):>7d} {len(ov):>9d} {on_grid:>9.3f}")
        except Exception as e:
            print(f"{sid:8s} 取数失败: {e}")

    if not raw:
        print("\n[fred] 全部取数失败(网络?)。脚本就绪, 有网/有key 时可重跑。")
        return

    # 对齐到项目交易日 + 日差分(与现有 DGS10 通道同口径)
    aligned = pd.DataFrame(index=proj_dates)
    for sid, s in raw.items():
        lvl = s.reindex(proj_dates).ffill()              # 节假日 ffill
        aligned[sid] = lvl
        aligned[f"{sid}_diff"] = lvl.diff()
    aligned.to_csv(OUT)
    print(f"\n[fred] 已对齐+差分, 存 {OUT} (shape {aligned.shape})")

    # 与现有 DGS10 通道的相关(看是否近重复/会计幻觉)
    if "DGS10" in proj.columns:
        dgs10_diff = proj["DGS10"].reindex(proj_dates)
        print("\n各序列日差分 vs 现有 DGS10 通道 相关(>0.9 = 近重复/会计幻觉, 不算独立信号):")
        for sid in raw:
            d = aligned[f"{sid}_diff"]
            both = pd.concat([d, dgs10_diff], axis=1).dropna()
            if len(both) > 30:
                c = float(both.corr().iloc[0, 1])
                tag = "⚠️近重复" if abs(c) > 0.9 else ("○正交性强" if abs(c) < 0.5 else "中等相关")
                print(f"  {sid:8s} corr={c:+.3f}  {tag}")
    print("\n[判读] DFII10/T10YIE 仅 2003 起 → 作必需通道会把训练窗砍到 ~2003+(更少独立窗);"
          "\n        宜作辅助信号/2003+ 子模型。DGS2/DGS30 覆盖更长(配 DGS10 加曲线 slope/curvature)。")


if __name__ == "__main__":
    main()
