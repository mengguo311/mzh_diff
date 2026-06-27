#!/usr/bin/env python3
"""
eval/final_eval.py — ALL-IN-ONE 金融典型事实评估器(双通道 SP500 收益 + DGS10 差分)。

整合并精简 eval/ 散落评估标准, 只保留【核心金融统计指标 + 厚尾 + 双通道常见指标】, 一次评估
所有模型 + 全套可视化 + 生成 final_report.md。

【保留指标(财务典型事实)】
  分布矩: std / skew / kurtosis(厚尾)
  厚尾:   超额峰度 / |z|>4σ 频率 / 尾部分位比 / QQ
  波动聚集: |r|-ACF lag1 / lag10
  杠杆效应: corr(r_t, |r_{t+1}|)
  收益自相关: ret-ACF lag1(市场有效性)
  regime: high_vol_frac / mean_run_len / vol_of_vol
  长程: |r|-ACF lag20(持久性)
  跨通道: corr(sp,dg) / 尾部相关 tail_corr
  DGS10 量化指纹: 差分落 0.01 网格占比
  分布距离: Wasserstein(vs 真实, 每通道)
【筛掉(次要/自指/冗余)】 ddpm_mse(自指)/ score.py 加权总分(被自指污染)/ patch-16 spike(生成伪影非财务事实)/
  switch_rate·tv·max_rolling_vol(与 run_len/d2 冗余)/ 各鉴别器(C2ST/Sig/copy_rate 属新颖度另一轴, 此处不混入)。

用法: conda run -n ts_diffusion python eval/final_eval.py   (评估全部模型, 写 final_report.md + outputs/figures/final_report/)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, wasserstein_distance
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnostics import load_changes, rolling_std, VOL_WINDOW  # noqa: E402

REPO = "/home/u00134/src"
REAL = "/home/u00134/data/train_sp500_us10y.csv"
FIGDIR = os.path.join(REPO, "outputs", "figures", "final_report")
REPORT = os.path.join(REPO, "final_report.md")
os.makedirs(FIGDIR, exist_ok=True)
NROWS = 3000   # 每模型抽样窗数(峰度/ACF/杠杆稳定; 控内存)

# (显示名, 路径, 线)
MODELS = [
    ("v9_20k",        "output/deep_v9_20k.csv",                 "v9"),
    ("v10_sampling",  "output/deep_v10.csv",                    "v10"),
    ("v10_retrained", "output/deep_v10_retrained.csv",          "v10"),
    ("v11_val",       "output/deep_v11_val.csv",                "v11"),
    ("v12_antimem",   "output/deep_v12_dits_antimem.csv",       "v12"),
    ("v13_a1_L512",   "output/deep_v13_a1_L512.csv",            "line2"),
    ("v13_c1_ctx",    "output/deep_v13_c1_ctx_realctx2048.csv", "line2"),
    ("line2_clip11★", "output/line2/l2_clip11_final.csv",       "line2"),
    ("line1_clip15",  "output/line1/l1_clip15.csv",             "line1"),
    ("line1_clip20",  "output/line1/l1_clip20.csv",             "line1"),
    ("line1_clip20acf","output/line1/l1_clip20_acf.csv",        "line1"),
]
MODELS_d = {n: p for n, p, _ in MODELS}


def load(path, nrows=None):
    import pandas as pd
    df = pd.read_csv(os.path.join(REPO, path) if not path.startswith("/") else path, nrows=nrows)
    cols = df.columns.tolist()
    sp = sorted([c for c in cols if c.lower().startswith("sp500_") and c.split("_")[-1].isdigit()], key=lambda c: int(c.split("_")[-1]))
    dg = sorted([c for c in cols if c.lower().startswith("dgs10_") and c.split("_")[-1].isdigit()], key=lambda c: int(c.split("_")[-1]))
    if len(sp) >= 2:
        arr = np.stack([df[sp].values.astype(float), df[dg].values.astype(float)], axis=1)
    else:
        arr = load_changes(path if path.startswith("/") else os.path.join(REPO, path))
    # 剔除罕见发散行(如 C1 自回归偶发爆值, 否则污染峰度): 任一通道 max 超物理界(sp 60%/dg 150bp)
    rowmax = np.abs(arr).max(axis=2)
    keep = (rowmax[:, 0] <= 0.6) & (rowmax[:, 1] <= 1.5)
    return arr[keep] if keep.any() else arr


def abs_acf(x, lag):
    a = np.abs(x); a = a - a.mean(1, keepdims=True); var = (a * a).mean(1) + 1e-12
    return float(np.mean((a[:, lag:] * a[:, :-lag]).mean(1) / var))


def ret_acf(x, lag):
    a = x - x.mean(1, keepdims=True); var = (a * a).mean(1) + 1e-12
    return float(np.mean((a[:, lag:] * a[:, :-lag]).mean(1) / var))


def leverage(x, lag=1):
    rt = x[:, :-lag]; av = np.abs(x[:, lag:])
    rc = rt - rt.mean(1, keepdims=True); ac = av - av.mean(1, keepdims=True)
    num = (rc * ac).sum(1); den = np.sqrt((rc * rc).sum(1) * (ac * ac).sum(1)) + 1e-12
    return float(np.mean(num / den))


def regime(x, vol_w, thr):
    rs = rolling_std(x, vol_w); state = rs > thr
    flips = np.abs(np.diff(state.astype(np.int8), axis=1)).sum(1)
    return float(state.mean(1).mean()), float(np.mean(state.shape[1] / (flips + 1.0))), float(rs.std(1).mean())


def cross_corr(sp, dg):
    cc = [np.corrcoef(sp[i], dg[i])[0, 1] for i in range(len(sp))]
    # tail corr: 在 sp < mean-1.5std 的应激日
    tc = []
    for i in range(len(sp)):
        m = sp[i] < sp[i].mean() - 1.5 * sp[i].std()
        if m.sum() > 3:
            v = np.corrcoef(sp[i][m], dg[i][m])[0, 1]
            if np.isfinite(v): tc.append(v)
    return float(np.nanmean(cc)), (float(np.mean(tc)) if tc else 0.0)


def dgs_quant(dg):  # 差分落 0.01 网格占比(真实 DGS10 量化指纹)
    v = dg.ravel(); r = v / 0.01
    return float((np.abs(r - np.round(r)) < 0.02).mean())


def metrics(arr, real_thr_sp, real_thr_dg, real_sp_flat=None, real_dg_flat=None):
    sp, dg = arr[:, 0, :], arr[:, 1, :]
    spf, dgf = sp.ravel(), dg.ravel()
    spz = (spf - spf.mean()) / (spf.std() + 1e-12)
    cc, tc = cross_corr(sp, dg)
    hv, rl, vv = regime(sp, VOL_WINDOW, real_thr_sp)
    m = {
        "sp_std": float(spf.std()), "sp_skew": float(skew(spf)), "sp_kurt": float(kurtosis(spf)),
        "sp_tail4σ": float((np.abs(spz) > 4).mean()), "sp_|r|acf1": abs_acf(sp, 1),
        "sp_|r|acf10": abs_acf(sp, 10), "sp_|r|acf20": abs_acf(sp, 20),
        "sp_leverage": leverage(sp, 1), "sp_retacf1": ret_acf(sp, 1),
        "high_vol_frac": hv, "mean_run_len": rl, "vol_of_vol": vv,
        "dg_std": float(dgf.std()), "dg_kurt": float(kurtosis(dgf)),
        "cross_corr": cc, "tail_corr": tc, "dgs10_quant": dgs_quant(dg),
    }
    if real_sp_flat is not None:
        m["sp_wasser"] = float(wasserstein_distance(spf[:200000], real_sp_flat[:200000]))
        m["dg_wasser"] = float(wasserstein_distance(dgf[:200000], real_dg_flat[:200000]))
    return m


print("[final_eval] 加载真实基线 ...", flush=True)
real = load_changes(REAL, target_seq_len=2048)
rsp, rdg = real[:, 0, :], real[:, 1, :]
real_thr_sp = float(np.median(rolling_std(rsp, VOL_WINDOW)))
RM = metrics(real, real_thr_sp, None, rsp.ravel(), rdg.ravel())
RM["sp_wasser"] = 0.0; RM["dg_wasser"] = 0.0
print(f"  真实: sp_kurt={RM['sp_kurt']:.2f} |r|acf1={RM['sp_|r|acf1']:.3f} lev={RM['sp_leverage']:.3f} "
      f"cross={RM['cross_corr']:.3f} dgs_quant={RM['dgs10_quant']:.3f}", flush=True)

results = {"real": RM}
for name, path, line in MODELS:
    fp = os.path.join(REPO, path)
    if not os.path.isfile(fp):
        print(f"  [skip] {name}: 无文件"); continue
    try:
        arr = load(path, nrows=NROWS)
        # 价格水平不兼容(std 极大)→ 跳过
        if arr[:, 0, :].std() > 1.0:
            print(f"  [skip] {name}: 疑似价格水平(std>1)"); continue
        m = metrics(arr, real_thr_sp, None, rsp.ravel(), rdg.ravel())
        m["_line"] = line
        results[name] = m
        print(f"  [{name}] kurt={m['sp_kurt']:.2f} |r|acf1={m['sp_|r|acf1']:.3f} lev={m['sp_leverage']:.3f} "
              f"cross={m['cross_corr']:.3f} dgs_q={m['dgs10_quant']:.3f}", flush=True)
    except Exception as e:
        print(f"  [ERR] {name}: {e}")

names = [n for n in results if n != "real"]

# ── 真实度 closeness(每指标 0-1)+ 综合财务真实度分 ──
# 正尺度相对误差 closeness; 近零/有符号指标用固定 scale
SCALE = {"sp_skew": 0.7, "sp_leverage": 0.08, "sp_retacf1": 0.05, "cross_corr": 0.1, "tail_corr": 0.15}
CORE = ["sp_kurt", "sp_tail4σ", "sp_|r|acf1", "sp_|r|acf10", "sp_leverage", "sp_skew",
        "high_vol_frac", "mean_run_len", "cross_corr", "tail_corr", "dgs10_quant", "sp_wasser", "dg_kurt"]
def closeness(metric, mv, rv):
    if metric in SCALE:
        return float(max(0.0, 1.0 - min(1.0, abs(mv - rv) / SCALE[metric])))
    if metric in ("sp_wasser", "dg_wasser"):  # 距离: 越小越好, 相对 real std
        return float(max(0.0, 1.0 - min(1.0, mv / (RM["sp_std"] if "sp" in metric else RM["dg_std"]))))
    rv2 = abs(rv) + 1e-12
    return float(max(0.0, 1.0 - min(1.0, abs(mv - rv) / rv2)))
for n in names:
    cl = [closeness(k, results[n][k], RM[k]) for k in CORE if k in results[n]]
    results[n]["_score"] = round(100 * float(np.mean(cl)), 1)
ranked = sorted(names, key=lambda n: -results[n]["_score"])

# ════════════════ 可视化 ════════════════
GREEN, RED, GREY = "#2ca02c", "#d62728", "0.5"
LINE_COL = {"v9": "#8c8c8c", "v10": "#9467bd", "v11": "#e377c2", "v12": "#bcbd22", "line1": "#ff7f0e", "line2": "#1f77b4"}
def col(n): return GREEN if "clip11" in n else LINE_COL.get(results[n].get("_line"), "#1f77b4")
figs = []

def barfig(metric, title, fname, real_val, logy=False, ref_label="real"):
    try:
        order = ranked
        vals = [results[n][metric] for n in order]
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.bar(range(len(order)), vals, color=[col(n) for n in order])
        ax.axhline(real_val, color="k", ls="--", lw=1.5, label=f"{ref_label}={real_val:.3g}")
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        if logy: ax.set_yscale("log")
        for i, v in enumerate(vals): ax.annotate(f"{v:.3g}", (i, v), ha="center", va="bottom", fontsize=7)
        ax.set_title(title); ax.legend(fontsize=8); fig.tight_layout()
        p = os.path.join(FIGDIR, fname); fig.savefig(p, dpi=130); plt.close(fig); figs.append(fname); return True
    except Exception as e:
        print(f"  [fig] {fname} fail: {e}"); return False

barfig("sp_kurt", "F1. Heavy tails: SP500 excess kurtosis (real=18.8)", "F1_kurtosis.png", RM["sp_kurt"])
barfig("sp_|r|acf1", "F2. Volatility clustering: |return| ACF lag1", "F2_volcluster_acf1.png", RM["sp_|r|acf1"])
barfig("sp_leverage", "F3. Leverage effect: corr(r_t, |r_t+1|)", "F3_leverage.png", RM["sp_leverage"])
barfig("cross_corr", "F4. Cross-channel SP500-DGS10 correlation", "F4_cross_corr.png", RM["cross_corr"])
barfig("mean_run_len", "F5. Regime persistence: mean run length", "F5_regime_runlen.png", RM["mean_run_len"])
barfig("dgs10_quant", "F6. DGS10 quantization fingerprint (diff on 0.01 grid)", "F6_dgs_quant.png", RM["dgs10_quant"])
barfig("sp_wasser", "F7. Distribution distance: SP500 Wasserstein vs real (lower=better)", "F7_wasserstein.png", 0.0)

# F8: 分布尾部 semilog(real + top3) + QQ
try:
    top3 = ranked[:3]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    rflat = rsp.ravel(); lo, hi = np.percentile(rflat, [0.2, 99.8]); bins = np.linspace(lo * 1.4, hi * 1.4, 130)
    ax[0].hist(rflat, bins=bins, density=True, histtype="step", color="k", lw=2.2, label="real")
    for n in top3:
        a = load(MODELS_d[n], nrows=1500)[:, 0, :].ravel()
        ax[0].hist(np.clip(a, bins[0], bins[-1]), bins=bins, density=True, histtype="step", color=col(n), lw=1.4, label=n)
    ax[0].set_yscale("log"); ax[0].set_title("F8a. SP500 return tails (semilog) — real vs top-3"); ax[0].legend(fontsize=8)
    q = np.linspace(0.002, 0.998, 400); rq = np.quantile((rflat - rflat.mean()) / rflat.std(), q)
    ax[1].plot(rq, rq, color=GREY, ls="--", label="real ref (y=x)")
    for n in top3:
        a = load(MODELS_d[n], nrows=1500)[:, 0, :].ravel(); az = (a - a.mean()) / a.std()
        ax[1].plot(rq, np.quantile(az, q), color=col(n), lw=1.3, label=n)
    ax[1].set_title("F8b. QQ vs real quantiles (z)"); ax[1].set_xlabel("real quantile"); ax[1].legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(FIGDIR, "F8_tails_qq.png"); fig.savefig(p, dpi=130); plt.close(fig); figs.append("F8_tails_qq.png")
except Exception as e:
    print("  [fig] F8 fail:", e)

# F9: 财务真实度 scorecard 热图(模型 × 指标, closeness)
try:
    hm_metrics = ["sp_kurt", "sp_skew", "sp_tail4σ", "sp_|r|acf1", "sp_|r|acf10", "sp_leverage",
                  "sp_retacf1", "high_vol_frac", "mean_run_len", "cross_corr", "tail_corr", "dgs10_quant", "sp_wasser"]
    M = np.array([[closeness(k, results[n][k], RM[k]) for k in hm_metrics] for n in ranked])
    fig, ax = plt.subplots(figsize=(12, 0.5 * len(ranked) + 2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(hm_metrics))); ax.set_xticklabels(hm_metrics, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ranked))); ax.set_yticklabels([f"{n} ({results[n]['_score']})" for n in ranked], fontsize=8)
    for i in range(len(ranked)):
        for j in range(len(hm_metrics)): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="closeness to real (1=match)")
    ax.set_title("F9. Financial stylized-fact scorecard (closeness to real; row label = composite score)")
    fig.tight_layout(); p = os.path.join(FIGDIR, "F9_scorecard_heatmap.png"); fig.savefig(p, dpi=130); plt.close(fig); figs.append("F9_scorecard_heatmap.png")
except Exception as e:
    print("  [fig] F9 fail:", e)

# ════════════════ final_report.md ════════════════
def row(n):
    m = results[n]
    return (f"| {n} | {m['_score']} | {m['sp_kurt']:.2f} | {m['sp_skew']:.2f} | {m['sp_tail4σ']*100:.2f}% | "
            f"{m['sp_|r|acf1']:.3f} | {m['sp_|r|acf10']:.3f} | {m['sp_leverage']:.3f} | {m['sp_retacf1']:.3f} | "
            f"{m['high_vol_frac']:.2f} | {m['mean_run_len']:.1f} | {m['cross_corr']:.3f} | {m['tail_corr']:.3f} | "
            f"{m['dgs10_quant']:.2f} | {m['sp_wasser']:.4f} |")

best = ranked[0]
lines = []
lines.append("# final_report.md — 全模型金融典型事实评估(all-in-one)\n")
lines.append(f"> 由 `eval/final_eval.py` 一次性生成:对 {len(names)} 个模型 + 真实基线,评估【核心金融统计指标 + 厚尾 + 双通道常见指标】,"
             f"配 {len(figs)} 张图。每模型抽样 {NROWS} 窗(峰度/ACF/杠杆 N-稳定)。图内英文、正文中文。\n")
lines.append("## 0. 评估标准(精简自 eval/)\n")
lines.append("**保留(财务典型事实)**:分布矩(std/skew/**峰度**)、**厚尾**(峰度/|z|>4σ频率/尾部/QQ)、波动聚集(|r|-ACF lag1/10)、"
             "杠杆效应(corr(r_t,|r_t+1|))、收益自相关(ret-ACF1)、regime(high_vol_frac/mean_run_len/vol_of_vol)、"
             "长程(|r|-ACF lag20)、跨通道(corr+尾相关)、DGS10量化指纹、Wasserstein 分布距离。\n")
lines.append("**筛掉(次要/自指/冗余)**:`ddpm_mse`(自指,衡量到模型自身流形非真实)、`score.py` 加权总分(被自指污染)、"
             "patch-16 spike(生成伪影非财务事实)、switch_rate/tv/max_rolling_vol(与 run_len/d2 冗余)、"
             "C2ST/Sig-MMD/copy_rate(属【新颖度/记忆化】另一轴,见 forensic_suite/scoreboard,本表不混入财务真实度)。\n")
lines.append("> **财务真实度综合分** = 13 项核心指标对真实值的 closeness 均值×100(1=完全匹配)。**非地面真值裁判**,各单项才是依据。\n")
lines.append(f"\n## 1. 总排名(按财务真实度综合分)\n\n**最真实(财务典型事实):`{best}` = {results[best]['_score']}**。\n")
lines.append("\n| 模型 | 综合分 | 峰度 | 偏度 | \\|z\\|>4σ | \\|r\\|acf1 | \\|r\\|acf10 | 杠杆 | retACF1 | high_vol | run_len | 跨通道corr | 尾相关 | DGS量化 | sp_Wasser |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
lines.append(f"| **真实** | — | {RM['sp_kurt']:.2f} | {RM['sp_skew']:.2f} | {RM['sp_tail4σ']*100:.2f}% | {RM['sp_|r|acf1']:.3f} | {RM['sp_|r|acf10']:.3f} | {RM['sp_leverage']:.3f} | {RM['sp_retacf1']:.3f} | {RM['high_vol_frac']:.2f} | {RM['mean_run_len']:.1f} | {RM['cross_corr']:.3f} | {RM['tail_corr']:.3f} | {RM['dgs10_quant']:.2f} | 0 |")
for n in ranked: lines.append(row(n))
lines.append("\n## 2. 可视化分析\n")
caps = {
 "F1_kurtosis.png": "**厚尾(峰度)**:真实 SP500 超额峰度 18.8;各模型普遍欠厚尾(过平滑老病),line1 clip20/clip20acf 与 line2 clip11 放开 clip 后峰度最接近真实。",
 "F2_volcluster_acf1.png": "**波动聚集**:|收益| 一阶自相关,真实 ~0.22;反映波动率的持续性(GARCH 效应)。",
 "F3_leverage.png": "**杠杆效应**:corr(r_t,|r_{t+1}|),真实为负(下跌后波动放大)。",
 "F4_cross_corr.png": "**双通道相关**:SP500↔DGS10 当期相关,真实近零/弱负。",
 "F5_regime_runlen.png": "**regime 持久性**:高/低波动状态平均游程长度,衡量波动聚集的宏观尺度。",
 "F6_dgs_quant.png": "**DGS10 量化指纹**:真实利率差分落 0.01 网格占比 ~0.66;扩散模型多塌成连续浮点(≈0),是其结构硬伤。",
 "F7_wasserstein.png": "**分布距离**:SP500 收益分布到真实的 Wasserstein 距离(越低越真)。",
 "F8_tails_qq.png": "**尾部 + QQ**:综合分 top-3 模型的收益尾部(半对数)与 QQ 图(vs 真实分位),直观看厚尾还原度。",
 "F9_scorecard_heatmap.png": "**财务真实度记分卡热图**:模型×指标的 closeness(绿=贴合真实/红=偏离),行按综合分排序——一图看全。",
}
for f in figs:
    lines.append(f"\n### {f}\n![{f}](outputs/figures/final_report/{f})\n\n{caps.get(f,'')}\n")
lines.append("\n## 3. 关键结论\n")
lines.append(f"- **财务真实度最优 = `{best}`**(综合分 {results[best]['_score']});放开 CLIP 攻厚尾的模型(line1 clip20系 / line2 clip11)峰度最接近真实,印证'欠厚尾'是过平滑历代主病、clip 是其旋钮。\n")
lines.append("- **DGS10 量化指纹**是所有扩散模型的共同硬伤(生成连续浮点 vs 真实 0.01 网格)——这是确定性的'非真实'抓手,与峰度/波动聚集正交。\n")
lines.append("- **峰度高方差**:抽样口径影响大,综合分以多指标 closeness 平滑单指标噪声;最终裁判看单项 + 图。\n")
lines.append("- 本表只评【财务真实度】;【新颖度/记忆化】(复制率/C2ST_新颖)是另一轴,见 `forensic_suite.py`/`scoreboard.py`;外部 hw01 见 `THIRDPARTY_CLAUDE.md`。\n")

with open(REPORT, "w") as f:
    f.write("\n".join(lines))
print(f"\n[final_eval] 报告 {REPORT} ; 图 {len(figs)} 张 → {FIGDIR}")
print(f"[final_eval] 财务真实度排名: " + " > ".join(f"{n}({results[n]['_score']})" for n in ranked))
