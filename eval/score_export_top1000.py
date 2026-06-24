#!/usr/bin/env python3
"""
eval/score_export_top1000.py — 用本项目评分系统给 output/ 各 CSV 评分, 导出每个文件【综合质量前1000条】。

按用户要求: 每个文件独立评分, 分别给出前 1000 条, 排序依据 = 综合质量(逼真 + 新颖 + 有效)。

逐样本综合质量分(0-100, 与 forensic_suite 评价标准一致):
  - 逼真度 realism = exp(-d_M / d_M_real_ref): 样本 24 维 stylized 特征到【真实特征流形】的
    Mahalanobis 距离(z 空间 + ridge 协方差), 以真实窗自身中位距离为尺度归一; 越接近真实分布越高。
  - 新颖度 novelty = clip(1 - max(raw_pearson_to_NN, 0), 0, 1): 与特征最近邻真实训练窗的原序列
    Pearson; 越不像逐点拷贝越高(>0.95 近拷贝 → novelty<0.05)。直接复用 memorization 量具。
  - 有效性 valid = 逐行发散门(任一通道 max|·| > max(0.5, 30×真实std) → 发散 → 排除)。
  - composite = 100 * (0.55*realism + 0.45*novelty); 发散样本 composite=-1 不参与。
  ★ 既奖励逼真又惩罚拷贝 → 避开"记忆化污染"(最像真的往往是抄训练窗)。

文件级评分另用 forensic_suite.run_forensic(集合级: 复制率/C2ST_新颖/SigP_新颖/综合), 写进汇总。

输出到 /home/u00134/data/generation/mguo/:
  <model>_top1000.csv         前1000条数据(原始宽表格式 sp500_i/dgs10_i, 可直接用)
  <model>_top1000_scores.csv  对应每条的 orig_row/composite/realism/novelty/raw_pearson
  _SUMMARY.{txt,json}         每个文件的文件级评分 + top1000 综合分统计 + 排名
"""
import os
import sys
import glob
import json
import time

import numpy as np
import pandas as pd

REPO = "/home/u00134/src"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval"))
import forensic_suite as F          # noqa: E402
import diagnostics as D             # noqa: E402
import c2st as C                    # noqa: E402
import memorization as M            # noqa: E402

REAL = F.REAL_DEFAULT
OUTDIR = "/home/u00134/data/generation/mguo"
os.makedirs(OUTDIR, exist_ok=True)
TOPK = 1000
W_REAL, W_NOVEL = 0.55, 0.45


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def per_sample_scores(fake, real, vol_thr, mu, sd, Xreal, Rz):
    """返回 dict: realism, novelty, raw_pearson, diverged(bool), composite (N,)。"""
    N = len(fake)
    # 有效性: 逐行发散门
    bound = np.array([max(0.5, 30.0 * float(real[:, 0, :].std())),
                      max(0.5, 30.0 * float(real[:, 1, :].std()))])
    rowmax = np.abs(fake).max(axis=2)
    diverged = (rowmax[:, 0] > bound[0]) | (rowmax[:, 1] > bound[1])

    # 逼真度: Mahalanobis(z 空间 + ridge)到真实特征流形
    Ff = C.featurize(fake, vol_thr)
    Fz = (Ff - mu) / sd
    Rz_full = (Xreal - mu) / sd
    cov = np.cov(Rz_full.T) + 1e-3 * np.eye(Rz_full.shape[1])
    inv = np.linalg.inv(cov)
    def maha(Z):
        d = Z  # 已减 mu(z空间均值≈0)
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, inv, d), 0.0))
    dM_fake = maha(Fz)
    dM_real_ref = float(np.median(maha(Rz_full))) + 1e-9
    realism = np.exp(-dM_fake / dM_real_ref)

    # 新颖度: 与特征最近邻真实窗的原序列 Pearson
    dmin, nn_idx = M.nn_to_bank(Fz, Rz)
    raw_p = np.array([np.mean([M._pearson(fake[j, ch], real[nn_idx[j], ch]) for ch in range(2)])
                      for j in range(N)])
    novelty = np.clip(1.0 - np.clip(raw_p, 0.0, 1.0), 0.0, 1.0)

    composite = 100.0 * (W_REAL * realism + W_NOVEL * novelty)
    composite = np.where(diverged, -1.0, composite)
    return {"realism": realism, "novelty": novelty, "raw_pearson": raw_p,
            "diverged": diverged, "composite": composite, "dM": dM_fake}


def export_one(csv_path):
    name = os.path.splitext(os.path.basename(csv_path))[0]
    fake = D.load_changes(csv_path)                      # (N,2,L) 行序=原CSV行序
    L = int(fake.shape[-1]); N = int(len(fake))
    sp_std = float(fake[:, 0, :].std())
    if sp_std > 1.0:
        return {"file": name, "status": "skipped(疑似价格水平/不兼容收益口径)", "L": L, "n": N, "sp_std": sp_std}
    fake = np.nan_to_num(fake, nan=0.0, posinf=0.0, neginf=0.0)
    real = D.load_changes(REAL, target_seq_len=L)
    vol_thr = float(np.median(D.rolling_std(real[:, 0, :], D.VOL_WINDOW)))
    Xreal = C.featurize(real, vol_thr)
    mu, sd = Xreal.mean(0), Xreal.std(0) + 1e-12
    Rz = M._std(Xreal, mu, sd)

    ps = per_sample_scores(fake, real, vol_thr, mu, sd, Xreal, Rz)
    valid = ~ps["diverged"]; n_div = int(ps["diverged"].sum())
    order = np.argsort(-ps["composite"])
    top = [i for i in order if valid[i]][:TOPK]

    # 写数据(原始宽表格式, 可直接用)
    cols = [f"sp500_{i}" for i in range(L)] + [f"dgs10_{i}" for i in range(L)]
    rows = np.stack([np.concatenate([fake[i, 0, :], fake[i, 1, :]]) for i in top])
    pd.DataFrame(rows, columns=cols).to_csv(os.path.join(OUTDIR, f"{name}_top1000.csv"), index=False)
    # 写分数 sidecar
    pd.DataFrame({"orig_row": top, "rank": np.arange(1, len(top) + 1),
                  "composite": ps["composite"][top], "realism": ps["realism"][top],
                  "novelty": ps["novelty"][top], "raw_pearson": ps["raw_pearson"][top]}
                 ).to_csv(os.path.join(OUTDIR, f"{name}_top1000_scores.csv"), index=False)

    # 文件级 forensic(集合级权威评分)
    try:
        rep, _ = F.run_forensic(csv_path, REAL, label=name)
        v = rep["verdict"]
        file_score = {"composite_score": v["composite_score"],
                      "copy_rate": rep["memorization"]["copy_rate_p95"],
                      "c2st_novel": rep["novelty_rerank"]["c2st_novel"],
                      "sig_p_novel": rep["novelty_rerank"]["sig_p_novel"],
                      "kurt": rep["stylized_facts"]["sp500"]["fake"]["kurt"],
                      "verdict": [v["memorization"], v["novelty_c2st"], v["novelty_signature"]]}
    except Exception as e:
        file_score = {"error": str(e)}

    topc = ps["composite"][top]
    return {"file": name, "status": "ok", "L": L, "n": N, "n_div": n_div, "n_valid": int(valid.sum()),
            "n_exported": len(top),
            "top1000_composite_mean": float(np.mean(topc)) if len(topc) else None,
            "top1000_composite_min": float(np.min(topc)) if len(topc) else None,
            "top1000_realism_mean": float(np.mean(ps["realism"][top])),
            "top1000_novelty_mean": float(np.mean(ps["novelty"][top])),
            "all_valid_composite_mean": float(np.mean(ps["composite"][valid])) if valid.any() else None,
            "file_level": file_score}


def main():
    files = sorted(glob.glob(os.path.join(REPO, "output", "*.csv")))
    files = [f for f in files if "_archive" not in f]
    log(f"待评分文件 {len(files)} 个 → 导出到 {OUTDIR}")
    summary = []
    for f in files:
        name = os.path.basename(f)
        try:
            t0 = time.time()
            r = export_one(f)
            summary.append(r)
            if r["status"] == "ok":
                fl = r["file_level"]
                log(f"[OK] {name} ({time.time()-t0:.0f}s): top1000 综合分均值={r['top1000_composite_mean']:.1f} "
                    f"(逼真{r['top1000_realism_mean']:.2f}/新颖{r['top1000_novelty_mean']:.2f}) | "
                    f"文件级: 综合={fl.get('composite_score')} 复制率={fl.get('copy_rate')} "
                    f"C2ST_新颖={fl.get('c2st_novel')} | n_div={r['n_div']}")
            else:
                log(f"[SKIP] {name}: {r['status']}")
        except Exception as e:
            log(f"[ERR] {name}: {e}")
            summary.append({"file": name, "status": f"error: {e}"})
        json.dump(summary, open(os.path.join(OUTDIR, "_SUMMARY.json"), "w"), indent=2, ensure_ascii=False)

    # 文本汇总(按 top1000 综合分均值排名)
    ok = [s for s in summary if s.get("status") == "ok"]
    ok.sort(key=lambda s: -(s["top1000_composite_mean"] or -1))
    with open(os.path.join(OUTDIR, "_SUMMARY.txt"), "w") as fh:
        fh.write("output/ 各文件评分 + top1000 导出 汇总\n" + "=" * 78 + "\n")
        fh.write(f"逐样本综合质量 = 0.55*逼真(Mahalanobis到真实流形) + 0.45*新颖(1-拷贝相关); 发散样本剔除。\n")
        fh.write(f"文件级 = forensic_suite 集合级评分。导出目录 {OUTDIR}\n\n")
        fh.write(f"{'model':<34}{'top1k综合':>9}{'逼真':>6}{'新颖':>6}{'文件级综合':>9}{'复制率':>8}{'C2ST新颖':>9}{'n_div':>6}\n")
        fh.write("-" * 90 + "\n")
        for s in ok:
            fl = s["file_level"]
            cs = fl.get("composite_score"); cs = f"{cs:.1f}" if isinstance(cs, (int, float)) else "N/A"
            cr = fl.get("copy_rate"); cr = f"{cr*100:.1f}%" if isinstance(cr, (int, float)) else "?"
            cn = fl.get("c2st_novel"); cn = f"{cn:.3f}" if isinstance(cn, (int, float)) else "?"
            fh.write(f"{s['file']:<34}{s['top1000_composite_mean']:>9.1f}{s['top1000_realism_mean']:>6.2f}"
                     f"{s['top1000_novelty_mean']:>6.2f}{cs:>9}{cr:>8}{cn:>9}{s['n_div']:>6}\n")
        skipped = [s for s in summary if s.get("status", "").startswith("skip") or "error" in s.get("status", "")]
        if skipped:
            fh.write("\n跳过/失败:\n")
            for s in skipped:
                fh.write(f"  {s['file']}: {s['status']}\n")
    log(f"完成。汇总 {OUTDIR}/_SUMMARY.txt ; 导出 {len(ok)} 个模型 × (top1000 数据 + 分数)")


if __name__ == "__main__":
    main()
