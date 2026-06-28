#!/usr/bin/env python3
"""
verdict_v11.py — v11 验证训练【步骤 F 判据】自动汇总。
读取 auto_eval_v11.sh 产出的 json, 与 v10_retrained 基线对比, 给出"是否值得上完整训练"的
初判 (步骤 G 的依据)。**判据只用非自指鉴别器 + 诊断 + wasserstein/峰度, 绝不用 ddpm_mse 总分。**

达标 (PASS) = 以下全部成立:
  1. v11 比 v10_retrained 更难被检出: C2ST 检出 acc < 0.750  或  Sig-MMD p > 0.395 (任一)
  2. roughness 缩小: sp & dg 的 d2_energy |gap| 不大于 v10_retrained (允许 ≤ +10% 容差)
  3. burst/regime 不回退: regime.high_vol_frac |gap| 不大于 v10_retrained (≤ +10% 容差)
  4. wasserstein 单项分不回退, sp/dg 峰度单项分不显著回退 (≤ 2 分容差)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(p):
    fp = os.path.join(HERE, p)
    if not os.path.exists(fp):
        print(f"  [缺失] {p}")
        return None
    with open(fp) as f:
        return json.load(f)


def g(d, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k) if isinstance(d, dict) else None
    return default if d is None else d


def main():
    c2 = load("c2st_v11.json")
    sg = load("signature_v11.json")
    dv = load("diag_v11_val.json")
    dr = load("diag_v10_retrained.json")
    sv = load("v11rw_v11_val.json")
    sr = load("v11rw_v10_retrained.json")

    print("\n" + "=" * 78)
    print("  v11 验证训练 —— 步骤 F 判据汇总 (vs v10_retrained 基线)")
    print("=" * 78)

    checks = []

    # ── 1) 非自指鉴别器 (关键判据) ──
    c2_v11 = g(c2, "detect", "v11", "test_acc")
    c2_v10 = g(c2, "detect", "v10_retrained", "test_acc")
    c2_cal = g(c2, "calibration", "real_vs_real", "test_acc")
    sg_v11 = g(sg, "detect", "v11", "p_value")
    sg_v10 = g(sg, "detect", "v10_retrained", "p_value")
    sg_cal = g(sg, "calibration", "real_vs_real", "p_value")
    print(f"\n  [1] 非自指鉴别器 (标定 real-vs-real: C2ST acc={c2_cal}, Sig-MMD p={sg_cal})")
    print(f"      C2ST   检出 acc:  v11={c2_v11}   v10_retrained={c2_v10}   (↓更真, 目标<0.750)")
    print(f"      Sig-MMD 检出 p :  v11={sg_v11}   v10_retrained={sg_v10}   (↑更真, 目标>0.395)")
    cond_c2 = (c2_v11 is not None and c2_v11 < 0.750)
    cond_sg = (sg_v11 is not None and sg_v11 > 0.395)
    ok1 = cond_c2 or cond_sg
    checks.append(("鉴别器: C2ST<0.750 或 Sig-MMD p>0.395", ok1))

    # ── 2) roughness (d2_energy gap) ──
    print(f"\n  [2] roughness d2_energy |gap| (↓更接近真实)")
    ok2 = True
    for ch in ["roughness_sp", "roughness_dg"]:
        gv = g(dv, ch, "d2_energy", "gap_mean")
        gr = g(dr, ch, "d2_energy", "gap_mean")
        if gv is None or gr is None:
            ok2 = False; print(f"      {ch}: 数据缺失"); continue
        better = abs(gv) <= abs(gr) * 1.10
        ok2 = ok2 and better
        print(f"      {ch}: v11 |{gv:+.2e}|  vs  v10 |{gr:+.2e}|  {'OK' if better else '回退'}")
    checks.append(("roughness d2_energy gap 不回退", ok2))

    # ── 3) burst / regime ──
    bv = g(dv, "regime", "high_vol_frac", "gap_mean")
    br = g(dr, "regime", "high_vol_frac", "gap_mean")
    print(f"\n  [3] regime high_vol_frac |gap| (↓更接近真实)")
    if bv is None or br is None:
        ok3 = False; print("      数据缺失")
    else:
        ok3 = abs(bv) <= abs(br) * 1.10
        print(f"      v11 |{bv:+.2e}|  vs  v10 |{br:+.2e}|  {'OK' if ok3 else '回退'}")
    checks.append(("regime high_vol_frac gap 不回退", ok3))

    # ── 4) wasserstein / 峰度 单项分不回退 (score.py) ──
    print(f"\n  [4] score.py 单项分 (data-vs-data, ↑更好; 总分仅参考不作判据)")
    ok4 = True
    for key in ["wasserstein", "sp_kurt", "dg_kurt"]:
        vv = g(sv, "scores", "batch_components", key)
        vr = g(sr, "scores", "batch_components", key)
        if vv is None or vr is None:
            ok4 = False; print(f"      {key}: 数据缺失"); continue
        ok = vv >= vr - 2.0
        ok4 = ok4 and ok
        print(f"      {key:12s}: v11={vv:6.2f}  vs  v10={vr:6.2f}  {'OK' if ok else '回退'}")
    tot_v = g(sv, "scores", "batch_total"); tot_r = g(sr, "scores", "batch_total")
    print(f"      (参考) 总分: v11={tot_v}  v10_retrained={tot_r}")
    checks.append(("wasserstein/峰度 单项分不回退", ok4))

    # ── 结论 ──
    print("\n" + "-" * 78)
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    overall = all(ok for _, ok in checks)
    print("-" * 78)
    print(f"  >>> 初判: {'★ 达标 (PASS) — 建议上完整训练 (步骤 G)' if overall else '✗ 未达标 (FAIL) — 先调权重/配方, 勿急于上完整训练'}")
    print("=" * 78)
    return overall


if __name__ == "__main__":
    main()
