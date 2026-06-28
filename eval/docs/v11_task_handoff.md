# v11 任务交接 (TASK HANDOFF — 给接手的 Claude CLI 会话)

> 本文件是**自包含、可独立执行**的任务说明。新会话请先读 `~/CLAUDE.md`(项目总览 + 版本日志),再读本文件按步骤执行。
> 环境:`conda activate ts_diffusion`;仓库 `~/src`(git 分支 `deep_v10`,remote `github.com:mengguo311/mzh_diff`);真实数据 `/home/u00134/data/train_sp500_us10y.csv`。
> **长任务(≥40min)必须 `nohup` 后台跑、抗 SSH 断连。**

## 0. 现状(已完成)
- v9(DiT-B 20k)基线 score.py=47.16;诊断发现 **过平滑/欠离散**(非过粗糙)。
- v10 **采样优化**(DDIM `eta=1,steps=200,w=1`,零重训)→ 50.23。
- v10 **重训** `logs/deep_v10_dit_b/`(min-SNR γ=5 + stylized-fact 辅助损失 + clip 5σ→8σ,12000ep,24.4h)→ 路径**最真实**;但 score.py 总分(同一 v9 公平打分)48.58 < 50.23。
- **发现 `eval/score.py` 的 `ddpm_mse`(权重 0.20)自指缺陷**:衡量样本到"模型自身过平滑流形"的距离,惩罚更真实的模型。
- 已建**非自指鉴别器**并验证:`eval/c2st.py`(feature-space C2ST)+ `eval/signature.py`(自实现 depth-3 截断签名 + Sig-MMD)。结论(real-vs-real 标定均通过):
  | | 标定 | v9 | v10采样 | v10重训 |
  |---|---|---|---|---|
  | C2ST 检出率(↓更真) | 0.495 | 0.867 | 0.874 | **0.750** |
  | Sig-MMD p(↑更真) | 0.326 | 0.003 | 0.003 | **0.395** |
  两检验一致判定重训最真实,实证 ddpm_mse 悖论。

## 1. 待办任务(按顺序)

### 步骤 A — 修 `eval/score.py`(零重训,先做)
1. `WEIGHTS` dict(约 line 152):`ddpm_mse` 0.20 → **0.05**,注释标注 "self-referential diagnostic"。释放的 0.15 重分配给 data-vs-data 指标,**确保 WEIGHTS 之和 = 1.0**。建议新权重(可微调,务必归一):
   `ddpm_mse 0.05, sp_acf 0.12, dg_acf 0.12, uncond_corr 0.10, tail_corr 0.10, wasserstein 0.13, sp_skew 0.08, sp_kurt 0.10, dg_skew 0.07, dg_kurt 0.13`(和=1.00,核对!)。
2. 修 `score_full` / self-check 里 `wasserstein=100` 的自评作弊(约 line 714):自评时 wasserstein 也应走真实计算或明确剔除,别给满分。
3. 加 `--forensic` 选项:调用 `eval/c2st.py` 与 `eval/signature.py`,输出**独立的取证 headline**(C2ST acc/AUC + Sig-MMD p),**不混入 fidelity 总分**。
4. **重算真实自评基线**(新权重下旧的 52.47 不可比)并对 v9/v10/v10重训 重打分,确认排序更合理。
> 验证:`conda run -n ts_diffusion python eval/score.py --model dit-b --checkpoint logs/deep_v9_dit_b_20k/checkpoint_final.pt --scaler logs/deep_v9_dit_b_20k/scaler.pt --real /home/u00134/data/train_sp500_us10y.csv --fake output/deep_v10_retrained.csv --json eval/v10_retrained_rescored.json`

### 步骤 B — 实现可微 torch Sig-MMD 辅助损失(`losses.py`)
**用 depth-2 向量化签名(便宜、可微、无 python 循环)**,而非移植 `eval/signature.py` 的 numpy Chen 循环。核心公式(对一批子窗 `path` (B,Lsub,d)):
```python
dX = path[:,1:,:] - path[:,:-1,:]              # (B, n, d)
S1 = dX.sum(1)                                  # (B, d)  level-1
prefix = torch.cumsum(dX, dim=1) - dX           # 排他前缀和 (B, n, d)
S2 = torch.einsum('bti,btj->bij', prefix, dX)   # (B, d, d) level-2
sig = torch.cat([S1, S2.reshape(S1.shape[0], -1)], dim=1)  # (B, d+d^2)  d=3 -> 12 维
```
- 通道处理:`path = stack([sp/scale_sp, dg/scale_dg, time_linspace01], dim=-1)`,`scale_*` = 真实数据全局 std(从 dataset/scaler 或预存常量;DGS10≈0.069, SP500≈0.011)。
- 子窗:每步用 `torch.randint` 取 K 个固定起点(K=1~2,Lsub=128),对 `recover_x0` 得到的 `x0_hat` 与同 batch 的真实 `x0` 各取**相同**子窗,算签名。
- MMD:`sig_mmd2(sig_fake, sig_real)` = 无偏 MMD²,RBF 核,**median-heuristic 带宽用 `.detach()`**(否则带宽随梯度漂)。可微 wrt `x0_hat`。
- 门控:仅对 `alphas_cumprod[t] > AUX_ABAR_MIN` 的可靠子集计算(复用现有逻辑),整体乘 `SIG_MMD_WEIGHT`。
- `config.py` 新增:`USE_SIG_MMD=True, SIG_MMD_WEIGHT=0.05(从小 ramp), SIG_DEPTH=2, SIG_SUB_LEN=128, SIG_N_SUB=2`。
- `train.py`:像现有 `loss_acf/loss_rough` 一样加进 `loss`,并在 epoch 日志加 `sig` 分项。
- **是否保留 stylized acf+rough**:建议验证轮**保留**(便宜),与签名互补;若想分辨贡献做消融再说。
> ⚠️ 可微性铁律:loss 路径里**不得**出现 `.item()` / numpy / `.detach()`(带宽除外) / data-dependent 整数索引破坏梯度。

### 步骤 C — 冰烟验证(2-3 epoch,GPU1 防与他人争用)
`CUDA_VISIBLE_DEVICES=1 conda run -n ts_diffusion python train.py --model dit-b --epochs 2 --run_name _smoke_v11`
确认:loss 有限、`mse/acf/rgh/sig` 分项合理无 NaN;**记录每 epoch 秒数**(决定步骤 D)。跑完删 `logs/_smoke_v11`。

### 步骤 D — 按 12h 预算反推 epoch 数
用户要求**先 ~12h 验证训练**,效果好再上完整训练。
`epochs = floor(43200 / 实测每epoch秒数)`(参考:v9 纯MSE 4.19s,v10含辅助 7.33s;v11 加签名会更慢)。预计 ~**5000–6000 epoch**。
`config.py` 把 `CHECKPOINT_EVERY` 调到 **2000**(中途可用 checkpoint)。cosine `T_max` 会随 epoch 自动压缩 → 短训练也完整退火,loss 收敛合理。

### 步骤 E — `nohup` 后台启动 ~12h 验证训练(抗 SSH 断)
```bash
cd /home/u00134/src
# 先看空闲卡: nvidia-smi
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n ts_diffusion \
  python -u train.py --model dit-b --epochs <步骤D的数> --batch_size 64 --lr 2e-4 \
    --warmup 150 --run_name deep_v11_dit_b_val \
  > logs/run_dit_v11_val.log 2>&1 &
echo "PID=$!"; tail -f logs/run_dit_v11_val.log
```
产物:`logs/deep_v11_dit_b_val/checkpoint_final.pt` + `scaler.pt`。

### 步骤 F — 评估 v11 验证模型(训练完成后)
```bash
cd /home/u00134/src
CKPT=logs/deep_v11_dit_b_val/checkpoint_final.pt; SCAL=logs/deep_v11_dit_b_val/scaler.pt
conda run -n ts_diffusion python generate.py --model dit-b --checkpoint $CKPT --scaler $SCAL \
  --num_samples 5120 --cond_mode dataset --num_inference_steps 200 --guidance_scale 1.0 --eta 1.0 \
  --output output/deep_v11_val.csv          # ~30min, 也可 nohup 后台
# 非自指鉴别器(关键判据) + 诊断 + 重设计后的 score.py
conda run -n ts_diffusion python eval/c2st.py --fakes v11=output/deep_v11_val.csv v10_retrained=output/deep_v10_retrained.csv --json eval/c2st_v11.json
conda run -n ts_diffusion python eval/signature.py --fakes v11=output/deep_v11_val.csv v10_retrained=output/deep_v10_retrained.csv --json eval/signature_v11.json
conda run -n ts_diffusion python eval/diagnostics.py --fake output/deep_v11_val.csv --json eval/diag_v11_val.json --label v11_val
```
**判据(达标=值得上完整训练)**:v11 验证模型应比 v10_retrained **更难被检出** —— C2ST 检出率 < 0.750 且/或 Sig-MMD p > 0.395;且 `diagnostics.py` 的 roughness(`d2_energy` gap)/burst(`max_rolling_vol` gap)继续缩小、wasserstein/峰度不回退。**不要用 ddpm_mse 总分判定。**

### 步骤 G — 达标则启动完整训练
同配方、`--epochs 12000`(或 16000)、`--run_name deep_v11_dit_b`、`CHECKPOINT_EVERY` 恢复(如 4000)、同样 `nohup` 后台。完成后重复步骤 F 评估,并更新 `~/CLAUDE.md` 版本日志 + `eval/docs/`。

## 2. 关键约定与坑
- **绝不用 `ddpm_mse` 总分判定生成质量**(已证自指);用 wasserstein/峰度/`diagnostics.py`/非自指鉴别器。
- 真实数据是**绝对路径** `/home/u00134/data/train_sp500_us10y.csv`(不是 `src/data/`)。
- score.py 的 `ddpm_mse` 跨模型不可比(用同一 v9 模型做固定参考才能比);本任务的判据已改用鉴别器,规避此问题。
- 提交习惯:改完 commit 到 `deep_v10` 分支并 push;commit message 末尾加 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。

## 3. 参考文件
- `~/CLAUDE.md` — 项目总览 + 版本日志
- `eval/docs/improvement_roadmap.md` — 多智能体综合路线图(签名脊梁、金融方法表、评估重设计)
- `eval/docs/v10_phase2_changelog.md` — Phase-2(min-SNR+stylized aux)改动细节
- `eval/docs/v10_improvement_plan.md` — 采样优化(eta)分析
- `eval/c2st.py` / `eval/signature.py` — 已验证的鉴别器(签名可微化的 numpy 参考)
- `losses.py` / `train.py` / `scheduler.py` / `config.py` — 训练/损失/调度/配置

> 备注:本次曾起过一个"设计/验证 workflow"对上述实现做对抗验证,但**本文件已自包含**,无需等待该 workflow。
