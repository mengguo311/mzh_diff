import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. 命令行参数解析
# ============================================================

parser = argparse.ArgumentParser(
    description="Generate visualizations for S&P500 and 10Y Treasury yield data"
)
parser.add_argument(
    "--csv",
    type=str,
    default="/home/u00134/data/train_sp500_us10y.csv",
    help="Path to the CSV data file (default: /home/u00134/data/train_sp500_us10y.csv)",
)
parser.add_argument(
    "--num-figures",
    type=int,
    default=1,
    choices=range(1, 7),
    help="Number of figures to generate (1-6, default: 1)",
)
parser.add_argument(
    "--paths",
    type=int,
    nargs="+",
    default=None,
    help="Indices of specific paths to plot (e.g. --paths 100 200). If not provided, random paths will be selected.",
)
parser.add_argument(
    "--segment-size",
    type=int,
    default=1000,
    help="Size of segments to divide paths for comparing envelope variations (default: 1000)",
)

args = parser.parse_args()

csv_path = args.csv
num_figures = args.num_figures
output_dir = "outputs/figures"

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. 读取数据
# ============================================================

df_raw = pd.read_csv(csv_path)

# ── 宽表检测 (DDPM 生成格式) ──
# 列名模式: sp500_level_0, ..., sp500_level_127, dgs10_level_0, ..., dgs10_level_127
# 或:        sp500_0, ..., sp500_127, dgs10_0, ..., dgs10_127
wide_format = False
sp_level_cols = sorted(
    [c for c in df_raw.columns if c.startswith("sp500_level_") and c[12:].isdigit()],
    key=lambda c: int(c[12:]),
)
dg_level_cols = sorted(
    [c for c in df_raw.columns if c.startswith("dgs10_level_") and c[12:].isdigit()],
    key=lambda c: int(c[12:]),
)

if len(sp_level_cols) >= 2 and len(dg_level_cols) >= 2:
    wide_format = True
    print(f"[Auto-detected] DDPM wide-table LEVELS format")
    print(f"  {len(df_raw)} paths × {len(sp_level_cols)} SP500 timesteps + {len(dg_level_cols)} DGS10 timesteps")

if wide_format:
    # ── 宽表模式: 选取要绘制的路径 ──
    if args.paths is not None:
        path_indices = [idx for idx in args.paths if 0 <= idx < len(df_raw)]
        if not path_indices:
            print(f"[Warning] All provided paths in --paths are out of bounds (range: [0, {len(df_raw)-1}]). Falling back to random selection.")
            args.paths = None
        else:
            num_paths_to_plot = len(path_indices)
    
    if args.paths is None:
        num_paths_to_plot = min(20, len(df_raw))
        rng = np.random.RandomState(42)
        path_indices = sorted(rng.choice(len(df_raw), size=num_paths_to_plot, replace=False))
        
    seq_len = len(sp_level_cols)

    sp500_all = df_raw[sp_level_cols].values   # (N, seq_len)
    dgs10_all = df_raw[dg_level_cols].values   # (N, seq_len)

    print(f"  Selected paths for plotting: {path_indices}")
    print(f"  SP500 level range: [{sp500_all.min():.2f}, {sp500_all.max():.2f}]")
    print(f"  DGS10 level range: [{dgs10_all.min():.4f}, {dgs10_all.max():.4f}]")

    # ── 绘制模拟路径 ──
    print(f"\nGenerating wide-format figures...\n")

    # 图 1: SP500 价格路径
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    for idx in path_indices:
        ax1.plot(range(seq_len), sp500_all[idx], linewidth=0.8, alpha=0.8,
                 label=f"Path #{idx}")
    ax1.set_xlabel("Trading Day (within window)")
    ax1.set_ylabel("S&P 500 Index Level")
    ax1.set_title(f"DDPM Generated S&P 500 Price Paths ({num_paths_to_plot} samples)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for idx in path_indices:
        ax2.plot(range(seq_len), dgs10_all[idx], linewidth=0.8, alpha=0.8,
                 label=f"Path #{idx}")
    ax2.set_xlabel("Trading Day (within window)")
    ax2.set_ylabel("10Y Treasury Yield Level")
    ax2.set_title(f"DDPM Generated 10Y Yield Paths ({num_paths_to_plot} samples)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "01_raw_time_series.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

    # 图 2: 路径扇形图 (百分位数包络)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    x = np.arange(seq_len)

    for q_lo, q_hi, alpha_val in [(5, 95, 0.15), (10, 90, 0.2), (25, 75, 0.3)]:
        sp_lo = np.percentile(sp500_all, q_lo, axis=0)
        sp_hi = np.percentile(sp500_all, q_hi, axis=0)
        ax1.fill_between(x, sp_lo, sp_hi, alpha=alpha_val, color="navy",
                         label=f"{q_lo}-{q_hi}th pctl" if alpha_val == 0.15 else "")
    sp_med = np.median(sp500_all, axis=0)
    ax1.plot(x, sp_med, color="navy", linewidth=1.5, label="Median")
    ax1.set_xlabel("Trading Day (within window)")
    ax1.set_ylabel("S&P 500 Index Level")
    ax1.set_title(f"S&P 500 Fan Chart ({len(df_raw)} simulated paths)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for q_lo, q_hi, alpha_val in [(5, 95, 0.15), (10, 90, 0.2), (25, 75, 0.3)]:
        dg_lo = np.percentile(dgs10_all, q_lo, axis=0)
        dg_hi = np.percentile(dgs10_all, q_hi, axis=0)
        ax2.fill_between(x, dg_lo, dg_hi, alpha=alpha_val, color="darkorange",
                         label=f"{q_lo}-{q_hi}th pctl" if alpha_val == 0.15 else "")
    dg_med = np.median(dgs10_all, axis=0)
    ax2.plot(x, dg_med, color="darkorange", linewidth=1.5, label="Median")
    ax2.set_xlabel("Trading Day (within window)")
    ax2.set_ylabel("10Y Treasury Yield Level")
    ax2.set_title(f"10Y Yield Fan Chart ({len(df_raw)} simulated paths)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "02_fan_chart.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

    # 图 3: 终端值分布直方图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sp_terminal = sp500_all[:, -1]
    ax1.hist(sp_terminal, bins=80, density=True, color="navy", alpha=0.7, edgecolor="black")
    ax1.axvline(sp_terminal.mean(), color="red", linestyle="--", label=f"Mean={sp_terminal.mean():.1f}")
    ax1.set_title(f"SP500 Terminal Price Distribution (t={seq_len})")
    ax1.set_xlabel("Price Level")
    ax1.set_ylabel("Density")
    ax1.legend()

    dg_terminal = dgs10_all[:, -1]
    ax2.hist(dg_terminal, bins=80, density=True, color="darkorange", alpha=0.7, edgecolor="black")
    ax2.axvline(dg_terminal.mean(), color="red", linestyle="--", label=f"Mean={dg_terminal.mean():.2f}")
    ax2.set_title(f"DGS10 Terminal Yield Distribution (t={seq_len})")
    ax2.set_xlabel("Yield Level")
    ax2.set_ylabel("Density")
    ax2.legend()

    fig.tight_layout()
    save_path = os.path.join(output_dir, "03_terminal_distribution.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

    # ── 图 4: 分段包络线对比 (0-1000, 1000-2000 等) ──
    seg_size = args.segment_size
    num_paths = len(df_raw)
    segments = []
    for start_idx in range(0, num_paths, seg_size):
        end_idx = min(start_idx + seg_size, num_paths)
        if end_idx - start_idx >= 10:
            segments.append((start_idx, end_idx))
            
    if len(segments) > 0:
        print(f"Generating segmented envelope comparison ({len(segments)} segments)...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        x = np.arange(seq_len)
        
        try:
            colormap = plt.colormaps.get_cmap("tab10")
        except AttributeError:
            colormap = plt.cm.get_cmap("tab10")
            
        for i, (start, end) in enumerate(segments):
            color = colormap(i % 10)
            sp_sub = sp500_all[start:end]
            dg_sub = dgs10_all[start:end]
            
            sp_med = np.median(sp_sub, axis=0)
            sp_lo = np.percentile(sp_sub, 10, axis=0)
            sp_hi = np.percentile(sp_sub, 90, axis=0)
            
            dg_med = np.median(dg_sub, axis=0)
            dg_lo = np.percentile(dg_sub, 10, axis=0)
            dg_hi = np.percentile(dg_sub, 90, axis=0)
            
            # SP500 Segment
            ax1.plot(x, sp_med, color=color, linewidth=1.5, label=f"Paths {start}-{end} (Median)")
            ax1.fill_between(x, sp_lo, sp_hi, color=color, alpha=0.08)
            ax1.plot(x, sp_lo, color=color, linestyle="--", linewidth=0.8, alpha=0.5)
            ax1.plot(x, sp_hi, color=color, linestyle="--", linewidth=0.8, alpha=0.5)
            
            # DGS10 Segment
            ax2.plot(x, dg_med, color=color, linewidth=1.5, label=f"Paths {start}-{end} (Median)")
            ax2.fill_between(x, dg_lo, dg_hi, color=color, alpha=0.08)
            ax2.plot(x, dg_lo, color=color, linestyle="--", linewidth=0.8, alpha=0.5)
            ax2.plot(x, dg_hi, color=color, linestyle="--", linewidth=0.8, alpha=0.5)
            
        ax1.set_xlabel("Trading Day (within window)")
        ax1.set_ylabel("S&P 500 Index Level")
        ax1.set_title(f"S&P 500 Segmented Envelopes (10th-90th pctl + Median, size={seg_size})")
        ax1.legend(fontsize=8, loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel("Trading Day (within window)")
        ax2.set_ylabel("10Y Treasury Yield Level")
        ax2.set_title(f"10Y Yield Segmented Envelopes (10th-90th pctl + Median, size={seg_size})")
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(True, alpha=0.3)
        
        fig.tight_layout()
        save_path = os.path.join(output_dir, "04_segmented_envelopes.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")

    print(f"\nAll wide-format figures generated successfully.")
    print(f"\nPath statistics:")
    print(f"  SP500 terminal mean: {sp_terminal.mean():.2f}, std: {sp_terminal.std():.2f}")
    print(f"  DGS10 terminal mean: {dg_terminal.mean():.4f}, std: {dg_terminal.std():.4f}")

    # 宽表模式不需要后续的长表处理逻辑，直接退出
    import sys
    sys.exit(0)


# ============================================================
# 2b. 长表格式处理 (原始逻辑)
# ============================================================

df = df_raw

# 检测CSV格式并选择真正的资产列。
# 支持：
#   day,sp500,DGS10                          → 日期列 "day"
#   date,sp500,DGS10                         → 日期列 (第一列字符串)
#   Asset_1_Return,Asset_2_Return            → 无日期列（纯数值）
#   ,Asset_1_Return,Asset_2_Return (0,1,2..) → 数值索引，非日期
has_date_index = False

if "day" in df.columns:
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    if df["day"].notna().all():
        df = df.set_index("day")
        df = df.sort_index()
        has_date_index = True
    else:
        df = df.drop(columns=["day"])

elif df.columns[0] in ["Unnamed: 0", ""] or (
    len(df) > 0 and pd.api.types.is_string_dtype(df.iloc[:, 0])
):
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "date"})
    # 判断第一列是否为真正的日期（而非纯数字索引）
    sample = df["date"].dropna().head(20)
    is_numeric_index = pd.to_numeric(sample, errors="coerce").notna().all()
    if is_numeric_index:
        # 第一列是纯数字行号（如 0, 1, 2, ...），丢弃它，使用整数索引
        df = df.drop(columns=["date"])
        df.index = range(len(df))
        df.index.name = "Trading Day"
    else:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if df["date"].notna().sum() > len(df) * 0.5:
                df = df.dropna(subset=["date"])
                df = df.set_index("date")
                df = df.sort_index()
                has_date_index = True
            else:
                df = df.drop(columns=["date"])
        except Exception:
            df = df.drop(columns=["date"])

if {"sp500", "DGS10"}.issubset(df.columns):
    col1, col2 = "sp500", "DGS10"
elif {"Asset_1_Return", "Asset_2_Return"}.issubset(df.columns):
    col1, col2 = "Asset_1_Return", "Asset_2_Return"
else:
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    numeric_cols = [col for col in numeric_cols if str(col).lower() not in {"day", "index"}]
    if len(numeric_cols) < 2:
        raise ValueError(f"Need at least two numeric asset columns, got {list(df.columns)}")
    col1, col2 = numeric_cols[0], numeric_cols[1]

# 将列重命名为标准名称以便后续处理
df = df.rename(columns={col1: "asset_1", col2: "asset_2"})[["asset_1", "asset_2"]]

# 确保数据列是数值类型
df["asset_1"] = pd.to_numeric(df["asset_1"], errors="coerce")
if col2:
    df["asset_2"] = pd.to_numeric(df["asset_2"], errors="coerce")
    df = df.dropna(subset=["asset_1", "asset_2"])
else:
    df = df.dropna(subset=["asset_1"])


# ============================================================
# 3. 打印基本信息
# ============================================================

print("Data preview:")
print(df.head())

print("\nData info:")
print(df.info())

print("\nDescriptive statistics:")
print(df.describe())

print("\nCorrelation:")
print(df[["asset_1", "asset_2"]].corr() if "asset_2" in df.columns else "Only one asset")


# ============================================================
# 4. 计算滚动统计量
# ============================================================

window = 20

df["asset_1_rolling_vol"] = df["asset_1"].rolling(window).std()
if "asset_2" in df.columns:
    df["asset_2_rolling_vol"] = df["asset_2"].rolling(window).std()
    df["rolling_corr"] = df["asset_1"].rolling(window).corr(df["asset_2"])


# X 轴标签：日期索引用 "Date"，数值索引用 "Trading Day"
x_label = "Date" if has_date_index else "Trading Day"

# ============================================================
# 5. 定义图形生成函数
# ============================================================

def plot_raw_time_series(df, output_dir):
    """图一：原始时间序列，双 y 轴"""
    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(
        df.index,
        df["asset_1"],
        label="Asset 1 Return",
        linewidth=0.8,
        color="navy",
    )

    ax1.axhline(0, linestyle="--", linewidth=1)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Asset 1 Daily Return")

    if "asset_2" in df.columns:
        ax2 = ax1.twinx()

        ax2.plot(
            df.index,
            df["asset_2"],
            label="Asset 2 Return",
            linewidth=0.8,
            alpha=0.8,
            color="darkorange",
        )

        ax2.axhline(0, linestyle=":", linewidth=1)
        ax2.set_ylabel("Asset 2 Return")

    plt.title("Asset Return Comparison")

    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    if "asset_2" in df.columns:
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    else:
        ax1.legend(lines_1, labels_1, loc="upper right")

    fig.tight_layout()

    save_path = os.path.join(output_dir, "01_raw_time_series.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {save_path}")


def plot_rolling_volatility(df, output_dir):
    """图二：20 日滚动波动率"""
    fig, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(
        df.index,
        df["asset_1_rolling_vol"],
        label="Asset 1 20D Rolling Volatility",
        linewidth=1.0,
        color="navy",
    )

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Asset 1 Rolling Volatility")

    if "asset_2_rolling_vol" in df.columns:
        ax2 = ax1.twinx()

        ax2.plot(
            df.index,
            df["asset_2_rolling_vol"],
            label="Asset 2 20D Rolling Volatility",
            linewidth=1.0,
            alpha=0.8,
            color="darkorange",
        )

        ax2.set_ylabel("Asset 2 Rolling Volatility")

    plt.title("20-Day Rolling Volatility")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    if "asset_2_rolling_vol" in df.columns:
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    else:
        ax1.legend(lines_1, labels_1, loc="upper right")

    fig.tight_layout()

    save_path = os.path.join(output_dir, "02_rolling_volatility.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {save_path}")


def plot_rolling_correlation(df, output_dir):
    """图三：20 日滚动相关系数"""
    if "rolling_corr" not in df.columns:
        print("Skipping plot_rolling_correlation: only one asset in data")
        return
    
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        df.index,
        df["rolling_corr"],
        linewidth=1.0,
        label="20D Rolling Correlation",
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylim(-1, 1)

    ax.set_title("20-Day Rolling Correlation: Asset 1 vs Asset 2")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Rolling Correlation")
    ax.legend(loc="upper right")

    fig.tight_layout()

    save_path = os.path.join(output_dir, "03_rolling_correlation.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {save_path}")


def plot_trend_overview(df, output_dir):
    """图四：综合趋势图，推荐用于报告"""
    # 根据是否有两个资产动态调整图表数量
    num_plots = 3 if "asset_2" in df.columns else 2
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 5*num_plots), sharex=True)
    if num_plots == 2:
        axes = [axes[0], axes[1]]  # 确保axes始终是列表

    # 第一行：Asset 1
    axes[0].plot(
        df.index,
        df["asset_1"],
        linewidth=0.8,
        color="navy",
    )

    axes[0].axhline(0, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Asset 1 Return")
    axes[0].set_title("Asset 1 Return")

    # 第二行：Asset 2（如果存在）
    if "asset_2" in df.columns:
        axes[1].plot(
            df.index,
            df["asset_2"],
            linewidth=0.8,
            color="darkorange",
        )

        axes[1].axhline(0, linestyle="--", linewidth=1)
        axes[1].set_ylabel("Asset 2 Return")
        axes[1].set_title("Asset 2 Return")

        # 第三行：20 日滚动相关
        if "rolling_corr" in df.columns:
            axes[2].plot(
                df.index,
                df["rolling_corr"],
                linewidth=1.0,
            )

            axes[2].axhline(0, linestyle="--", linewidth=1)
            axes[2].set_ylim(-1, 1)
            axes[2].set_ylabel("Correlation")
            axes[2].set_title("20-Day Rolling Correlation")
            axes[2].set_xlabel(x_label)
    else:
        axes[-1].set_xlabel(x_label)

    plt.tight_layout()

    save_path = os.path.join(output_dir, "04_trend_overview.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {save_path}")


def plot_distribution_histograms(df, output_dir):
    """图五：分布直方图"""
    num_assets = 2 if "asset_2" in df.columns else 1
    fig, axes = plt.subplots(num_assets, 1, figsize=(10, 4*num_assets))
    if num_assets == 1:
        axes = [axes]

    axes[0].hist(df["asset_1"], bins=100, density=True, color="navy", alpha=0.7, edgecolor="black")
    axes[0].set_title("Distribution of Asset 1 Return")
    axes[0].set_xlabel("Asset 1 Return")
    axes[0].set_ylabel("Density")

    if "asset_2" in df.columns:
        axes[1].hist(df["asset_2"], bins=100, density=True, color="darkorange", alpha=0.7, edgecolor="black")
        axes[1].set_title("Distribution of Asset 2 Return")
        axes[1].set_xlabel("Asset 2 Return")
        axes[1].set_ylabel("Density")

    plt.tight_layout()

    save_path = os.path.join(output_dir, "05_distribution_histograms.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {save_path}")


def plot_scatter(df, output_dir):
    """图六：散点图，观察两个资产的关系"""
    if "asset_2" not in df.columns:
        print("Skipping plot_scatter: only one asset in data")
        return
    
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        df["asset_1"],
        df["asset_2"],
        s=8,
        alpha=0.5,
        color="darkgreen",
        edgecolor="none",
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_title("Scatter Plot: Asset 1 vs Asset 2 Return")
    ax.set_xlabel("Asset 1 Return")
    ax.set_ylabel("Asset 2 Return")

    fig.tight_layout()

    save_path = os.path.join(output_dir, "06_scatter_sp500_dgs10.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {save_path}")


# ============================================================
# 6. 调用指定数量的图形生成函数
# ============================================================

print(f"\nGenerating {num_figures} figure(s)...\n")

if num_figures >= 1:
    plot_raw_time_series(df, output_dir)

if num_figures >= 2:
    plot_rolling_volatility(df, output_dir)

if num_figures >= 3:
    if "asset_2" in df.columns:
        plot_rolling_correlation(df, output_dir)
    else:
        print("Skipping figure 3: requires two assets")

if num_figures >= 4:
    plot_trend_overview(df, output_dir)

if num_figures >= 5:
    plot_distribution_histograms(df, output_dir)

if num_figures >= 6:
    if "asset_2" in df.columns:
        plot_scatter(df, output_dir)
    else:
        print("Skipping figure 6: requires two assets")

# ============================================================
# 7. 保存带滚动统计量的数据
# ============================================================

processed_dir = "data/processed"
os.makedirs(processed_dir, exist_ok=True)

processed_path = os.path.join(processed_dir, "mizuho_data_with_rolling_stats.csv")
df.to_csv(processed_path)

print(f"\nProcessed data saved: {processed_path}")
print(f"All {num_figures} figure(s) generated successfully.")
