#!/usr/bin/env python3
"""
Convert between the project's two table formats with fixed Asset1/Asset2 rules.

Rules:
  - Asset1: S&P 500 daily return <-> S&P daily index level
  - Asset2: 10Y Treasury yield daily difference <-> 10Y yield daily level

Simple usage:
  python data/price_table_converter.py to-level data/train_sp500_us10y.csv
  python data/price_table_converter.py to-diff data/train_sp500_us10y_levels.csv

If --output is omitted, the script writes next to the input file with a suffix:
  *_levels.csv for to-level
  *_changes.csv for to-diff
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SP500_STANDARD_COL = "sp500"
DGS10_STANDARD_COL = "DGS10"
DEFAULT_RETURN_TABLE = Path("data/train_sp500_us10y.csv")
DEFAULT_LEVEL_TABLE = Path("data/train_sp500_us10y_levels.csv")
DEFAULT_REBUILT_DIFF_TABLE = Path("data/train_sp500_us10y_rebuilt_changes.csv")

# 原始差值/收益率表没有绝对初始值，因此这里使用查看用的基准值。
SP500_INITIAL_LEVEL = 500.0
DGS10_INITIAL_LEVEL = 2.0


MODE_ALIASES = {
    "to-level": "to-level",
    "diff-to-level": "to-level",
    "returns-to-level": "to-level",
    "level-to-diff": "to-diff",
    "levels-to-diff": "to-diff",
    "to-diff": "to-diff",
}

COLUMN_PAIRS = [
    ("sp500", "DGS10"),
    ("SP500", "DGS10"),
    ("Asset_1_Return", "Asset_2_Return"),
    ("Asset_1_Level", "Asset_2_Level"),
    ("asset_1", "asset_2"),
    ("Asset1", "Asset2"),
]


def read_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")

    first_col = df.columns[0]
    first_values = df.iloc[:, 0]
    first_is_index = (
        str(first_col).startswith("Unnamed")
        or str(first_col).lower() in {"date", "day", "index"}
        or pd.api.types.is_string_dtype(first_values)
    )
    if first_is_index:
        df = df.set_index(first_col)
        if str(first_col).lower() in {"", "unnamed: 0"}:
            df.index.name = "date"
    return df


def choose_asset_columns(
    df: pd.DataFrame,
    asset1_col: str | None,
    asset2_col: str | None,
) -> tuple[str, str]:
    if asset1_col or asset2_col:
        if not asset1_col or not asset2_col:
            raise ValueError("Please provide both --asset1-col and --asset2-col, or neither.")
        missing = [col for col in (asset1_col, asset2_col) if col not in df.columns]
        if missing:
            raise ValueError(f"Missing requested columns {missing}; got {list(df.columns)}")
        return asset1_col, asset2_col

    for col1, col2 in COLUMN_PAIRS:
        if col1 in df.columns and col2 in df.columns:
            return col1, col2

    numeric_cols = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            numeric_cols.append(col)

    if len(numeric_cols) < 2:
        raise ValueError(
            "Could not auto-detect two numeric asset columns. "
            "Use --asset1-col and --asset2-col to specify them explicitly."
        )
    return numeric_cols[0], numeric_cols[1]


def read_asset_table(
    path: Path,
    asset1_col: str | None,
    asset2_col: str | None,
) -> tuple[pd.DataFrame, tuple[str, str]]:
    raw = read_raw_csv(path)
    col1, col2 = choose_asset_columns(raw, asset1_col, asset2_col)

    df = raw[[col1, col2]].copy()
    df[col1] = pd.to_numeric(df[col1], errors="coerce")
    df[col2] = pd.to_numeric(df[col2], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df, (col1, col2)


def detect_wide_format(df: pd.DataFrame) -> bool:
    """
    检测 CSV 是否为 DDPM 生成的宽表格式。
    宽表特征: 列名匹配 sp500_0, sp500_1, ..., sp500_N, dgs10_0, dgs10_1, ..., dgs10_N
    每行是一条独立的模拟路径，列是时间步。
    """
    cols = df.columns.tolist()
    sp500_cols = sorted([c for c in cols if c.startswith("sp500_") and c[6:].isdigit()],
                        key=lambda c: int(c[6:]))
    dgs10_cols = sorted([c for c in cols if c.startswith("dgs10_") and c[6:].isdigit()],
                        key=lambda c: int(c[6:]))
    if len(sp500_cols) >= 2 and len(dgs10_cols) >= 2:
        # 验证列号是否连续: sp500_0, sp500_1, ...
        sp_indices = [int(c[6:]) for c in sp500_cols]
        dg_indices = [int(c[6:]) for c in dgs10_cols]
        if sp_indices == list(range(len(sp_indices))) and dg_indices == list(range(len(dg_indices))):
            return True
    return False


def wide_changes_to_levels(
    df: pd.DataFrame,
    sp500_initial: float,
    dgs10_initial: float,
) -> pd.DataFrame:
    """
    宽表格式 diff-to-level 转换。
    每行是一条独立模拟路径，对每条路径独立执行:
      - SP500: price_t = initial * cumprod(1 + r_t)   (沿时间轴 axis=1)
      - DGS10: yield_t = initial + cumsum(d_t)        (沿时间轴 axis=1)
    """
    cols = df.columns.tolist()
    sp500_cols = sorted([c for c in cols if c.startswith("sp500_") and c[6:].isdigit()],
                        key=lambda c: int(c[6:]))
    dgs10_cols = sorted([c for c in cols if c.startswith("dgs10_") and c[6:].isdigit()],
                        key=lambda c: int(c[6:]))

    seq_len = len(sp500_cols)

    # 提取 numpy 数组: (N, seq_len)
    sp500_returns = df[sp500_cols].values
    dgs10_diffs = df[dgs10_cols].values

    # 逐路径 diff → level
    sp500_levels = sp500_initial * np.cumprod(1.0 + sp500_returns, axis=1)  # (N, seq_len)
    dgs10_levels = dgs10_initial + np.cumsum(dgs10_diffs, axis=1)           # (N, seq_len)

    # 构造输出 DataFrame
    sp_level_cols = [f"sp500_level_{i}" for i in range(seq_len)]
    dg_level_cols = [f"dgs10_level_{i}" for i in range(seq_len)]

    out = pd.DataFrame(
        np.concatenate([sp500_levels, dgs10_levels], axis=1),
        columns=sp_level_cols + dg_level_cols,
    )

    print(f"[Wide Format] Detected {len(df)} simulation paths × {seq_len} timesteps")
    print(f"  SP500 levels: range [{sp500_levels.min():.2f}, {sp500_levels.max():.2f}]")
    print(f"  DGS10 levels: range [{dgs10_levels.min():.4f}, {dgs10_levels.max():.4f}]")
    print(f"  SP500 per-path return std: {sp500_returns.std(axis=1).mean():.6f}")

    return out


def changes_to_levels(
    df: pd.DataFrame,
    columns: tuple[str, str],
    sp500_initial: float,
    dgs10_initial: float,
) -> pd.DataFrame:
    """Asset1 daily return -> index level; Asset2 daily diff -> yield level. (长表格式)"""
    col1, col2 = columns
    out = pd.DataFrame(index=df.index)
    out[col1] = sp500_initial * (1.0 + df[col1]).cumprod()
    out[col2] = dgs10_initial + df[col2].cumsum()
    return out


def levels_to_changes(df: pd.DataFrame, columns: tuple[str, str]) -> pd.DataFrame:
    """Asset1 index level -> daily return; Asset2 yield level -> daily diff."""
    col1, col2 = columns
    out = pd.DataFrame(index=df.index)
    out[col1] = df[col1].pct_change()
    out[col2] = df[col2].diff()
    return out.iloc[1:]


def default_paths(mode: str) -> tuple[Path, Path]:
    if mode == "to-level":
        return DEFAULT_RETURN_TABLE, DEFAULT_LEVEL_TABLE
    return DEFAULT_LEVEL_TABLE, DEFAULT_REBUILT_DIFF_TABLE


def derive_output_path(input_path: Path, mode: str) -> Path:
    suffix = "_levels" if mode == "to-level" else "_changes"
    return input_path.with_name(f"{input_path.stem}{suffix}.csv")


def resolve_paths(args: argparse.Namespace, mode: str) -> tuple[Path, Path]:
    if args.input and args.csv:
        raise ValueError("Use either positional CSV or --input, not both.")

    if args.input:
        input_path = Path(args.input)
    elif args.csv:
        input_path = Path(args.csv)
    else:
        input_path, default_output = default_paths(mode)
        output_path = Path(args.output) if args.output else default_output
        return input_path, output_path

    output_path = Path(args.output) if args.output else derive_output_path(input_path, mode)
    return input_path, output_path


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CSVs between Asset1/Asset2 return-difference tables and level tables."
    )
    parser.add_argument(
        "direction",
        choices=sorted(MODE_ALIASES),
        help="Use 'to-level' for returns/differences -> levels, or 'to-diff' for levels -> returns/differences.",
    )
    parser.add_argument("csv", nargs="?", help="Input CSV path")
    parser.add_argument("--input", default=None, help="Input CSV path, kept for compatibility")
    parser.add_argument("--output", default=None, help="Output CSV path; defaults to *_levels.csv or *_changes.csv")
    parser.add_argument("--asset1-col", default=None, help="Column for Asset1 / S&P 500")
    parser.add_argument("--asset2-col", default=None, help="Column for Asset2 / 10Y Treasury yield")
    parser.add_argument("--sp500-initial", type=float, default=SP500_INITIAL_LEVEL)
    parser.add_argument("--dgs10-initial", type=float, default=DGS10_INITIAL_LEVEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = MODE_ALIASES[args.direction]
    input_path, output_path = resolve_paths(args, mode)

    if mode == "to-level":
        # 先读取原始 CSV 检测是否为 DDPM 宽表格式
        raw_df = read_raw_csv(input_path)
        if detect_wide_format(raw_df):
            print(f"\n[Auto-detected] DDPM wide-table format (rows=paths, cols=timesteps)")
            out = wide_changes_to_levels(raw_df, args.sp500_initial, args.dgs10_initial)
            description = "宽表 差值/收益率 -> 宽表 绝对值/水平 (逐路径独立转换)"
            write_table(out, output_path)
            print(f"\nMode: {description}")
            print(f"Input: {input_path}")
            print(f"Output: {output_path}")
            print(f"Rows (paths): {len(out)}")
            print(f"Columns: {len(out.columns)}")
            print(f"Initial levels: SP500={args.sp500_initial}, DGS10={args.dgs10_initial}")
            print("\nPreview (first 3 paths, first 5 timesteps):")
            preview_cols = [c for c in out.columns[:5]] + ["..."] if len(out.columns) > 5 else out.columns.tolist()
            print(out.iloc[:3, :5].to_string())
            return

        # 长表格式处理
        df, columns = read_asset_table(input_path, args.asset1_col, args.asset2_col)
        out = changes_to_levels(df, columns, args.sp500_initial, args.dgs10_initial)
        description = "差值/收益率表 -> 绝对值/水平表"
    else:
        df, columns = read_asset_table(input_path, args.asset1_col, args.asset2_col)
        out = levels_to_changes(df, columns)
        description = "绝对值/水平表 -> 差值/收益率表"

    write_table(out, output_path)

    print(f"Mode: {description}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(out)}")
    print("Columns:")
    print(f"  Asset1 / S&P 500: {columns[0]}")
    print(f"  Asset2 / 10Y Treasury: {columns[1]}")
    print("Rules:")
    print("  Asset1: S&P 500 daily return <-> S&P daily index level")
    print("  Asset2: 10Y Treasury yield daily difference <-> 10Y yield daily level")
    if mode == "to-level":
        print(f"Initial levels: Asset1={args.sp500_initial}, Asset2={args.dgs10_initial}")
    print("\nPreview:")
    print(out.head().to_string())


if __name__ == "__main__":
    main()
