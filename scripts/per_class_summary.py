#!/usr/bin/env python3
"""
Compute per-class (per-sequence) summary from 0_all_frames_metrics_results.xlsx.
Output: mean ADD-S, AR, VSD, MSSD, MSPD, R_error, T_error per Class, and save to CSV/Excel.

Usage:
  python scripts/per_class_summary.py \
    --input ./results/ycbineoat_results/ycbineoat_query_full_no_gate/2026-02-27_11-30-05/0_all_frames_metrics_results.xlsx \
    --output ./results/ycbineoat_results/ycbineoat_query_full_no_gate/2026-02-27_11-30-05/per_class_means.xlsx
"""
import argparse
import os
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Per-class summary from all_frames_metrics_results.xlsx")
    parser.add_argument("--input", type=str, required=True, help="Path to 0_all_frames_metrics_results.xlsx")
    parser.add_argument("--output", type=str, default="", help="Output path (default: same dir as input, per_class_means.xlsx)")
    parser.add_argument("--format", type=str, default="xlsx", choices=["xlsx", "csv"], help="Output format")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_excel(input_path)

    # Drop aggregate row(s): Frame_ID or Class is MEAN/ALL
    if "Frame_ID" in df.columns:
        df = df[df["Frame_ID"].astype(str).str.strip().str.upper() != "MEAN"]
    if "Class" in df.columns:
        df = df[df["Class"].astype(str).str.strip().str.upper() != "ALL"]

    if "Class" not in df.columns:
        print("No 'Class' column found.")
        sys.exit(1)

    numeric_cols = ["ADD-S", "ADD", "AR", "MSSD", "MSPD", "VSD", "R_error", "T_error"]
    present = [c for c in numeric_cols if c in df.columns]
    if not present:
        print("No metric columns found among", numeric_cols)
        sys.exit(1)

    summary = df.groupby("Class", as_index=True)[present].agg("mean")
    # Optional: add frame count per class
    count = df.groupby("Class", as_index=True).size().rename("Frame_Count")
    summary = summary.join(count)

    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        base = os.path.splitext(input_path)[0]
        out_path = base.replace("0_all_frames_metrics_results", "per_class_means") + ("." + args.format)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if args.format == "csv":
        summary.to_csv(out_path)
    else:
        summary.to_excel(out_path)
    print(f"Per-class summary saved to: {out_path}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
