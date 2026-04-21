#!/usr/bin/env python3
"""
Script đọc nội dung các file .pt trong thư mục checkpoints.
Usage:
    python read_checkpoints.py                          # Liệt kê tất cả checkpoints
    python read_checkpoints.py --all                    # Đọc tất cả
    python read_checkpoints.py --name bpr_mf__none__seed42   # Đọc 1 checkpoint cụ thể
    python read_checkpoints.py --summary                # Bảng tóm tắt metrics tất cả runs
"""

import argparse
import sys
from pathlib import Path

import torch


CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"


def load_pt(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def fmt_value(v):
    shape = getattr(v, "shape", None)
    if shape is not None:
        return f"{v.dtype} {list(shape)}"
    if isinstance(v, dict):
        return f"dict({len(v)} keys): {list(v.keys())}"
    if isinstance(v, list):
        return f"list({len(v)} items)"
    return repr(v)


def print_model(ckpt, indent=2):
    pad = " " * indent
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            print(f"{pad}{k}: {fmt_value(v)}")
    else:
        print(f"{pad}{type(ckpt).__name__}: {fmt_value(ckpt)}")


def print_training_state(ckpt, verbose=False):
    pad = "  "
    scalar_keys = ["epoch", "best_epoch", "patience_counter", "best_ndcg"]
    metric_keys = ["best_metrics"]

    for k in scalar_keys:
        if k in ckpt:
            print(f"{pad}{k}: {ckpt[k]}")

    if "best_metrics" in ckpt:
        print(f"{pad}best_metrics:")
        for mk, mv in ckpt["best_metrics"].items():
            print(f"{pad}  {mk}: {float(mv):.6f}")

    if "history" in ckpt and ckpt["history"]:
        history = ckpt["history"]
        eval_epochs = [h for h in history if "NDCG@10" in h]
        print(f"{pad}training_history: {len(history)} epochs, {len(eval_epochs)} eval points")
        if verbose and eval_epochs:
            print(f"{pad}  {'Epoch':>6}  {'Loss':>10}  {'HR@10':>8}  {'NDCG@10':>8}  {'Recall@10':>10}")
            print(f"{pad}  " + "-" * 50)
            for e in eval_epochs:
                loss = history[e["epoch"] - 1].get("train_loss", float("nan"))
                print(
                    f"{pad}  {e['epoch']:>6}  {loss:>10.6f}  "
                    f"{float(e['HR@10']):>8.4f}  {float(e['NDCG@10']):>8.4f}  "
                    f"{float(e['Recall@10']):>10.6f}"
                )

    for k, v in ckpt.items():
        if k not in scalar_keys + metric_keys + ["history", "model_state_dict", "optimizer_state_dict"]:
            print(f"{pad}{k}: {fmt_value(v)}")

    if "model_state_dict" in ckpt:
        print(f"{pad}model_state_dict:")
        for k, v in ckpt["model_state_dict"].items():
            print(f"{pad}  {k}: {v.dtype} {list(v.shape)}")


def read_checkpoint(name: str, verbose=False):
    path = CHECKPOINTS_DIR / name
    if not path.exists():
        print(f"[!] Không tìm thấy: {path}")
        return

    print(f"\n{'='*60}")
    print(f"Checkpoint: {name}")
    print(f"{'='*60}")

    for pt_file in sorted(path.glob("*.pt")):
        print(f"\n--- {pt_file.name} ---")
        ckpt = load_pt(pt_file)

        if pt_file.name == "best_model.pt":
            print_model(ckpt)
        elif pt_file.name == "training_state.pt":
            print_training_state(ckpt, verbose=verbose)
        else:
            print_model(ckpt)


def list_checkpoints():
    dirs = sorted(d for d in CHECKPOINTS_DIR.iterdir() if d.is_dir())
    print(f"{'#':>3}  {'Checkpoint':50}  {'Files'}")
    print("-" * 75)
    for i, d in enumerate(dirs, 1):
        files = [f.name for f in d.glob("*.pt")]
        print(f"{i:>3}  {d.name:50}  {', '.join(files)}")
    print(f"\nTổng: {len(dirs)} checkpoints")


def summary_table():
    dirs = sorted(d for d in CHECKPOINTS_DIR.iterdir() if d.is_dir())
    rows = []
    for d in dirs:
        state_file = d / "training_state.pt"
        if not state_file.exists():
            continue
        ckpt = load_pt(state_file)
        m = ckpt.get("best_metrics", {})
        rows.append({
            "name": d.name,
            "epoch": ckpt.get("best_epoch", "-"),
            "HR@10": float(m.get("HR@10", float("nan"))),
            "NDCG@10": float(m.get("NDCG@10", float("nan"))),
            "Recall@10": float(m.get("Recall@10", float("nan"))),
            "HR@20": float(m.get("HR@20", float("nan"))),
            "NDCG@20": float(m.get("NDCG@20", float("nan"))),
        })

    if not rows:
        print("Không có training_state.pt nào.")
        return

    rows.sort(key=lambda r: r["NDCG@10"], reverse=True)

    header = f"{'#':>3}  {'Checkpoint':45}  {'Ep':>4}  {'HR@10':>7}  {'NDCG@10':>8}  {'R@10':>7}  {'HR@20':>7}  {'NDCG@20':>8}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        print(
            f"{i:>3}  {r['name']:45}  {str(r['epoch']):>4}  "
            f"{r['HR@10']:>7.4f}  {r['NDCG@10']:>8.4f}  {r['Recall@10']:>7.4f}  "
            f"{r['HR@20']:>7.4f}  {r['NDCG@20']:>8.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Đọc nội dung file .pt trong checkpoints/")
    parser.add_argument("--name", "-n", help="Tên checkpoint cụ thể")
    parser.add_argument("--all", "-a", action="store_true", help="Đọc tất cả checkpoints")
    parser.add_argument("--summary", "-s", action="store_true", help="Bảng tóm tắt metrics")
    parser.add_argument("--verbose", "-v", action="store_true", help="Hiển thị lịch sử training đầy đủ")
    args = parser.parse_args()

    if args.summary:
        summary_table()
    elif args.name:
        read_checkpoint(args.name, verbose=args.verbose)
    elif args.all:
        dirs = sorted(d.name for d in CHECKPOINTS_DIR.iterdir() if d.is_dir())
        for name in dirs:
            read_checkpoint(name, verbose=args.verbose)
    else:
        list_checkpoints()
        print("\nDùng --summary để xem bảng metrics, hoặc --name <tên> để xem chi tiết.")


if __name__ == "__main__":
    main()
