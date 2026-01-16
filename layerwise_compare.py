#!/usr/bin/env python3
"""
Compare two layerwise trace pickles (HF vs FMS) produced by layerwise_trace.py.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch


@dataclass(frozen=True)
class DiffStats:
    shape: Tuple[int, ...]
    max_abs: float
    mean_abs: float
    sum_abs: float
    cos_sim: Optional[float]


def _load(path: str) -> Dict[str, Any]:
    with open(Path(path).expanduser(), "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "records" not in obj:
        raise ValueError(f"Not a trace payload: {path}")
    return obj


def _load_map(path: str) -> Dict[str, str]:
    with open(Path(path).expanduser(), "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--map must be a JSON object of {prefix: prefix}")
    return {str(k): str(v) for k, v in data.items()}


def _map_layer_name(name: str, mapping: Mapping[str, str]) -> str:
    for src, dst in mapping.items():
        if name.startswith(src):
            return dst + name[len(src) :]
    return name


def _to_tensor(x: Any) -> Optional[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return x
    return None


def _iter_tensors(obj: Any, prefix: str = "") -> Iterable[Tuple[str, torch.Tensor]]:
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
        return
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_tensors(v, f"{prefix}[{i}]")
        return
    if isinstance(obj, tuple):
        for i, v in enumerate(obj):
            yield from _iter_tensors(v, f"{prefix}({i})")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("__"):
                continue
            yield from _iter_tensors(v, f"{prefix}.{k}" if prefix else str(k))
        return


def _tensor_stats(a: torch.Tensor, b: torch.Tensor) -> DiffStats:
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    diff = (a - b).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    sum_abs = float(diff.sum().item()) if diff.numel() else 0.0

    cos_sim: Optional[float] = None
    if a.numel():
        a_flat = a.flatten()
        b_flat = b.flatten()
        denom = (a_flat.norm() * b_flat.norm()).item()
        cos_sim = float((a_flat @ b_flat).item() / denom) if denom else None

    return DiffStats(shape=tuple(a.shape), max_abs=max_abs, mean_abs=mean_abs, sum_abs=sum_abs, cos_sim=cos_sim)


def _index_by_name(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        name = r.get("module_name")
        if not isinstance(name, str):
            continue
        out.setdefault(name, []).append(r)
    return out


def _best_record(records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    # Prefer the first call (lowest call_index) within the first root forward.
    sorted_records = sorted(records, key=lambda r: (r.get("root_forward_index", 0), r.get("call_index", 0)))
    return sorted_records[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two trace pickles layer-by-layer.")
    ap.add_argument("--hf", required=True, help="HF trace pickle")
    ap.add_argument("--fms", required=True, help="FMS trace pickle")
    ap.add_argument("--map", required=True, help="JSON mapping of HF module prefix -> FMS module prefix")
    ap.add_argument("--compare", choices=["inputs", "outputs"], default="outputs")
    ap.add_argument("--metric", choices=["max_abs", "mean_abs", "sum_abs", "cos_sim"], default="max_abs")
    ap.add_argument("--limit", type=int, default=50, help="Max rows to print")

    args = ap.parse_args()

    hf = _load(args.hf)
    fms = _load(args.fms)
    mapping = _load_map(args.map)

    hf_by_name = _index_by_name(hf["records"])
    fms_by_name = _index_by_name(fms["records"])

    rows: List[Dict[str, Any]] = []
    for hf_name, hf_records in hf_by_name.items():
        hf_rec = _best_record(hf_records)
        if hf_rec is None:
            continue

        fms_name = _map_layer_name(hf_name, mapping)
        fms_rec = _best_record(fms_by_name.get(fms_name, []))
        if fms_rec is None:
            continue

        hf_obj = hf_rec.get(args.compare)
        fms_obj = fms_rec.get(args.compare)

        # Heuristic: compare the first tensor we can find.
        hf_tensors = list(_iter_tensors(hf_obj))
        fms_tensors = list(_iter_tensors(fms_obj))
        if not hf_tensors or not fms_tensors:
            continue

        hf_path, hf_tensor = hf_tensors[0]
        fms_path, fms_tensor = fms_tensors[0]
        try:
            stats = _tensor_stats(hf_tensor, fms_tensor)
        except Exception as e:
            rows.append(
                {
                    "hf_layer": hf_name,
                    "fms_layer": fms_name,
                    "hf_path": hf_path,
                    "fms_path": fms_path,
                    "error": str(e),
                }
            )
            continue

        row = {
            "hf_layer": hf_name,
            "fms_layer": fms_name,
            "hf_path": hf_path,
            "fms_path": fms_path,
            "shape": stats.shape,
            "max_abs": stats.max_abs,
            "mean_abs": stats.mean_abs,
            "sum_abs": stats.sum_abs,
            "cos_sim": stats.cos_sim,
        }
        rows.append(row)

    def sort_key(r: Dict[str, Any]) -> float:
        if "error" in r:
            return float("inf")
        val = r.get(args.metric)
        if args.metric == "cos_sim":
            # Lower cosine similarity is "worse"
            return 1.0 - float(val) if val is not None else 1.0
        return float(val)

    rows_sorted = sorted(rows, key=sort_key, reverse=args.metric != "cos_sim")
    rows_sorted = rows_sorted[: max(0, int(args.limit))]

    print(json.dumps({"hf_meta": hf.get("meta"), "fms_meta": fms.get("meta")}, indent=2))
    print("")
    for r in rows_sorted:
        if "error" in r:
            print(f"{r['hf_layer']} -> {r['fms_layer']} ERROR: {r['error']}")
            continue
        print(
            f"{r['hf_layer']} -> {r['fms_layer']} "
            f"{r['shape']} max={r['max_abs']:.3e} mean={r['mean_abs']:.3e} sum={r['sum_abs']:.3e} "
            + (f"cos={r['cos_sim']:.6f}" if r["cos_sim"] is not None else "cos=None")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

