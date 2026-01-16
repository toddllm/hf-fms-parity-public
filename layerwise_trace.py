#!/usr/bin/env python3
"""
Capture per-layer forward-pass I/O traces (args, kwargs, outputs) for offline comparison.

Designed for HF ↔ FMS parity debugging:
- Run Hugging Face (transformers) and save a trace pickle.
- Run FMS (get_model) and save a trace pickle.
- Compare using hf-fms-parity/layerwise_compare.py.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import torch


def _setup_fms_import(fms_path: str | None) -> None:
    if not fms_path:
        return
    resolved = str(Path(fms_path).expanduser().resolve())
    if not os.path.exists(resolved):
        raise ValueError(f"--fms-path does not exist: {resolved}")
    sys.path.insert(0, resolved)


def _parse_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower().strip()
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        out = [_to_device(v, device) for v in obj]
        return type(obj)(out) if isinstance(obj, tuple) else out
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


@dataclass(frozen=True)
class TensorSummary:
    shape: Tuple[int, ...]
    dtype: str
    numel: int
    mean: float
    max: float


def _summarize_tensor(t: torch.Tensor) -> TensorSummary:
    t_cpu = t.detach()
    if t_cpu.device.type != "cpu":
        t_cpu = t_cpu.cpu()
    t_float = t_cpu.float()
    return TensorSummary(
        shape=tuple(t_cpu.shape),
        dtype=str(t_cpu.dtype),
        numel=t_cpu.numel(),
        mean=float(t_float.mean().item()) if t_float.numel() else 0.0,
        max=float(t_float.abs().max().item()) if t_float.numel() else 0.0,
    )


def _is_safe_primitive(x: Any) -> bool:
    return x is None or isinstance(x, (bool, int, float, str))


def _make_picklable(
    obj: Any,
    *,
    tensor_max_numel: int,
    keep_full_tensors: bool,
) -> Any:
    if _is_safe_primitive(obj):
        return obj

    if isinstance(obj, torch.Tensor):
        if keep_full_tensors and obj.numel() <= tensor_max_numel:
            t = obj.detach()
            if t.device.type != "cpu":
                t = t.cpu()
            return t.contiguous()
        return asdict(_summarize_tensor(obj))

    if isinstance(obj, (list, tuple)):
        out = [_make_picklable(v, tensor_max_numel=tensor_max_numel, keep_full_tensors=keep_full_tensors) for v in obj]
        return tuple(out) if isinstance(obj, tuple) else out

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            try:
                key = str(k)
            except Exception:
                key = repr(k)
            out[key] = _make_picklable(v, tensor_max_numel=tensor_max_numel, keep_full_tensors=keep_full_tensors)
        return out

    # Fallback: try to pickle as-is; if not possible, store repr.
    try:
        pickle.dumps(obj)
        return obj
    except Exception:
        return {"__repr__": repr(obj), "__type__": str(type(obj))}


@dataclass
class LayerRecord:
    module_name: str
    module_type: str
    call_index: int
    root_forward_index: int
    depth: int
    complexity: int
    param_shapes: Dict[str, Tuple[int, ...]]
    inputs: Any
    kwargs: Any
    outputs: Any


class TraceRecorder:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        first_pass_only: bool,
        tensor_max_numel: int,
        keep_full_tensors: bool,
    ):
        self._model = model
        self._first_pass_only = first_pass_only
        self._tensor_max_numel = tensor_max_numel
        self._keep_full_tensors = keep_full_tensors

        self._hooks: List[Any] = []
        self._records: List[LayerRecord] = []
        self._call_index = 0
        self._root_forward_index = 0
        self._pending: Dict[int, Tuple[int, int]] = {}  # id(module) -> (call_index, root_forward_index)

        self._name_by_module: Dict[int, str] = {id(m): n for n, m in model.named_modules()}

    def __enter__(self) -> "TraceRecorder":
        self._install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    @property
    def records(self) -> List[LayerRecord]:
        return self._records

    def remove(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()

    def _install(self) -> None:
        def root_pre_hook(module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
            self._root_forward_index += 1

        self._hooks.append(self._model.register_forward_pre_hook(root_pre_hook, with_kwargs=True))

        def pre_hook(module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
            if self._first_pass_only and self._root_forward_index > 1:
                return

            self._call_index += 1
            self._pending[id(module)] = (self._call_index, self._root_forward_index)

        def post_hook(
            module: torch.nn.Module,
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any],
            output: Any,
        ) -> None:
            pending = self._pending.pop(id(module), None)
            if pending is None:
                return
            call_index, root_forward_index = pending
            if self._first_pass_only and root_forward_index > 1:
                return

            module_name = self._name_by_module.get(id(module), module.__class__.__name__)
            depth = module_name.count(".") + (module_name.count("[") if "[" in module_name else 0)
            complexity = len(getattr(module, "_modules", {}) or {})
            param_shapes: Dict[str, Tuple[int, ...]] = {
                name: tuple(p.shape) for name, p in module.named_parameters(recurse=False)
            }

            rec = LayerRecord(
                module_name=module_name,
                module_type=module.__class__.__name__,
                call_index=call_index,
                root_forward_index=root_forward_index,
                depth=depth,
                complexity=complexity,
                param_shapes=param_shapes,
                inputs=_make_picklable(
                    list(args),
                    tensor_max_numel=self._tensor_max_numel,
                    keep_full_tensors=self._keep_full_tensors,
                ),
                kwargs=_make_picklable(
                    dict(kwargs),
                    tensor_max_numel=self._tensor_max_numel,
                    keep_full_tensors=self._keep_full_tensors,
                ),
                outputs=_make_picklable(
                    output,
                    tensor_max_numel=self._tensor_max_numel,
                    keep_full_tensors=self._keep_full_tensors,
                ),
            )
            self._records.append(rec)

        for _, m in self._model.named_modules():
            self._hooks.append(m.register_forward_pre_hook(pre_hook, with_kwargs=True))
            self._hooks.append(m.register_forward_hook(post_hook, with_kwargs=True))


def _load_hf_model_and_tokenizer(
    model_id: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> Tuple[torch.nn.Module, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model, tok


def _load_fms_model(
    *,
    fms_path: str | None,
    architecture: str,
    variant: str | None,
    model_path: str | None,
    device_type: str,
    dtype: torch.dtype,
) -> torch.nn.Module:
    _setup_fms_import(fms_path)
    from fms.models import get_model

    kwargs: Dict[str, Any] = {"architecture": architecture}
    if variant is not None:
        kwargs["variant"] = variant
    if model_path is not None:
        kwargs["model_path"] = model_path

    model = get_model(
        device_type=device_type,
        data_type=dtype,
        fused_weights=False,
        **kwargs,
    )
    model.eval()
    return model


def _run_hf(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    mode: str,
) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = _to_device(inputs, device)
    with torch.no_grad():
        if mode == "forward":
            _ = model(**inputs, use_cache=True)
        else:
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    return {"input_ids": inputs.get("input_ids")}


def _run_fms(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    prompt: str,
    device: torch.device,
    device_type: str,
    max_new_tokens: int,
    mode: str,
) -> Dict[str, Any]:
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    ids = ids.to(device)

    with torch.no_grad():
        if mode == "forward":
            _ = model(ids, use_cache=True)
        else:
            from fms.utils.generation import generate

            max_seq_len = getattr(getattr(model, "config", None), "max_expected_seq_len", ids.shape[1])
            _ = generate(
                model,
                ids,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=False,
                max_seq_len=max_seq_len,
                timing="",
                eos_token_id=None,
                contiguous_cache=True,
                extra_kwargs={},
            )
    return {"input_ids": ids}


def main() -> int:
    p = argparse.ArgumentParser(description="Capture per-layer traces for HF or FMS.")
    p.add_argument("--impl", choices=["hf", "fms"], required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu", help="cpu|cuda|mps")
    p.add_argument("--dtype", type=str, default="float32", help="float32|bfloat16|float16")
    p.add_argument("--mode", choices=["generate", "forward"], default="generate")
    p.add_argument("--max-new-tokens", type=int, default=1)
    p.add_argument("--first-pass-only", action="store_true")
    p.add_argument("--tensor-max-numel", type=int, default=2_000_000)
    p.add_argument("--keep-full-tensors", action="store_true", help="Store full tensors up to --tensor-max-numel.")
    p.add_argument("--out", type=str, required=True)

    # HF args
    p.add_argument("--model", type=str, help="HF model id/path (for --impl hf).")
    p.add_argument("--trust-remote-code", action="store_true")

    # FMS args
    p.add_argument("--fms-path", type=str, default=None)
    p.add_argument("--architecture", type=str, default="hf_pretrained")
    p.add_argument("--variant", type=str, default=None)
    p.add_argument("--model-path", type=str, default=None)

    args = p.parse_args()

    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)

    meta: Dict[str, Any] = {
        "impl": args.impl,
        "prompt": args.prompt,
        "device": str(device),
        "dtype": str(dtype),
        "mode": args.mode,
        "max_new_tokens": args.max_new_tokens,
        "first_pass_only": bool(args.first_pass_only),
        "tensor_max_numel": int(args.tensor_max_numel),
        "keep_full_tensors": bool(args.keep_full_tensors),
        "timestamp": time.time(),
    }

    if args.impl == "hf":
        if not args.model:
            raise SystemExit("--model is required for --impl hf")
        model, tokenizer = _load_hf_model_and_tokenizer(
            args.model,
            device=device,
            dtype=dtype,
            trust_remote_code=bool(args.trust_remote_code),
        )
        meta["model"] = args.model
        runner = lambda: _run_hf(  # noqa: E731
            model,
            tokenizer,
            prompt=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            mode=args.mode,
        )
    else:
        if not args.variant and not args.model_path:
            raise SystemExit("For --impl fms, pass --variant or --model-path")
        model = _load_fms_model(
            fms_path=args.fms_path,
            architecture=args.architecture,
            variant=args.variant,
            model_path=args.model_path,
            device_type=args.device,
            dtype=dtype,
        )
        from transformers import AutoTokenizer

        tok_id = args.variant or args.model_path
        tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=bool(args.trust_remote_code))
        meta["architecture"] = args.architecture
        meta["variant"] = args.variant
        meta["model_path"] = args.model_path
        runner = lambda: _run_fms(  # noqa: E731
            model,
            tokenizer,
            prompt=args.prompt,
            device=device,
            device_type=args.device,
            max_new_tokens=args.max_new_tokens,
            mode=args.mode,
        )

    with TraceRecorder(
        model,
        first_pass_only=bool(args.first_pass_only),
        tensor_max_numel=int(args.tensor_max_numel),
        keep_full_tensors=bool(args.keep_full_tensors),
    ) as tr:
        runner_info = runner()

    payload = {
        "meta": meta,
        "runner_info": _make_picklable(runner_info, tensor_max_numel=args.tensor_max_numel, keep_full_tensors=True),
        "records": [asdict(r) for r in tr.records],
    }

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(json.dumps({"out": str(out_path), "records": len(tr.records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

