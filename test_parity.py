#!/usr/bin/env python3
"""
Idefics3 FMS vs HuggingFace Parity Validation Suite

Validates parity between:
- FMS-native implementation loaded via `fms.models.get_model("hf_pretrained", ...)`
- HuggingFace reference implementation (transformers)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def setup_fms_import(fms_path: str | None):
    if fms_path:
        fms_path = str(Path(fms_path).resolve())
        if not os.path.exists(fms_path):
            raise ValueError(f"FMS path does not exist: {fms_path}")
        print(f"Using FMS from: {fms_path}")
        sys.path.insert(0, fms_path)
        return

    try:
        import fms  # noqa: F401

        return
    except ImportError as e:
        raise ImportError(
            "FMS not found. Either:\n"
            "  1) Install editable: pip install -e /path/to/foundation-model-stack\n"
            "  2) Or pass --fms-path /path/to/foundation-model-stack"
        ) from e


def _resolve_checkpoint(checkpoint: str, hf_revision: str | None) -> tuple[str, str]:
    """
    Resolve a Hugging Face model ID (+ optional revision) to a local directory so that
    HF and FMS load from the exact same snapshot.
    """
    # If the user already provided a local path, do not attempt any HF download.
    if os.path.exists(checkpoint):
        return str(Path(checkpoint).resolve()), str(Path(checkpoint).resolve())

    label = checkpoint if hf_revision is None else f"{checkpoint}@{hf_revision}"
    if hf_revision is None:
        return checkpoint, label

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for --hf-revision. Install with: pip install huggingface_hub"
        ) from e

    local_dir = snapshot_download(repo_id=checkpoint, revision=hf_revision)
    return local_dir, label


class ParityTester:
    def __init__(
        self,
        checkpoint_name: str,
        checkpoint_label: str,
        device: str,
        seed: int,
        default_atol: float,
        default_rtol: float,
        max_new_tokens: int,
    ):
        self.checkpoint_name = checkpoint_name
        self.checkpoint_label = checkpoint_label
        self.device = device
        self.seed = seed
        self.default_atol = default_atol
        self.default_rtol = default_rtol
        self.max_new_tokens = max_new_tokens
        self.results: Dict[str, Dict] = {}

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"\n{'='*80}")
        print("Initializing Parity Tester")
        print(f"Checkpoint: {checkpoint_label}")
        print(f"Device: {device}")
        print(f"Seed: {seed}")
        print(f"Default tolerances: atol={default_atol:.2e}, rtol={default_rtol:.2e}")
        print(f"{'='*80}\n")

        print("Loading HuggingFace processor + model...")
        self.hf_processor = AutoProcessor.from_pretrained(
            checkpoint_name,
            trust_remote_code=True,
        )
        self.hf_model = AutoModelForVision2Seq.from_pretrained(
            checkpoint_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(device)
        self.hf_model.eval()
        print("✓ HF loaded\n")

        print("Loading FMS model via hf_pretrained (exercises hf/utils + adapters)...")
        from fms.models import get_model
        from fms.models.idefics3 import load_smolvlm_preprocessor

        self.fms_preprocessor_factory = load_smolvlm_preprocessor
        self.fms_model = get_model(
            architecture="hf_pretrained",
            variant=checkpoint_name,
            device_type=device,
            data_type=torch.float32,
        )
        self.fms_model.eval()
        print("✓ FMS loaded\n")

        self.test_image = self._create_test_image()
        self.second_image = self._create_second_image()

    def _create_test_image(self) -> Image.Image:
        i_grid = np.arange(512)[:, None]
        j_grid = np.arange(512)[None, :]
        arr = np.zeros((512, 512, 3), dtype=np.uint8)
        arr[:, :, 0] = (i_grid * 255 // 512).astype(np.uint8)
        arr[:, :, 1] = (j_grid * 255 // 512).astype(np.uint8)
        arr[:, :, 2] = 128
        return Image.fromarray(arr)

    def _create_second_image(self) -> Image.Image:
        base = np.array(self._create_test_image(), dtype=np.uint8)
        return Image.fromarray(255 - base)

    def compare_tensors(
        self,
        fms_tensor: torch.Tensor,
        hf_tensor: torch.Tensor,
        name: str,
        atol: float | None = None,
        rtol: float | None = None,
    ) -> Dict[str, float]:
        atol = self.default_atol if atol is None else atol
        rtol = self.default_rtol if rtol is None else rtol

        print(f"\n{'-'*80}")
        print(f"Comparing: {name}")
        print(f"{'-'*80}")

        if fms_tensor.shape != hf_tensor.shape:
            print("❌ Shape mismatch!")
            print(f"   FMS: {tuple(fms_tensor.shape)}")
            print(f"   HF:  {tuple(hf_tensor.shape)}")
            return {"passed": False, "reason": "shape_mismatch"}

        fms_cpu = fms_tensor.detach().cpu().float()
        hf_cpu = hf_tensor.detach().cpu().float()

        diff = (fms_cpu - hf_cpu).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        passed = torch.allclose(fms_cpu, hf_cpu, atol=atol, rtol=rtol)
        print(f"max_abs_diff={max_diff:.3e} mean_abs_diff={mean_diff:.3e} -> {passed}")

        return {"passed": passed, "max_diff": max_diff, "mean_diff": mean_diff}

    def _squeeze_single_patch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Some processors return (B, N, C, H, W) with N=1.
        if pixel_values.dim() == 5 and pixel_values.shape[1] == 1:
            return pixel_values.squeeze(1)
        return pixel_values

    def test_preprocessing(self) -> bool:
        print(f"\n{'='*80}")
        print("PHASE 0: Preprocessing Parity")
        print(f"{'='*80}")

        hf_inputs = self.hf_processor(images=self.test_image, return_tensors="pt")
        hf_pixels = self._squeeze_single_patch(hf_inputs["pixel_values"])

        fms_pre = self.fms_preprocessor_factory(self.checkpoint_name)
        fms_out = fms_pre.preprocess(self.test_image)
        fms_pixels = self._squeeze_single_patch(fms_out["pixel_values"])

        result = self.compare_tensors(fms_pixels, hf_pixels, "pixel_values", atol=1e-6)
        self.results["preprocessing"] = result
        return bool(result["passed"])

    def test_vision_encoder(self) -> bool:
        print(f"\n{'='*80}")
        print("PHASE 1.1: Vision Encoder Parity")
        print(f"{'='*80}")

        hf_inputs = self.hf_processor(images=self.test_image, return_tensors="pt")
        pixels = hf_inputs["pixel_values"].to(self.device)

        if pixels.dim() == 5:
            b, n, c, h, w = pixels.shape
            pixels_flat = pixels.view(b * n, c, h, w)
        else:
            pixels_flat = pixels

        with torch.no_grad():
            hf_out = self.hf_model.model.vision_model(pixels_flat)
            hf_feats = hf_out.last_hidden_state if hasattr(hf_out, "last_hidden_state") else hf_out

            fms_feats, _ = self.fms_model.vision_tower(pixels_flat)

        # Note: FMS SiglipVision uses `nn.GELU` modules (see `fms.models.siglip_vision`)
        # which can introduce small numeric differences vs HF's implementation.
        result = self.compare_tensors(
            fms_feats, hf_feats, "vision_last_hidden", atol=3e-4, rtol=3e-4
        )
        self.results["vision_encoder"] = result
        return bool(result["passed"])

    def test_connector(self) -> bool:
        print(f"\n{'='*80}")
        print("PHASE 1.2: Connector Parity")
        print(f"{'='*80}")

        hf_inputs = self.hf_processor(images=self.test_image, return_tensors="pt")
        pixels = hf_inputs["pixel_values"].to(self.device)

        if pixels.dim() == 5:
            b, n, c, h, w = pixels.shape
            pixels_flat = pixels.view(b * n, c, h, w)
        else:
            pixels_flat = pixels

        with torch.no_grad():
            hf_vision_out = self.hf_model.model.vision_model(pixels_flat)
            vision_feats = (
                hf_vision_out.last_hidden_state
                if hasattr(hf_vision_out, "last_hidden_state")
                else hf_vision_out
            )

            hf_conn = self.hf_model.model.connector(vision_feats)

            grid = int(vision_feats.shape[1] ** 0.5)
            fms_conn = self.fms_model.connector(vision_feats, H=grid, W=grid)

        result = self.compare_tensors(fms_conn, hf_conn, "connector_tokens")
        self.results["connector"] = result
        return bool(result["passed"])

    def test_forward(self) -> bool:
        print(f"\n{'='*80}")
        print("PHASE 2.1: Forward Pass Parity")
        print(f"{'='*80}")

        prompt = "Describe the image: <image>."
        hf_inputs = self.hf_processor(
            text=prompt,
            images=self.test_image,
            return_tensors="pt",
        )
        hf_inputs = {k: v.to(self.device) for k, v in hf_inputs.items()}

        with torch.no_grad():
            hf_out = self.hf_model(**hf_inputs)
            hf_logits = hf_out.logits[:, -1:, :]

            fms_out = self.fms_model(
                input_ids=hf_inputs["input_ids"],
                attention_mask=hf_inputs.get("attention_mask"),
                pixel_values=hf_inputs.get("pixel_values"),
            )
            fms_logits = fms_out["logits"][:, -1:, :]

        result = self.compare_tensors(
            fms_logits, hf_logits, "logits(last_token)", atol=5e-5, rtol=5e-5
        )
        self.results["forward"] = result
        return bool(result["passed"])

    def test_multi_image_forward(self) -> bool:
        print(f"\n{'='*80}")
        print("PHASE 2.1b: Multi-Image Forward Parity")
        print(f"{'='*80}")

        prompt = "Compare: <image> then <image>."
        hf_inputs = self.hf_processor(
            text=prompt,
            images=[self.test_image, self.second_image],
            return_tensors="pt",
        )
        hf_inputs = {k: v.to(self.device) for k, v in hf_inputs.items()}

        with torch.no_grad():
            hf_out = self.hf_model(**hf_inputs)
            hf_logits = hf_out.logits[:, -1:, :]

            fms_out = self.fms_model(
                input_ids=hf_inputs["input_ids"],
                attention_mask=hf_inputs.get("attention_mask"),
                pixel_values=hf_inputs.get("pixel_values"),
            )
            fms_logits = fms_out["logits"][:, -1:, :]

        result = self.compare_tensors(
            fms_logits,
            hf_logits,
            "logits(last_token,multi_image)",
            atol=5e-5,
            rtol=5e-5,
        )
        self.results["multi_image_forward"] = result
        return bool(result["passed"])

    def test_generation(self) -> bool:
        print(f"\n{'='*80}")
        print("PHASE 2.2: Generation Parity")
        print(f"{'='*80}")

        prompt = "Describe the image: <image>."
        hf_inputs = self.hf_processor(text=prompt, images=self.test_image, return_tensors="pt")
        hf_inputs = {k: v.to(self.device) for k, v in hf_inputs.items()}

        gen_kwargs = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": False,
            "use_cache": True,
        }
        with torch.no_grad():
            hf_ids = self.hf_model.generate(**hf_inputs, **gen_kwargs)
            fms_ids = self.fms_model.generate(
                input_ids=hf_inputs["input_ids"],
                attention_mask=hf_inputs.get("attention_mask"),
                pixel_values=hf_inputs.get("pixel_values"),
                max_new_tokens=gen_kwargs["max_new_tokens"],
                eos_token_id=getattr(self.hf_processor.tokenizer, "eos_token_id", None),
            )

        same = torch.equal(fms_ids.cpu(), hf_ids.cpu())
        if not same:
            print("❌ token ids mismatch")
            print("HF:", hf_ids[0, -30:].tolist())
            print("FMS:", fms_ids[0, -30:].tolist())
        else:
            print("✓ token ids match")

        self.results["generation"] = {"passed": bool(same)}
        return bool(same)

    def test_multi_image_generation(self) -> bool:
        print(f"\n{'='*80}")
        print("PHASE 2.2b: Multi-Image Generation Parity")
        print(f"{'='*80}")

        prompt = "Compare: <image> then <image>."
        hf_inputs = self.hf_processor(
            text=prompt,
            images=[self.test_image, self.second_image],
            return_tensors="pt",
        )
        hf_inputs = {k: v.to(self.device) for k, v in hf_inputs.items()}

        gen_kwargs = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": False,
            "use_cache": True,
        }
        with torch.no_grad():
            hf_ids = self.hf_model.generate(**hf_inputs, **gen_kwargs)
            fms_ids = self.fms_model.generate(
                input_ids=hf_inputs["input_ids"],
                attention_mask=hf_inputs.get("attention_mask"),
                pixel_values=hf_inputs.get("pixel_values"),
                max_new_tokens=gen_kwargs["max_new_tokens"],
                eos_token_id=getattr(self.hf_processor.tokenizer, "eos_token_id", None),
            )

        same = torch.equal(fms_ids.cpu(), hf_ids.cpu())
        if not same:
            print("❌ token ids mismatch (multi-image)")
            print("HF:", hf_ids[0, -30:].tolist())
            print("FMS:", fms_ids[0, -30:].tolist())
        else:
            print("✓ token ids match (multi-image)")

        self.results["multi_image_generation"] = {"passed": bool(same)}
        return bool(same)

    def run(self, phase: str) -> int:
        phases = {
            "preprocessing": [self.test_preprocessing],
            "vision": [self.test_vision_encoder],
            "connector": [self.test_connector],
            "forward": [self.test_forward, self.test_multi_image_forward],
            "generate": [self.test_generation, self.test_multi_image_generation],
            "all": [
                self.test_preprocessing,
                self.test_vision_encoder,
                self.test_connector,
                self.test_forward,
                self.test_multi_image_forward,
                self.test_generation,
                self.test_multi_image_generation,
            ],
        }
        if phase not in phases:
            raise ValueError(f"Unknown phase: {phase}. Choose from {sorted(phases.keys())}")

        ok = True
        for fn in phases[phase]:
            ok = fn() and ok

        print("\n" + "=" * 80)
        print(f"RESULT: {'PASS' if ok else 'FAIL'}")
        print("=" * 80)
        return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fms-path", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default="HuggingFaceTB/SmolVLM-256M-Instruct")
    ap.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="Optional HF git revision (tag/branch/SHA). If set, downloads a snapshot and loads HF+FMS from it.",
    )
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["preprocessing", "vision", "connector", "forward", "generate", "all"],
    )
    args = ap.parse_args()

    setup_fms_import(args.fms_path)
    checkpoint_path, checkpoint_label = _resolve_checkpoint(args.checkpoint, args.hf_revision)
    tester = ParityTester(
        checkpoint_name=checkpoint_path,
        checkpoint_label=checkpoint_label,
        device=args.device,
        seed=args.seed,
        default_atol=args.atol,
        default_rtol=args.rtol,
        max_new_tokens=args.max_new_tokens,
    )
    return tester.run(args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
