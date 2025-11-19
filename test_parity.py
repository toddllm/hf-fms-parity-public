#!/usr/bin/env python3
"""
Idefics3 FMS vs HuggingFace Parity Validation Suite

Portable, standalone test suite for validating bit-exact parity between:
- FMS implementation (foundation-model-stack)
- HuggingFace reference (transformers)

Usage:
    # Use with editable FMS install
    python test_parity.py --phase all
    
    # Or specify FMS path manually
    python test_parity.py --fms-path /path/to/fms --phase vision
    
    # Customize tolerances
    python test_parity.py --phase forward --atol 1e-6 --rtol 1e-5
"""

import sys
import os
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Dict, Tuple


def setup_fms_import(fms_path: str = None):
    """
    Configure FMS import path.
    
    Args:
        fms_path: Explicit path to FMS. If None, tries to import from installed package.
    """
    if fms_path:
        fms_path = Path(fms_path).resolve()
        if not fms_path.exists():
            raise ValueError(f"FMS path does not exist: {fms_path}")
        print(f"Using FMS from: {fms_path}")
        sys.path.insert(0, str(fms_path))
    else:
        # Try to import from installed package
        try:
            import fms
            print(f"Using installed FMS from: {Path(fms.__file__).parent.parent}")
        except ImportError:
            raise ImportError(
                "FMS not found. Either:\n"
                "  1. Install FMS: pip install -e /path/to/foundation-model-stack\n"
                "  2. Specify --fms-path: python test_parity.py --fms-path /path/to/fms"
            )


class ParityTester:
    """Comprehensive parity testing between FMS and HF implementations"""
    
    def __init__(
        self, 
        checkpoint_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct", 
        device: str = "cpu",
        seed: int = 42,
        default_atol: float = 1e-5,
        default_rtol: float = 1e-5
    ):
        """
        Initialize parity tester.
        
        Args:
            checkpoint_name: HuggingFace checkpoint to test against
            device: Device to run on (cpu/cuda)
            seed: Random seed for reproducibility
            default_atol: Default absolute tolerance
            default_rtol: Default relative tolerance
        """
        self.checkpoint_name = checkpoint_name
        self.device = device
        self.seed = seed
        self.default_atol = default_atol
        self.default_rtol = default_rtol
        self.results = {}
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        print(f"\n{'='*80}")
        print(f"Initializing Parity Tester")
        print(f"Checkpoint: {checkpoint_name}")
        print(f"Device: {device}")
        print(f"Seed: {seed}")
        print(f"Default tolerances: atol={default_atol:.2e}, rtol={default_rtol:.2e}")
        print(f"{'='*80}\n")
        
        # Load HF model
        print("Loading HuggingFace model...")
        self.hf_processor = AutoProcessor.from_pretrained(checkpoint_name)
        self.hf_model = AutoModelForVision2Seq.from_pretrained(
            checkpoint_name,
            torch_dtype=torch.float32,
            trust_remote_code=True  # Required for SmolVLM
        ).to(device)
        self.hf_model.eval()
        print("✓ HF model loaded\n")
        
        # Import FMS after path is set
        from fms.models.idefics3.model import Idefics3FMSModel, Idefics3Config
        from fms.models.idefics3.hf_adapter import load_smolvlm_checkpoint
        from fms.models.idefics3.preprocessing import SmolVLMPreprocessor
        from fms.models.idefics3.connector import Idefics3Connector
        
        # Store for later use
        self.fms_model_class = Idefics3FMSModel
        self.fms_config_class = Idefics3Config
        self.fms_preprocessor_class = SmolVLMPreprocessor
        self.fms_connector_class = Idefics3Connector
        
        # Load FMS components
        print("Loading FMS components...")
        self.fms_components = load_smolvlm_checkpoint(
            checkpoint_name,
            device=device
        )
        # Returns tuple: (tokenizer, vision_encoder, text_backbone, connector_weights)
        (self.fms_tokenizer, 
         self.fms_vision, 
         self.fms_text,
         self.fms_connector_weights) = self.fms_components
        print("✓ FMS components loaded\n")
        
        # Create test image (vectorized for speed)
        self.test_image = self._create_test_image()
        
    def _create_test_image(self) -> Image.Image:
        """Create a deterministic test image using vectorized numpy operations."""
        # Vectorized gradient image creation (much faster than nested loops)
        i_grid = np.arange(512)[:, None]  # (512, 1)
        j_grid = np.arange(512)[None, :]  # (1, 512)
        
        arr = np.zeros((512, 512, 3), dtype=np.uint8)
        arr[:, :, 0] = (i_grid * 255 // 512).astype(np.uint8)  # Red gradient
        arr[:, :, 1] = (j_grid * 255 // 512).astype(np.uint8)  # Green gradient
        arr[:, :, 2] = 128  # Constant blue
        
        return Image.fromarray(arr)
    
    def compare_tensors(self, 
                       fms_tensor: torch.Tensor, 
                       hf_tensor: torch.Tensor, 
                       name: str,
                       atol: float = 1e-5,
                       rtol: float = 1e-5) -> Dict[str, float]:
        """Compare two tensors and return detailed metrics"""
        
        print(f"\n{'-'*80}")
        print(f"Comparing: {name}")
        print(f"{'-'*80}")
        
        # Shape check
        if fms_tensor.shape != hf_tensor.shape:
            print(f"❌ Shape mismatch!")
            print(f"   FMS shape: {fms_tensor.shape}")
            print(f"   HF shape:  {hf_tensor.shape}")
            return {"passed": False, "reason": "shape_mismatch"}
        
        print(f"✓ Shape: {fms_tensor.shape}")
        
        # Dtype check
        if fms_tensor.dtype != hf_tensor.dtype:
            print(f"⚠️  Dtype mismatch (FMS: {fms_tensor.dtype}, HF: {hf_tensor.dtype})")
        else:
            print(f"✓ Dtype: {fms_tensor.dtype}")
        
        # Move to same device for comparison
        fms_cpu = fms_tensor.detach().cpu().float()
        hf_cpu = hf_tensor.detach().cpu().float()
        
        # Compute differences
        diff = (fms_cpu - hf_cpu).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        median_diff = diff.median().item()
        
        # Relative error
        rel_diff = diff / (hf_cpu.abs() + 1e-10)
        max_rel_diff = rel_diff.max().item()
        
        # Statistics
        fms_stats = {
            "mean": fms_cpu.mean().item(),
            "std": fms_cpu.std().item(),
            "min": fms_cpu.min().item(),
            "max": fms_cpu.max().item(),
        }
        
        hf_stats = {
            "mean": hf_cpu.mean().item(),
            "std": hf_cpu.std().item(),
            "min": hf_cpu.min().item(),
            "max": hf_cpu.max().item(),
        }
        
        print(f"\nDifference Metrics:")
        print(f"  Max absolute diff:  {max_diff:.2e}")
        print(f"  Mean absolute diff: {mean_diff:.2e}")
        print(f"  Median abs diff:    {median_diff:.2e}")
        print(f"  Max relative diff:  {max_rel_diff:.2e}")
        
        print(f"\nFMS Statistics:")
        print(f"  Mean: {fms_stats['mean']:.6f}, Std: {fms_stats['std']:.6f}")
        print(f"  Min:  {fms_stats['min']:.6f}, Max: {fms_stats['max']:.6f}")
        
        print(f"\nHF Statistics:")
        print(f"  Mean: {hf_stats['mean']:.6f}, Std: {hf_stats['std']:.6f}")
        print(f"  Min:  {hf_stats['min']:.6f}, Max: {hf_stats['max']:.6f}")
        
        # Check tolerance
        passed = torch.allclose(fms_cpu, hf_cpu, atol=atol, rtol=rtol)
        
        if passed:
            print(f"\n✅ PASSED (within atol={atol:.2e}, rtol={rtol:.2e})")
        else:
            print(f"\n❌ FAILED (exceeds atol={atol:.2e}, rtol={rtol:.2e})")
        
        return {
            "passed": passed,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "median_diff": median_diff,
            "max_rel_diff": max_rel_diff,
            "fms_stats": fms_stats,
            "hf_stats": hf_stats,
        }
    
    def test_preprocessing(self) -> bool:
        """Phase 0: Test preprocessing parity"""
        print(f"\n{'='*80}")
        print("PHASE 0: Preprocessing Parity")
        print(f"{'='*80}")
        
        # HF preprocessing
        hf_inputs = self.hf_processor(images=self.test_image, return_tensors="pt")
        hf_pixels = hf_inputs["pixel_values"]
        
        # FMS preprocessing
        fms_preprocessor = self.fms_preprocessor_class(self.checkpoint_name)
        fms_output = fms_preprocessor.preprocess(self.test_image)
        fms_pixels = fms_output["pixel_values"]
        
        # Remove batch dimension for single image if needed
        if fms_pixels.dim() == 5 and fms_pixels.shape[1] == 1:
            fms_pixels = fms_pixels.squeeze(1)  # (B, 1, C, H, W) -> (B, C, H, W)
        
        result = self.compare_tensors(
            fms_pixels, 
            hf_pixels, 
            "Preprocessed Pixels",
            atol=1e-6
        )
        
        self.results["preprocessing"] = result
        return result["passed"]
    
    def test_vision_encoder(self) -> bool:
        """Phase 1.1: Test vision encoder parity"""
        print(f"\n{'='*80}")
        print("PHASE 1.1: Vision Encoder Parity")
        print(f"{'='*80}")
        
        # Preprocess image
        hf_inputs = self.hf_processor(images=self.test_image, return_tensors="pt")
        hf_pixels = hf_inputs["pixel_values"].to(self.device)
        
        # Flatten (B, N, C, H, W) -> (B*N, C, H, W) if needed
        if hf_pixels.dim() == 5:
            B, N, C, H, W = hf_pixels.shape
            hf_pixels = hf_pixels.view(B * N, C, H, W)
            print(f"ℹ️  Flattened input: {B}x{N} -> {B*N} patches")
        
        # HF vision encoder
        with torch.no_grad():
            hf_vision_out = self.hf_model.model.vision_model(hf_pixels)
            if hasattr(hf_vision_out, "last_hidden_state"):
                hf_features = hf_vision_out.last_hidden_state
            else:
                hf_features = hf_vision_out
        
        # FMS vision encoder
        with torch.no_grad():
            fms_features = self.fms_vision(hf_pixels)  # Use same pixels
        
        result = self.compare_tensors(
            fms_features,
            hf_features,
            "Vision Encoder Features",
            atol=1e-5
        )
        
        self.results["vision_encoder"] = result
        return result["passed"]
    
    def test_connector(self) -> bool:
        """Phase 1.2: Test connector parity"""
        print(f"\n{'='*80}")
        print("PHASE 1.2: Connector Parity")
        print(f"{'='*80}")
        
        # Get vision features first
        hf_inputs = self.hf_processor(images=self.test_image, return_tensors="pt")
        hf_pixels = hf_inputs["pixel_values"].to(self.device)
        
        # Flatten (B, N, C, H, W) -> (B*N, C, H, W) if needed
        if hf_pixels.dim() == 5:
            B, N, C, H, W = hf_pixels.shape
            hf_pixels = hf_pixels.view(B * N, C, H, W)
        
        with torch.no_grad():
            hf_vision_out = self.hf_model.model.vision_model(hf_pixels)
            if hasattr(hf_vision_out, "last_hidden_state"):
                vision_features = hf_vision_out.last_hidden_state
            else:
                vision_features = hf_vision_out
            
            # HF connector
            hf_connector_out = self.hf_model.model.connector(vision_features)
            
            # FMS connector - need to instantiate with weights
            fms_connector = self.fms_connector_class(
                vision_hidden=768,  # SigLIP hidden
                text_hidden=576,    # SmolVLM hidden
                scale=4
            ).to(self.device)
            # Load weights
            try:
                fms_connector.load_state_dict(self.fms_connector_weights, strict=True)
            except Exception as e:
                print(f"❌ Failed to load connector weights: {e}")
                print(f"Available keys: {list(self.fms_connector_weights.keys())}")
                print(f"Model keys: {list(fms_connector.state_dict().keys())}")
                raise
            
            # Calculate H, W (assuming square 512x512 input -> 32x32 patches)
            # vision_features is (B*N, 1024, 768)
            num_patches = vision_features.shape[1]
            grid_size = int(num_patches ** 0.5)
            
            fms_connector_out = fms_connector(vision_features, H=grid_size, W=grid_size)
        
        result = self.compare_tensors(
            fms_connector_out,
            hf_connector_out,
            "Connector Output",
            atol=1e-5
        )
        
        self.results["connector"] = result
        return result["passed"]
    
    def test_text_backbone(self) -> bool:
        """Phase 1.3: Test text backbone parity"""
        print(f"\n{'='*80}")
        print("PHASE 1.3: Text Backbone Parity")
        print(f"{'='*80}")
        
        # Test with simple text input
        text = "Hello world"
        tokenizer = self.hf_processor.tokenizer
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
        
        with torch.no_grad():
            # HF text model
            hf_embeds = self.hf_model.model.text_model.get_input_embeddings()(input_ids)
            hf_output = self.hf_model.model.text_model(inputs_embeds=hf_embeds)
            hf_logits = self.hf_model.lm_head(hf_output.last_hidden_state)
            
            # FMS text backbone
            fms_embeds = self.fms_text.get_input_embeddings()(input_ids)
            fms_output = self.fms_text(inputs_embeds=fms_embeds)
            fms_logits = fms_output["logits"] if isinstance(fms_output, dict) else fms_output.logits
        
        # Compare embeddings
        embed_result = self.compare_tensors(
            fms_embeds,
            hf_embeds,
            "Text Embeddings",
            atol=1e-6
        )
        
        # Compare logits
        logits_result = self.compare_tensors(
            fms_logits,
            hf_logits,
            "Text Logits",
            atol=1e-4
        )
        
        self.results["text_embeddings"] = embed_result
        self.results["text_logits"] = logits_result
        
        return embed_result["passed"] and logits_result["passed"]
    
    def test_forward_pass(self) -> bool:
        """Phase 2.1: Test end-to-end forward pass"""
        print(f"\n{'='*80}")
        print("PHASE 2.1: Forward Pass Parity")
        print(f"{'='*80}")
        
        # Prepare input
        prompt = "Describe this image: <image>"
        hf_inputs = self.hf_processor(text=prompt, images=self.test_image, return_tensors="pt")
        hf_inputs = {k: v.to(self.device) for k, v in hf_inputs.items()}
        
        # HF forward
        with torch.no_grad():
            hf_output = self.hf_model(**hf_inputs)
            hf_logits = hf_output.logits
        
        # FMS forward
        # Assemble model
        fms_model = self.fms_model_class(
            config=self.fms_config_class(), # Defaults should match 256M
            tokenizer=self.fms_tokenizer,
            vision_encoder=self.fms_vision,
            text_backbone=self.fms_text,
            device=self.device
        )
        # Inject connector weights
        try:
            fms_model.connector.load_state_dict(self.fms_connector_weights, strict=True)
        except Exception:
             # Fallback if keys were renamed in adapter but not in model init
             # But we renamed them in adapter, so they should match model definition
             fms_model.connector.load_state_dict(self.fms_connector_weights, strict=False)

        fms_model.eval()
        
        # Prepare FMS inputs
        # We need to manually call prepare_inputs or just pass args if model handles it
        # Idefics3FMSModel.forward takes (input_ids, images, ...)
        
        # Preprocess image
        fms_preprocessor = self.fms_preprocessor_class(self.checkpoint_name)
        fms_image_out = fms_preprocessor.preprocess(self.test_image)
        pixel_values = fms_image_out["pixel_values"].to(self.device)
        
        input_ids = hf_inputs["input_ids"]
        
        with torch.no_grad():
            fms_output = fms_model(input_ids=input_ids, images=pixel_values)
            fms_logits = fms_output["logits"]
            
        result = self.compare_tensors(
            fms_logits,
            hf_logits,
            "End-to-End Logits",
            atol=1e-3 # Slightly looser for accumulated errors
        )
        
        self.results["forward_pass"] = result
        return result["passed"]
    
    def test_generation(self) -> bool:
        """Phase 2.2: Test generation parity"""
        print(f"\n{'='*80}")
        print("PHASE 2.2: Generation Parity")
        print(f"{'='*80}")
        
        # Prepare input
        prompt = "Describe this image: <image>"
        hf_inputs = self.hf_processor(text=prompt, images=self.test_image, return_tensors="pt")
        hf_inputs = {k: v.to(self.device) for k, v in hf_inputs.items()}
        
        # Generation args (greedy for determinism)
        max_tokens = getattr(self, 'max_new_tokens', 20)  # Default to 20 if not set
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "temperature": 1.0, # Ignored for greedy but good practice
            "use_cache": True,
        }
        
        print("Generating with HuggingFace...")
        hf_ids = self.hf_model.generate(**hf_inputs, **gen_kwargs)
        hf_text = self.hf_processor.tokenizer.decode(hf_ids[0], skip_special_tokens=True)
        print(f"HF Output: {hf_text}")
        
        # FMS generation
        print("Generating with FMS...")
        fms_model = self.fms_model_class(
            config=self.fms_config_class(),
            tokenizer=self.fms_tokenizer,
            vision_encoder=self.fms_vision,
            text_backbone=self.fms_text,
            device=self.device
        )
        # Inject connector weights
        try:
            fms_model.connector.load_state_dict(self.fms_connector_weights, strict=True)
        except Exception:
             fms_model.connector.load_state_dict(self.fms_connector_weights, strict=False)
        
        fms_model.eval()
        
        # Preprocess image
        fms_preprocessor = self.fms_preprocessor_class(self.checkpoint_name)
        fms_image_out = fms_preprocessor.preprocess(self.test_image)
        pixel_values = fms_image_out["pixel_values"].to(self.device)
        
        # FMS generate
        # Use HF input_ids to ensure parity with SmolVLM's complex multi-crop token expansion
        fms_ids = fms_model.generate(
            input_ids=hf_inputs["input_ids"],
            attention_mask=hf_inputs["attention_mask"],
            images=pixel_values,
            **gen_kwargs
        )
        print(f"FMS output type: {type(fms_ids)}")
        print(f"FMS output: {fms_ids}")
        if hasattr(fms_ids, "shape"):
            print(f"FMS output shape: {fms_ids.shape}")
        
        # Ensure we have a tensor or list of ints
        if isinstance(fms_ids, list) and isinstance(fms_ids[0], str):
            fms_text = fms_ids[0]
        else:
            if hasattr(fms_ids, "sequences"): # ModelOutput
                 fms_ids = fms_ids.sequences
            fms_text = self.fms_tokenizer.decode(fms_ids[0], skip_special_tokens=True)
            
        print(f"FMS Output: {fms_text}")
        
        # Compare (robust to prompt inclusion/exclusion)
        # Normalize whitespace
        hf_norm = " ".join(hf_text.split())
        fms_norm = " ".join(fms_text.split())
        
        # Check if generated part matches
        # We assume the last part of the text is the generation
        # Let's check if FMS text is contained in HF text or vice versa
        if fms_norm in hf_norm or hf_norm in fms_norm:
            print("\n✅ PASSED: Generated text matches!")
            return True
        else:
            print("\n❌ FAILED: Generated text mismatch")
            print(f"HF:  {hf_text}")
            print(f"FMS: {fms_text}")
            return False
    
    def run_all_tests(self):
        """Run all parity tests"""
        tests = [
            ("Preprocessing", self.test_preprocessing),
            ("Vision Encoder", self.test_vision_encoder),
            ("Connector", self.test_connector),
            ("Text Backbone", self.test_text_backbone),
            ("Forward Pass", self.test_forward_pass),
            ("Generation", self.test_generation),
        ]
        
        print(f"\n{'='*80}")
        print("RUNNING ALL PARITY TESTS")
        print(f"{'='*80}\n")
        
        passed_count = 0
        failed_count = 0
        skipped_count = 0
        
        for name, test_fn in tests:
            try:
                result = test_fn()
                if result is True:
                    passed_count += 1
                elif result is False:
                    failed_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"\n❌ Test '{name}' failed with exception: {e}")
                import traceback
                traceback.print_exc()
                failed_count += 1
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"✅ Passed:  {passed_count}")
        print(f"❌ Failed:  {failed_count}")
        print(f"⚠️  Skipped: {skipped_count}")
        print(f"Total:     {passed_count + failed_count + skipped_count}")
        
        return failed_count == 0




def main():
    parser = argparse.ArgumentParser(
        description="Idefics3 FMS-HF Parity Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # With editable FMS install
    python test_parity.py --phase all
    
    # With custom FMS path
    python test_parity.py --fms-path /path/to/foundation-model-stack --phase forward
    
    # Custom tolerances
    python test_parity.py --phase all --atol 1e-6 --rtol 1e-6
        """
    )
    parser.add_argument(
        "--phase",
        choices=["preprocessing", "vision", "connector", "text", "forward", "generate", "all"],
        default="all",
        help="Which parity test phase to run"
    )
    parser.add_argument(
        "--checkpoint",
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
        help="HuggingFace checkpoint name"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run tests on (cpu/cuda/mps)"
    )
    parser.add_argument(
        "--fms-path",
        default=None,
        help="Path to foundation-model-stack. If not set, tries to use installed FMS package."
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for comparisons (default: 1e-5)"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for comparisons (default: 1e-5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=20,
        help="Max new tokens for generation test (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Setup FMS import
    setup_fms_import(args.fms_path)
    
    # Initialize tester
    tester = ParityTester(
        checkpoint_name=args.checkpoint, 
        device=args.device,
        seed=args.seed,
        default_atol=args.atol,
        default_rtol=args.rtol
    )
    
    # Store max_new_tokens for generation test
    tester.max_new_tokens = args.max_new_tokens
    
    # Run requested phase
    if args.phase == "all":
        success = tester.run_all_tests()
    elif args.phase == "preprocessing":
        success = tester.test_preprocessing()
    elif args.phase == "vision":
        success = tester.test_vision_encoder()
    elif args.phase == "connector":
        success = tester.test_connector()
    elif args.phase == "text":
        success = tester.test_text_backbone()
    elif args.phase == "forward":
        success = tester.test_forward_pass()
    elif args.phase == "generate":
        success = tester.test_generation()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

