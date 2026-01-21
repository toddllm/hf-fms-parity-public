# Idefics3 Parity Validation Suite

Portable, standalone test suite for validating parity between the FMS `idefics3` implementation and the HuggingFace `SmolVLM` reference.

What “parity” means here:
- **Numerical parity within tolerance** for intermediate tensors/logits (via `torch.allclose`).
- **Exact match** for greedy generation token IDs.

## Reproducible quickstart (recommended)

This parity suite is **sensitive to Hugging Face dependency versions**. For reproducible results, use a clean virtualenv and pin a known-good HF stack:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools

# Known-good versions (cleanroom validated)
python -m pip install -U \
  "transformers==4.51.3" \
  "huggingface_hub==0.36.0" \
  accelerate pillow numpy requests

# Install this parity suite
python -m pip install -e .
```

Then run against a local `foundation-model-stack` checkout (recommended):

```bash
# This parity suite is intended to be run against the *review branch* in the toddllm FMS fork
# (it may not be merged upstream yet).
git clone git@github.com:toddllm/foundation-model-stack.git
cd foundation-model-stack
git checkout review/idefics3
pip install -e .
cd ..

python verify_setup.py --fms-path ./foundation-model-stack
python test_parity.py --fms-path ./foundation-model-stack --phase all --device cpu
```

Notes:
- This suite uses `AutoModelForVision2Seq` which is deprecated in newer Transformers releases; pinning helps avoid breaking API changes.
- First run will download `HuggingFaceTB/SmolVLM-256M-Instruct` (~500MB) into your HF cache.

## Installation

This suite is designed to work with an editable install of `foundation-model-stack`:

```bash
# 1. Clone and install FMS
git clone https://github.com/foundation-model-stack/foundation-model-stack
cd foundation-model-stack
pip install -e .

# 2. Clone this repo (or use it as a subdirectory)
cd ..
git clone <this-repo> hf-fms-parity
cd hf-fms-parity

# 3. Install parity suite dependencies
pip install -e .
```

## Usage

### Basic Usage (with editable FMS install)

```bash
# Run all parity tests
python test_parity.py --phase all

# Run specific phases
python test_parity.py --phase vision
python test_parity.py --phase connector
python test_parity.py --phase forward
python test_parity.py --phase generate
```

### Advanced Usage

```bash
# Specify custom FMS path (if not using pip install -e)
python test_parity.py --fms-path /path/to/foundation-model-stack --phase all

# Choose a different HF checkpoint (default: HuggingFaceTB/SmolVLM-256M-Instruct)
python test_parity.py --checkpoint HuggingFaceTB/SmolVLM-256M-Instruct --phase all

# Pin to a specific HF revision (tag/branch/SHA) for long-term reproducibility
python test_parity.py --checkpoint HuggingFaceTB/SmolVLM-256M-Instruct --hf-revision <rev> --phase all

# Customize tolerances
# Note: some phases use tuned, explicit tolerances; --atol/--rtol are defaults used where a phase does not override them.
python test_parity.py --phase connector --atol 1e-6 --rtol 1e-5

# Use GPU
python test_parity.py --device cuda --phase all

# Customize generation test
python test_parity.py --phase generate --max-new-tokens 50

# Set random seed
python test_parity.py --seed 123 --phase all
```

## Test Phases

| Phase | Description | Tolerance |
|-------|-------------|-----------|
| `preprocessing` | Image preprocessing | atol=1e-6 |
| `vision` | Vision encoder (SigLIP) | atol=3e-4, rtol=3e-4 |
| `connector` | Pixel-unshuffle + projection | atol=1e-5 |
| `forward` | End-to-end forward pass (last-token logits) | atol=5e-5, rtol=5e-5 |
| `generate` | Text generation | Exact match (greedy) |
| `all` | Run all of the above | - |

## Expected Results

When run against the FMS implementation in the PR branch, all tests should **PASS** (within the phase tolerances above):

```
✅ PASSED: Preprocessing
✅ PASSED: Vision Encoder  
✅ PASSED: Connector
✅ PASSED: Forward Pass
✅ PASSED: Generation
```

## ⚠️ Important Notes

### First Run
The **first time** you run the parity suite, it will:
1. Download the HuggingFace `SmolVLM-256M-Instruct` model (~500MB)
2. Cache it in `~/.cache/huggingface/`
3. This may take 5-15 minutes depending on your internet connection

**This is expected behavior.** Subsequent runs will be much faster.

### Verification
Before running the full parity suite, verify your setup:
```bash
# Quick verification (< 1 minute)
python verify_setup.py --fms-path /path/to/foundation-model-stack

# If all tests pass, proceed with parity
python test_parity.py --fms-path /path/to/foundation-model-stack --phase all
```

### Layer-wise Debugging (HF ↔ FMS traces)

When end-to-end outputs are wrong but the model runs, capture per-layer I/O traces and compare offline:

```bash
python layerwise_trace.py --impl hf --model <hf_id> --prompt "<prompt>" --out /tmp/hf.pkl --mode generate --first-pass-only
python layerwise_trace.py --impl fms --variant <hf_id> --prompt "<prompt>" --out /tmp/fms.pkl --mode generate --first-pass-only

# Create a simple prefix mapping JSON (HF module prefix -> FMS module prefix), then:
python layerwise_compare.py --hf /tmp/hf.pkl --fms /tmp/fms.pkl --map /tmp/hf_to_fms_map.json --compare outputs --metric max_abs
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.1.0
- Transformers >= 4.35.0
- FMS `foundation-model-stack` with `idefics3` module

See `requirements.txt` for full dependencies.

## Portability

This suite is designed to be:
- **System-agnostic**: No hard-coded paths
- **Reproducible**: Fixed random seeds
- **Configurable**: CLI flags for all parameters
- **Standalone**: Can be checked out and run on any system with FMS installed

## Directory Structure

```
idefics3-parity/
├── test_parity.py          # Main test script
├── pyproject.toml          # Pip installation config
├── requirements.txt        # Dependencies
├── README.md               # This file
└── .gitignore              # Git ignore patterns
```

## Troubleshooting

### FMS not found

If you see `ImportError: FMS not found`, either:
1. Install FMS: `pip install -e /path/to/foundation-model-stack`
2. Use `--fms-path`: `python test_parity.py --fms-path /path/to/foundation-model-stack`

### Transformers / HF version drift
If you see large parity deltas (especially in vision) or import errors involving `transformers.utils`, re-create your venv and pin the known-good versions:

```bash
python -m pip install -U "transformers==4.51.3" "huggingface_hub==0.36.0"
```

### Pytest import mode note (FMS repo)
If you run the FMS unit tests in the same environment and see Triton/import-path collisions, prefer:

```bash
python -m pytest -q --import-mode=importlib tests/models/test_idefics3.py
```

### GPU memory issues

For testing on GPU with limited memory, use CPU for component tests and GPU only for end-to-end:

```bash
python test_parity.py --phase vision --device cpu
python test_parity.py --phase forward --device cuda
```

### Tolerance failures

If tests fail due to tolerance, you can adjust:

```bash
python test_parity.py --phase forward --atol 1e-2 --rtol 1e-2
```

