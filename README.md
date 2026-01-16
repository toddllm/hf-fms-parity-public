# Idefics3 Parity Validation Suite

Portable, standalone test suite for validating bit-exact parity between the FMS `idefics3` implementation and the HuggingFace `SmolVLM` reference.

## Installation

This suite is designed to work with an editable install of `foundation-model-stack`:

```bash
# 1. Clone and install FMS
git clone https://github.com/foundation-model-stack/foundation-model-stack
cd foundation-model-stack
pip install -e .

# 2. Clone this repo (or use it as a subdirectory)
cd ..
git clone <this-repo>
cd idefics3-parity

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

# Customize tolerances
python test_parity.py --phase forward --atol 1e-6 --rtol 1e-5

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
| `vision` | Vision encoder (SigLIP) | atol=1e-5 |
| `connector` | Pixel-unshuffle + projection | atol=1e-5 |
| `text` | Text backbone (Llama) | atol=1e-6 (embeds), atol=1e-4 (logits) |
| `forward` | End-to-end forward pass | atol=1e-3 |
| `generate` | Text generation | Exact match (greedy) |
| `all` | Run all of the above | - |

## Expected Results

When run against the FMS implementation in the PR branch, all tests should **PASS** with bit-exact or near-exact parity:

```
✅ PASSED: Preprocessing
✅ PASSED: Vision Encoder  
✅ PASSED: Connector
✅ PASSED: Text Backbone
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
python verify_setup.py

# If all tests pass, proceed with parity
python test_parity.py --phase all
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

## License

Apache 2.0
