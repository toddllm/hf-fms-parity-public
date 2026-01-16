#!/usr/bin/env python3
"""
Portable Parity Suite - Final Verification Test

This test verifies that the portable parity suite works correctly
without any hard-coded paths.
"""

import sys
import os
from pathlib import Path
import argparse

def test_fms_auto_detection():
    """Test that FMS can be auto-detected from pip install."""
    print("\n" + "="*70)
    print("TEST 1: FMS Auto-Detection")
    print("="*70)
    
    # Import the setup function
    sys.path.insert(0, str(Path(__file__).parent))
    import test_parity
    
    # This should work without --fms-path
    try:
        test_parity.setup_fms_import(None)
        print("✅ PASSED: FMS auto-detected from installed package")
        return True
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        return False

def test_fms_imports(fms_path: str | None):
    """Test that all FMS modules can be imported."""
    print("\n" + "="*70)
    print("TEST 2: FMS Module Imports")
    print("="*70)
    
    try:
        # Ensure we import the intended FMS checkout (if provided).
        sys.path.insert(0, str(Path(__file__).parent))
        import test_parity
        test_parity.setup_fms_import(fms_path)

        from fms.models.idefics3 import (
            Idefics3,
            Idefics3Config,
            load_smolvlm_preprocessor,
        )
        print("✅ PASSED: All FMS modules imported successfully")
        print("   - Idefics3")
        print("   - Idefics3Config")
        print("   - load_smolvlm_preprocessor")
        return True
    except ImportError as e:
        print(f"❌ FAILED: {e}")
        return False

def test_cli_args():
    """Test that CLI argument parsing works."""
    print("\n" + "="*70)
    print("TEST 3: CLI Argument Parsing")
    print("="*70)
    
    import subprocess
    
    # Test help
    result = subprocess.run(
        [sys.executable, "test_parity.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent
    )
    
    if result.returncode == 0:
        required_args = ["--fms-path", "--atol", "--rtol", "--seed", "--phase", "--device"]
        missing = []
        for arg in required_args:
            if arg not in result.stdout:
                missing.append(arg)
        
        if not missing:
            print("✅ PASSED: All CLI arguments present")
            for arg in required_args:
                print(f"   - {arg}")
            return True
        else:
            print(f"❌ FAILED: Missing arguments: {missing}")
            return False
    else:
        print(f"❌ FAILED: Help command failed with code {result.returncode}")
        return False

def test_dependencies():
    """Test that all required dependencies are installed."""
    print("\n" + "="*70)
    print("TEST 4: Required Dependencies")
    print("="*70)
    
    required = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "PIL": "Pillow",
        "numpy": "NumPy",
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} MISSING")
            all_ok = False
    
    if all_ok:
        print("\n✅ PASSED: All dependencies installed")
        return True
    else:
        print("\n❌ FAILED: Some dependencies missing")
        return False

def main():
    ap = argparse.ArgumentParser(description="Verify parity suite portability.")
    ap.add_argument(
        "--fms-path",
        type=str,
        default=None,
        help="Optional path to a foundation-model-stack checkout (if not installed editable).",
    )
    args = ap.parse_args()

    print("\n" + "="*70)
    print("PORTABLE PARITY SUITE - VERIFICATION TEST")
    print("="*70)
    print("\nThis test verifies the suite works without hard-coded paths.")
    
    results = []
    if args.fms_path is None:
        results.append(("FMS Auto-Detection", test_fms_auto_detection()))
    else:
        print("\nSkipping auto-detection (using --fms-path)")
    results.append(("FMS Module Imports", test_fms_imports(args.fms_path)))
    results.append(("CLI Arguments", test_cli_args()))
    results.append(("Dependencies", test_dependencies()))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED - SUITE IS PORTABLE!")
        print("="*70)
        print("\nYou can now:")
        print("  1. Run parity tests: python test_parity.py --phase all")
        print("  2. Commit to git and push to remote")
        print("  3. Clone on another system and use with editable FMS install")
        print("="*70)
        return 0
    else:
        print("\n❌ Some tests failed. See above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
