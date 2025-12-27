#!/bin/bash
# Build and test script for rocWMMA patch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Building rocWMMA Patch for PyTorch"
echo "=========================================="

# Check for ROCm
if [ -z "$ROCM_PATH" ]; then
    ROCM_PATH="/opt/rocm"
fi

if [ ! -d "$ROCM_PATH" ]; then
    echo "❌ ROCm not found at $ROCM_PATH"
    echo "   Please set ROCM_PATH environment variable"
    exit 1
fi

echo "✅ ROCm found at: $ROCM_PATH"

# Check for hipcc
if ! command -v hipcc &> /dev/null; then
    echo "❌ hipcc not found in PATH"
    exit 1
fi

echo "✅ hipcc found: $(which hipcc)"

# Check for Python and PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    echo "⚠️  PyTorch not found in system Python"
    echo "   Attempting to use virtual environment..."
    
    if [ -f "../../.venv/bin/activate" ]; then
        source ../../.venv/bin/activate
        echo "✅ Activated virtual environment"
    else
        echo "❌ No virtual environment found"
        echo "   Please install PyTorch or set up a virtual environment"
        exit 1
    fi
fi

# Check PyTorch
if ! python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    echo "❌ PyTorch not available"
    exit 1
fi

echo "✅ PyTorch available"

# Build the extension
echo ""
echo "Building extension..."
echo "---------------------"

# Try pip install first (recommended)
if python3 -m pip install -e . --no-build-isolation 2>&1 | tee /tmp/build.log; then
    echo ""
    echo "✅ Build successful with pip install"
    BUILD_SUCCESS=true
else
    echo ""
    echo "⚠️  pip install failed, trying setup.py..."
    
    # Fallback to setup.py
    if python3 setup.py build_ext --inplace 2>&1 | tee -a /tmp/build.log; then
        echo ""
        echo "✅ Build successful with setup.py"
        BUILD_SUCCESS=true
    else
        echo ""
        echo "❌ Build failed"
        echo "   Check /tmp/build.log for details"
        exit 1
    fi
fi

# Test import
echo ""
echo "Testing import..."
echo "-----------------"

if python3 -c "import wmma_ops; print('✅ wmma_ops imported successfully')" 2>&1; then
    echo "✅ Module import test passed"
else
    echo "❌ Module import test failed"
    echo "   The extension may not be in the Python path"
    exit 1
fi

# Run tests if test script exists
if [ -f "test_rocwmma_patch.py" ]; then
    echo ""
    echo "Running test suite..."
    echo "---------------------"
    python3 test_rocwmma_patch.py
else
    echo ""
    echo "⚠️  test_rocwmma_patch.py not found, skipping tests"
fi

echo ""
echo "=========================================="
echo "✅ Build and test completed successfully!"
echo "=========================================="






