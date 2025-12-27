#!/bin/bash
# Script to analyze ISA from compiled kernel

set -e

echo "=========================================="
echo "ISA Inspection for WMMA Kernel"
echo "=========================================="

# Find assembly files
ASM_FILES=$(find . -name "*.s" -type f | grep -E "(wmma|gemm)" | head -1)

if [ -z "$ASM_FILES" ]; then
    echo "❌ No assembly files found. Make sure to compile with -save-temps"
    exit 1
fi

echo "📄 Found assembly file: $ASM_FILES"
echo ""

# Analyze for WMMA instructions
echo "=========================================="
echo "1. WMMA Instruction Usage"
echo "=========================================="
if grep -q "wmma\|WMMA" "$ASM_FILES"; then
    echo "✅ WMMA instructions found:"
    grep -i "wmma" "$ASM_FILES" | head -10
else
    echo "❌ No WMMA instructions found!"
fi
echo ""

# Analyze for dual-issue instructions
echo "=========================================="
echo "2. Dual-Issue Instructions (v_dual_*)"
echo "=========================================="
if grep -q "v_dual" "$ASM_FILES"; then
    echo "✅ Dual-issue instructions found:"
    grep "v_dual" "$ASM_FILES" | head -10
else
    echo "⚠️  No dual-issue instructions found (may indicate optimization opportunity)"
fi
echo ""

# Analyze vector operations
echo "=========================================="
echo "3. Vector Load/Store Instructions"
echo "=========================================="
echo "Vector loads (global_load_*):"
grep -E "global_load_(b32|b64|b128)" "$ASM_FILES" | head -10 || echo "  (none found)"
echo ""
echo "Vector stores (global_store_*):"
grep -E "global_store_(b32|b64|b128)" "$ASM_FILES" | head -10 || echo "  (none found)"
echo ""

# Analyze LDS operations
echo "=========================================="
echo "4. LDS (Shared Memory) Operations"
echo "=========================================="
echo "LDS loads:"
grep -E "ds_read" "$ASM_FILES" | head -10 || echo "  (none found)"
echo ""
echo "LDS stores:"
grep -E "ds_write" "$ASM_FILES" | head -10 || echo "  (none found)"
echo ""

# Analyze VALU operations
echo "=========================================="
echo "5. VALU (Vector ALU) Operations"
echo "=========================================="
echo "FMA operations:"
grep -E "v_fmac|v_fma" "$ASM_FILES" | head -10 || echo "  (none found)"
echo ""
echo "ADD operations:"
grep -E "v_add_f32|v_add_u32" "$ASM_FILES" | head -10 || echo "  (none found)"
echo ""

# Count instruction types
echo "=========================================="
echo "6. Instruction Statistics"
echo "=========================================="
TOTAL_INST=$(grep -c "^[[:space:]]*v_\|^[[:space:]]*s_\|^[[:space:]]*global_\|^[[:space:]]*ds_" "$ASM_FILES" 2>/dev/null || echo "0")
WMMA_COUNT=$(grep -ci "wmma" "$ASM_FILES" || echo "0")
DUAL_COUNT=$(grep -c "v_dual" "$ASM_FILES" || echo "0")
VLOAD_COUNT=$(grep -cE "global_load_(b32|b64|b128)" "$ASM_FILES" || echo "0")
LDS_COUNT=$(grep -cE "ds_read|ds_write" "$ASM_FILES" || echo "0")

echo "Total instructions: $TOTAL_INST"
echo "WMMA instructions: $WMMA_COUNT"
echo "Dual-issue instructions: $DUAL_COUNT"
echo "Vector loads: $VLOAD_COUNT"
echo "LDS operations: $LDS_COUNT"
echo ""

# Look for optimization opportunities
echo "=========================================="
echo "7. Optimization Opportunities"
echo "=========================================="

if [ "$DUAL_COUNT" -eq 0 ]; then
    echo "⚠️  No dual-issue instructions found"
    echo "   Recommendation: Optimize instruction scheduling for dual-issue"
fi

if [ "$WMMA_COUNT" -eq 0 ]; then
    echo "❌ CRITICAL: No WMMA instructions found!"
    echo "   The kernel may not be using WMMA intrinsics correctly"
fi

# Check for scalar operations that could be vectorized
SCALAR_LOADS=$(grep -c "global_load_b32" "$ASM_FILES" || echo "0")
if [ "$SCALAR_LOADS" -gt 100 ]; then
    echo "⚠️  Many scalar loads found ($SCALAR_LOADS)"
    echo "   Recommendation: Use vectorized loads (b64, b128)"
fi

echo ""
echo "=========================================="
echo "Analysis Complete"
echo "=========================================="

