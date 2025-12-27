# WMMA Fragment Layout for RDNA3 (gfx1151)

Technical reference for Wave Matrix Multiply-Accumulate (WMMA) fragment layouts on AMD RDNA3.5 architecture.

---

## Overview

The `v_wmma_f32_16x16x16_f16` instruction performs a 16×16×16 matrix multiply-accumulate:
- **A Matrix**: 16×16 (M×K), FP16 input
- **B Matrix**: 16×16 (K×N), FP16 input  
- **C/D Matrix**: 16×16 (M×N), FP32 accumulator

Each wave (32 threads) cooperatively loads and processes one 16×16 tile.

---

## Fragment Register Layout

### A Matrix Fragment (16×16, FP16)

**Hardware expects**: `a_frag[i] = A[i][lane % 16]`

| Fragment Index | Matrix Element | Description |
|----------------|----------------|-------------|
| `a_frag[0]` | `A[0][lane % 16]` | Row 0, column = lane |
| `a_frag[1]` | `A[1][lane % 16]` | Row 1, column = lane |
| ... | ... | ... |
| `a_frag[15]` | `A[15][lane % 16]` | Row 15, column = lane |

**Key insight**: Each lane loads one **column** of the A matrix (all 16 rows of that column).

**Lane replication**: Lanes 0-15 and lanes 16-31 must contain identical data.

### B Matrix Fragment (16×16, FP16)

**Hardware expects**: `b_frag[k] = B[k][lane % 16]`

| Fragment Index | Matrix Element | Description |
|----------------|----------------|-------------|
| `b_frag[0]` | `B[0][lane % 16]` | Row 0, column = lane |
| `b_frag[1]` | `B[1][lane % 16]` | Row 1, column = lane |
| ... | ... | ... |
| `b_frag[15]` | `B[15][lane % 16]` | Row 15, column = lane |

**Key insight**: Each lane loads one **column** of the B matrix.

**Lane replication**: Lanes 0-15 and lanes 16-31 must contain identical data.

### C/D Matrix Fragment (16×16, FP32)

**Hardware produces**: `c_frag[i] = C[i*2 + (lane/16)][lane % 16]`

| Fragment Index | Matrix Element (lanes 0-15) | Matrix Element (lanes 16-31) |
|----------------|-----------------------------|-----------------------------|
| `c_frag[0]` | `C[0][lane]` | `C[1][lane-16]` |
| `c_frag[1]` | `C[2][lane]` | `C[3][lane-16]` |
| `c_frag[2]` | `C[4][lane]` | `C[5][lane-16]` |
| `c_frag[3]` | `C[6][lane]` | `C[7][lane-16]` |
| `c_frag[4]` | `C[8][lane]` | `C[9][lane-16]` |
| `c_frag[5]` | `C[10][lane]` | `C[11][lane-16]` |
| `c_frag[6]` | `C[12][lane]` | `C[13][lane-16]` |
| `c_frag[7]` | `C[14][lane]` | `C[15][lane-16]` |

**Key insight**: 
- Lanes 0-15 cover **even rows** (0, 2, 4, ..., 14)
- Lanes 16-31 cover **odd rows** (1, 3, 5, ..., 15)
- Each lane covers one column

---

## Loading Fragments from LDS

### A Fragment from Row-Major LDS

When A is stored in LDS as `A_lds[row][col]` (row-major):

```cpp
const int lane = threadIdx.x % 16;

// Load column 'lane' from all 16 rows
#pragma unroll
for (int i = 0; i < 16; i++) {
    a_frag[i] = A_lds[row_offset + i][lane];
}
```

**Pattern**: Iterate over rows, take column `lane` from each row.

### B Fragment from Transposed LDS

When B is transposed in LDS as `B_lds[N][K]` (each row contains one column of original B):

```cpp
const int lane = threadIdx.x % 16;

// Load row 'lane' from transposed B (= column 'lane' of original B)
#pragma unroll
for (int k = 0; k < 16; k++) {
    b_frag[k] = B_lds[col_offset + lane][k];
}
```

**Pattern**: Load row `(col_offset + lane)` from transposed B_lds.

### Alternative: Using Helper Functions

The `rocwmma_patch/rocwmma_gfx1151.hpp` provides helper functions that handle the fragment layout correctly:

```cpp
#include "rocwmma_patch/rocwmma_gfx1151.hpp"

// Load A fragment
half16 a_frag;
load_matrix_sync_lds(a_frag, &A_lds[warp_m_base][0], A_STRIDE);

// Load B fragment (from transposed B)
half16 b_frag;
load_matrix_sync_lds_b_transposed(b_frag, &B_lds[warp_n_base][0], B_STRIDE);
```

**Recommendation**: Use helper functions for correctness. Manual fragment construction is error-prone due to packing/ABI requirements.

---

## Storing C Fragment to Global Memory

```cpp
const int lane = threadIdx.x;
const int frag_col = lane % 16;
const int frag_row_offset = lane / 16;  // 0 for lanes 0-15, 1 for lanes 16-31

#pragma unroll
for (int i = 0; i < 8; i++) {
    int row = tile_row + i * 2 + frag_row_offset;
    int col = tile_col + frag_col;
    
    if (row < M && col < N) {
        C[row * N + col] = c_frag[i];
    }
}
```

**Pattern**:
- Fragment index `i` maps to rows `i*2` (lanes 0-15) or `i*2+1` (lanes 16-31)
- Column is always `lane % 16`

---

## LDS Layout Considerations

### Padded LDS (Recommended for Simplicity)

Add padding to avoid bank conflicts:

```cpp
#define LDS_PAD 8
constexpr int A_STRIDE = BLOCK_K + LDS_PAD;  // 16 + 8 = 24 halfs

__shared__ __half A_lds[BLOCK_M][A_STRIDE];
```

**Access pattern**: Direct `[row][col]` indexing works correctly.

### XOR Swizzle LDS (Memory Efficient)

XOR swizzle eliminates bank conflicts without padding:

```cpp
// Store with swizzle
int k_group = col / 8;
int k_local = col % 8;
int k_group_swz = k_group ^ (row & 1);
int col_swz = k_group_swz * 8 + k_local;
A_lds[row * BLOCK_K + col_swz] = value;

// Load with inverse swizzle
int k_group_orig = col / 8;
int k_group_swz = k_group_orig ^ (row & 1);
int col_swz = k_group_swz * 8 + (col % 8);
value = A_lds[row * BLOCK_K + col_swz];
```

**Important**: When using XOR swizzle, fragment loading must apply the inverse swizzle transformation.

---

## Common Pitfalls

### 1. Lane Replication Missing

RDNA3 WMMA requires lanes 0-15 and 16-31 to contain identical data for A and B fragments.

```cpp
// WRONG: Different data in each half-wave
a_frag[i] = A_lds[row][threadIdx.x];

// CORRECT: Same data via modulo
a_frag[i] = A_lds[row][threadIdx.x % 16];
```

### 2. Row vs Column Confusion

A fragments need **columns** loaded (iterate rows, fixed column per lane).
B fragments from transposed LDS need **rows** loaded (fixed row per lane, iterate K).

### 3. Manual Fragment Packing Errors

The `half16` type has specific packing requirements. Manual construction often fails due to ABI mismatches.

```cpp
// RISKY: Manual packing
_Float16 a_data[16];
for (int i = 0; i < 16; i++) a_data[i] = ...;
half16 a_frag = *reinterpret_cast<half16*>(a_data);

// SAFER: Use helper functions
half16 a_frag;
load_matrix_sync_lds(a_frag, ptr, stride);
```

### 4. Wrong C Fragment Store Pattern

```cpp
// WRONG: Ignores lane-to-row mapping
for (int i = 0; i < 8; i++) {
    C[tile_row + i][tile_col + lane] = c_frag[i];
}

// CORRECT: Account for interleaved rows
for (int i = 0; i < 8; i++) {
    int row = tile_row + i * 2 + (lane / 16);
    int col = tile_col + (lane % 16);
    C[row * N + col] = c_frag[i];
}
```

---

## Verification Checklist

1. **Lane replication**: Verify `lane % 16` is used for A and B fragment indexing
2. **Column loading for A**: Iterate rows (0-15), fixed column per lane
3. **Row loading for B**: Fixed row per lane, iterate K (0-15)
4. **C store pattern**: Use `i*2 + (lane/16)` for row mapping
5. **LDS swizzle**: If using XOR swizzle, apply inverse during fragment load
6. **Helper functions**: Prefer `load_matrix_sync_lds` over manual construction

---

## References

- [AMD RDNA3.5 ISA Reference](https://docs.amd.com/v/u/en-US/rdna35_instruction_set_architecture)
- [rocWMMA Documentation](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/)
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
