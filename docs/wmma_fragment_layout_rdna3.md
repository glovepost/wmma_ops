# WMMA Fragment Layout for RDNA3 (gfx1151)

Technical reference for Wave Matrix Multiply-Accumulate (WMMA) fragment layouts on AMD RDNA3.5 architecture.

**Sources**: 
- [AMD GPUOpen: How to accelerate AI applications on RDNA 3 using WMMA](https://gpuopen.com/learn/wmma_on_rdna3/)
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
- [rocWMMA Documentation](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/)

---

## Overview

The `v_wmma_f32_16x16x16_f16` instruction performs a 16×16×16 matrix multiply-accumulate:
- **A Matrix**: 16×16 (M×K), FP16 input, stored in **column-major** order in registers
- **B Matrix**: 16×16 (K×N), FP16 input, stored in **row-major** order in registers
- **C/D Matrix**: 16×16 (M×N), FP32 accumulator, stored in **row-major** order

Each wave (32 threads) cooperatively loads and processes one 16×16 tile.

**Critical**: RDNA3 WMMA requires that A_frag and B_frag contents are **replicated** between lanes 0-15 and lanes 16-31.

---

## Fragment Register Layout

### A Matrix Fragment (16×16, FP16)

**Storage**: Column-major in registers. Each lane holds one **row** of matrix A.

**Hardware expects**: `a_frag[k] = A[lane % 16][k]` (lane's row, all K columns)

| Fragment Index | Matrix Element | Description |
|----------------|----------------|-------------|
| `a_frag[0]` | `A[lane % 16][0]` | Lane's row, K=0 |
| `a_frag[1]` | `A[lane % 16][1]` | Lane's row, K=1 |
| ... | ... | ... |
| `a_frag[15]` | `A[lane % 16][15]` | Lane's row, K=15 |

**Key insight**: Each lane loads one **row** of the A matrix (all 16 K values of that row).

> **Verified** two ways.
>
> First, the RDNA3.5 ISA itself. Section 7.9 contains "A/B/C & D Matrix: VGPR View
> for Wave32" diagrams giving the full element-to-register mapping. **They are
> vector figures, so they are invisible to `pdftotext` and to any grep of the text
> layer** — which is very likely why this layout had to be reverse-engineered
> twice here (`e61b06b`, `acaabd4`). The A-matrix figure reads:
>
> ```text
>              Lane 0   Lane 1   Lane 2   ...   Lane 15
>   V0[15:0]   A0_0     A1_0     A2_0     ...   A15_0
>   V0[31:16]  A0_1     A1_1     A2_1     ...   A15_1
>   ...
>   V7[31:16]  A0_15    A1_15    A2_15    ...   A15_15
> ```
>
> so lane `L` holds `A[L][0..15]` — **row L**. The ISA's own legend explains the
> naming that causes the confusion: *"Row Major Order" = one matrix row is stored
> in one VGPR (or one half of a VGPR for 16-bit values); "Column Major Order" =
> one matrix column is stored in one VGPR.* The ISA labels A "column major" and
> GPUOpen says each lane holds a column, but those describe **different axes** —
> per-VGPR versus per-lane. Both are consistent with lane L holding row L.
>
> Second, the AMD Matrix Instruction Calculator, which agrees. Running
> `matrix_calculator.py -a gfx1151 -i v_wmma_f32_16x16x16_f16 -A -M -w 32` reports
> lane 29 -> `A[13][*]`, lane 30 -> `A[14][*]`, lane 31 -> `A[15][*]`: lane L holds
> **row** `L % 16`, all 16 K values, two per VGPR (`v0{0}.[15:0] = A[0][0]`,
> `v0{0}.[31:16] = A[0][1]`).
>
> Pass the gfx target, not the generic arch name. The calculator maps
> `gfx1150/1151/1152/1153` (RDNA3.5) onto its `rdna3` matrix model — output is
> byte-identical — so the WMMA layout and timings below are AMD's answer for
> gfx1151, not an RDNA3 result assumed to carry over.
>
> A transcription of the ISA's VGPR-view figures (A, B, and C/D) is kept in the
> markdown conversion of the ISA, alongside the rendered pages, so this mapping is
> greppable rather than locked in a vector figure.
>
> The GPUOpen article describes A as "each lane stores one column", which reads as
> the opposite of this. Its code is right; its prose is not. That is the bug fixed
> in `e61b06b` and `acaabd4` — please do not "fix" it back.

**Lane replication**: Lanes 0-15 and lanes 16-31 must contain identical data.

### B Matrix Fragment (16×16, FP16)

**Storage**: Row-major in registers. Each lane holds one **column** of matrix B.

**Hardware expects**: `b_frag[k] = B[k][lane % 16]` (all K rows, lane's column)

| Fragment Index | Matrix Element | Description |
|----------------|----------------|-------------|
| `b_frag[0]` | `B[0][lane % 16]` | K=0, lane's column |
| `b_frag[1]` | `B[1][lane % 16]` | K=1, lane's column |
| ... | ... | ... |
| `b_frag[15]` | `B[15][lane % 16]` | K=15, lane's column |

**Key insight**: Each lane loads one **column** of the B matrix (all 16 K values of that column).

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

## Loading Fragments from Global/LDS Memory

### A Fragment from Row-Major Memory

When A is stored in row-major format `A[row][col]` (standard C layout):

```cpp
const int lane = threadIdx.x % 16;

// Each lane loads its own ROW of A (all 16 K values)
#pragma unroll
for (int k = 0; k < 16; k++) {
    a_frag[k] = A[16 * lane + k];  // A[lane][k] in row-major
}
```

**Pattern**: Each lane loads row `lane`, iterating over all K columns.

**From AMD GPUOpen example**:
```cpp
for (int ele = 0; ele < 16; ++ele) {
    a_frag[ele] = a[16 * lane + ele];  // Load row 'lane'
}
```

### B Fragment from Row-Major Memory

When B is stored in row-major format `B[row][col]`:

```cpp
const int lane = threadIdx.x % 16;

// Each lane loads its own COLUMN of B (all 16 K values)
#pragma unroll
for (int k = 0; k < 16; k++) {
    b_frag[k] = B[16 * k + lane];  // B[k][lane] in row-major
}
```

**Pattern**: Each lane loads column `lane`, iterating over all K rows.

**From AMD GPUOpen example**:
```cpp
for (int ele = 0; ele < 16; ++ele) {
    b_frag[ele] = b[16*ele + lane];  // Load column 'lane'
}
```

### Loading from LDS with Transposed B

When B is transposed in LDS as `B_lds[N][K]` (common optimization):

```cpp
const int lane = threadIdx.x % 16;

// B is transposed: B_lds[n][k] = B_original[k][n]
// Each lane loads row 'lane' from transposed B = column 'lane' of original B
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

**From AMD GPUOpen example** (FP16 output with OPSEL=false):
```cpp
const int lane = threadIdx.x % 16;
const int lIdx = threadIdx.x;

for (int ele = 0; ele < 8; ++ele) {
    const int r = ele * 2 + (lIdx / 16);
    // For FP16 with OPSEL=false, use even indices
    c[16 * r + lane] = c_frag[ele * 2];
}
```

**For FP32 output** (v_wmma_f32_16x16x16_f16):
```cpp
const int lane = threadIdx.x % 16;
const int lIdx = threadIdx.x;

#pragma unroll
for (int i = 0; i < 8; i++) {
    int row = tile_row + i * 2 + (lIdx / 16);
    int col = tile_col + lane;
    
    if (row < M && col < N) {
        C[row * N + col] = c_frag[i];  // FP32: direct indexing
    }
}
```

**Pattern**:
- Fragment index `i` maps to rows `i*2` (lanes 0-15) or `i*2+1` (lanes 16-31)
- Column is always `lane % 16`
- For FP16 output with OPSEL=false: use `c_frag[ele*2]`
- For FP16 output with OPSEL=true: use `c_frag[ele*2 + 1]`
- For FP32 output: use `c_frag[i]` directly

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

**Correct loading patterns**:
- **A fragments**: Each lane loads its own **row** of A (all 16 K values)
- **B fragments**: Each lane loads its own **column** of B (all 16 K values)
- **B from transposed LDS**: Each lane loads its own **row** from transposed B_lds

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
