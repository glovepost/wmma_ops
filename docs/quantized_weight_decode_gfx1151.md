# Quantized-Weight Decode on gfx1151: Notes from halogen-flash-server

Companion to the GEMM material in this repo, in a **different regime**: decode-time
dequantization of 4- and 6-bit weights for bandwidth-bound GEMV, not compute-bound
WMMA GEMM. Nothing here is a `wmma-ops` measurement.

**Source**: the gfx1151 code objects bundled in `halogen-tools` from
`ghcr.io/peonist-ai/halogen-flash-server:0.18.1` (Peonist's Qwen3.8-Flash-Next server
for Strix Halo). The device code keeps its symbol names, so each claim below names
the kernel it was read from. Peonist's EULA (`LICENSE.md` in their repo) does not
prohibit reverse engineering and asks that derived kernels or source not be
redistributed as one's own. This file therefore describes formats and techniques and
does not reproduce their kernels.

**Method**: disassembled with `llvm-objdump --mcpu=gfx1151`, then each format was
checked by decoding tensors from their published v2 checkpoint
(`peonist-ai/halogen-qwen3.8-flash-next`, Apache-2.0 weights) and comparing them
against the BF16 originals in `Qwen/Qwen3.8-Flash-Next`. "Checked" below means that
comparison was run. "Read from ISA" means it was inferred from the disassembly and
was not exercised separately.

Recorded 2026-10-12.

---

## Extracting the device code

```bash
cid=$(docker create ghcr.io/peonist-ai/halogen-flash-server:0.18.1)
docker export "$cid" | tar -x -C rootfs usr/local/bin/halogen-tools; docker rm "$cid"
objcopy -O binary --only-section=.hip_fatbin rootfs/usr/local/bin/halogen-tools tools.fatbin
# tools.fatbin is several concatenated clang offload bundles; split on the
# "__CLANG_OFFLOAD_BUNDLE__" magic, then for each piece:
clang-offload-bundler --unbundle --type=o --input=bNN.bundle \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx1151 --output=bNN.co
llvm-objdump -d --mcpu=gfx1151 --disassemble-symbols=<mangled> bNN.co
```

The system LLVM (23.x) handles gfx1151; no ROCm install is needed. The checkpoint
header alone is enough for the vendor's own `halogen-tools inspect`, so a sparse copy
works without downloading the 62 GiB file: `truncate -s <size>` a file holding the
first 1 MiB. That run is how the format names below were obtained.

---

## Formats in the v2 checkpoint

| code | name | decode kernel (symbol) | used for | bits/weight |
|---|---|---|---|---|
| 16 | `ht4` | `k_ht_deq_t<4>` | attention, DeltaNet projections, shared expert, `lm_head` | 4.00 |
| 23 | `i4r` | `k_i4r_rows<640/2560>` | routed experts (about 94% of bytes) | 4.125 |
| 24 | `q6g64` | `k_deq_q6g64` | hyper-connection mixers | 6.5 (+ row padding) |
| 5 | `q4c` | `k_deq_q4cp*` | embeddings, small projections | 4.5 |
| 7 | `q8g64` | `k_deq_q8g64` | MTP head | 8.5 |

`ht4` and `i4r` both work in an **incoherence-rotated basis**. One fp16 sign or scale
vector sits on each side (`suh` on the input dimension, `svh` on the output). A
128-block normalized Sylvester Hadamard H is applied on both sides:

```
W = diag(svh) · Hout( Hin(M) · diag(suh) )      (H symmetric, H·H = I)
```

The `ht4` and `i4r` formulas match each other up to which side is read as rows. On
layer-3 `o_proj`, undoing the rotation drops the BF16 weights' excess kurtosis from
7.0 to 3.06, which is Gaussian; that Gaussian shape is what the codebooks below assume.

### `ht4`: EXL3-style trellis, K=4, `mul1` codebook (checked)

This is exllamav3's EXL3 scheme (QTIP-style), laid out for gfx1151:

- **Tiles**: 16×16, 128 bytes each. The matrix is (out, in). Tiles are ordered in
  128-row bands. Within a band, column tiles are outer and the 8 row tiles inner.
- **Bitstream**: the tile's 32 little-endian u32 words, MSB-first within each word,
  read as one 1024-bit circular stream. The state for weight i is the 16 bits ending
  at bit (i+1)·4, so the start of the stream wraps around to the end.
- **Codebook**: exllamav3's `mul1` (`codebook.cuh`, cb=2).
  `x = state * 0x83DCD12D`; `s` = the sum of x's four bytes;
  `v = half(0x6400 + s) * half(0x1eee) + half(0xc931)`.
  On gfx1151 the byte sum is a single `v_sad_u8 x, 0, 0x6400`. That does the job of
  the `__dp4a(x, 0x01010101, 0x6400)` CUDA uses, and the 0x6400 bias turns the
  result directly into an fp16 bit pattern for 1024+s. One `v_pk_fma_f16` then
  scales and offsets two values at once.
- **Lane mapping**: each of 32 lanes decodes 8 consecutive stream positions,
  i0 = 8·lane. Element i lands at row `(i/8) % 16` and column `8·(i/128) + i%8`, so
  every lane writes one 16-byte vector. Two states come from the lane's own word
  using `v_bfe_u32`. The other four take a 64-bit window
  (`v_lshrrev_b64`) over the two words before it, because states straddle words.
- Checked: layer-3 `o_proj` correlation 0.9957, relative error 9.2%; `q_proj`
  0.9963, 8.6%.

This tile layout differs from exllamav3's own. exllamav3 permutes tiles to suit
NVIDIA tensor cores (`tensor_core_perm`) and swaps uint16 halves within each
32-bit word. Without re-tiling, exllamav3's decoder and encoder produce data in the
wrong order for gfx1151.

### `i4r`: rotated int4, one fp16 scale per 128 (checked)

- **Rotation**: the same two-sided sign-and-Hadamard rotation. The sign vectors are
  ±1 and shared by every expert in the layer.
- **Codes**: all experts' codes come first, expert-major, each expert row-major
  (out, in). Byte j holds input 2j in its low nibble and 2j+1 in its high nibble.
  The value is `(nibble − 8) · scale`.
- **Scales**: fp16, stored after all codes, laid out as `[expert][out][in/128]`. Their
  signs carry information, and about half are negative.
- **GEMV inner loop**, read from ISA (`k_i4r_rows<640>`):
  1. `v_and 0x0f0f0f0f` and a 4-bit shift split the nibbles.
  2. `v_perm_b32` with selectors `0x0c040c00` / `0x0c050c01` / ... places one
     nibble in the low byte of each fp16 half.
  3. `v_or 0x64006400` turns each into fp16 1024+q, and `v_pk_add_f16 0xe408`
     (−1032) gives q−8. That is exact with no conversion instruction.
  4. `v_dot2acc_f32_f16` accumulates against the input, which has already been
     rotated and staged in LDS.
  5. Each 128-group's partial sum is scaled once with `v_fma_mix_f32`.
  6. A two-step DPP `row_xmask` reduction combines the 4 lanes that share a row.
- Checked: expert `down_proj` correlation 0.993 (two experts); `gate_up_proj` 0.9925,
  relative error 12.3%.

### `q6g64`: affine 6-bit, groups of 64 (checked)

Each row is padded to a 16-byte stride, `align16(C/2 + C/4 + C/16)`. It holds C/2
bytes of low nibbles, then C/4 bytes of 2-bit high pairs (LSB-first), then one
`(fp16 d, fp16 m)` pair per 64 weights. The value is `(lo | hi<<4)·d + m`, applied
with `v_fma_mix_f32` using `op_sel` to take d and m from one packed register.
Checked: correlation 0.99968, relative error 2.5%.

---

## Takeaways for our kernels

- **Procedural codebooks are cheap on gfx1151.** Each `mul1` decode is one
  `v_mul_lo_u32`, one `v_sad_u8` and half of a `v_pk_fma_f16`, with no memory
  lookup. That is the main reason a trellis format can run at 4.00 bpw without a
  decode-speed penalty.
- **Exact int4→fp16 needs no conversion instruction**: OR in the exponent bits
  (`0x6400`), then subtract the bias in packed fp16.
- **Pre-rotate activations, not weights.** Both rotated formats expect the input to
  be Hadamard-rotated already. A separate `k_ht_rot_in` kernel exists, inferred from
  its name and signature to do that rotation. The weight path then has no transform
  cost.
- **Reusing exllamav3 needs a re-tile, not a new encoder.** Its `mul1` K=4 encoder
  produces the right codes; only the tile order and word packing differ.

## Not established

- The upper half of each table entry's last 64-bit field is a format parameter
  (`qparam`, for example `0x1208` on `ht4` and `0x40` on `q6g64`) whose bit meaning
  is not decoded. The lower half is a checksum: the XOR of the payload's
  little-endian u32 words, verified on several v2 tensors.
- `q4c` and `q8g64` were identified from kernel names only and not decoded here.
- No throughput was measured. These are layout and arithmetic findings, not
  performance claims.
