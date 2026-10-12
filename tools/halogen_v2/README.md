# halogen v2 checkpoint decoders

CPU reference decoders for the weight formats in Peonist's halogen v2 checkpoint
(`qwen38-flash-next-v2.hgn`). The format writeup is in
[`docs/quantized_weight_decode_gfx1151.md`](../../docs/quantized_weight_decode_gfx1151.md).

| file | purpose |
|---|---|
| `hgn.py` | header and table parser, the XOR32 payload checksum, and HTTP range reads. `python3 hgn.py [path-or-url]` prints a per-family format summary. |
| `decode.py` | `ht4`, `i4r` and `q6g64` decoders, plus the two-sided sign-and-Hadamard `rotate` / `unrotate` |
| `check_bf16.py` | decodes one tensor and compares it with the Qwen BF16 original. It range-fetches only the bytes it needs. |
| `disasm/` | the gfx1151 kernels the decoders were derived from (Peonist's code, internal reference only) |

Needs only NumPy, plus network access to huggingface.co unless `--hgn` points at a
local file. Run from this directory without `python3 -I`, because the scripts import
each other.

Results recorded 2026-10-12, layer 3:

| tensor | format | corr | rel. error |
|---|---|---|---|
| `self_attn.o_proj` | ht4 | 0.99573 | 9.4% |
| `mlp.experts.down_proj`, expert 1 | i4r | 0.99324 | 11.7% |
| `mlp.experts.gate_up_proj`, expert 0 | i4r | 0.99251 | 12.3% |
| `attn_hyper_connection.input_mix_weight_up` | q6g64 | 0.99968 | 2.5% |

The XOR32 checksum matched the table on the `ht4` and `q6g64` tensors. Tensors over
256 MiB are not hashed.

Not covered here: `q4c` (the layout `convert` writes, already handled by the
halogen-abliterate pipeline), `q8g64` (MTP head), the bit meaning of the table's
`qparam` field, and any encoder.
