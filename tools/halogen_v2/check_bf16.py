"""Decode one v2 tensor and compare it with the Qwen BF16 original.

Fetches only the needed byte ranges over HTTP (or reads a local .hgn).

    python3 check_bf16.py layers.3.self_attn.o_proj                # ht4
    python3 check_bf16.py layers.3.mlp.experts.down_proj --expert 0 # i4r
    python3 check_bf16.py layers.3.attn_hyper_connection.input_mix_weight_up  # q6g64
"""
import argparse
import json
import struct

import numpy as np

import decode
import hgn

QWEN = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"


def base_tensor(name, expert=None):
    idx = json.loads(hgn.read_range(QWEN + "model.safetensors.index.json", 0, 1 << 26))["weight_map"]
    key = "model.language_model." + name
    key = key if key in idx else key + ".weight"
    f = QWEN + idx[key]
    n = struct.unpack("<Q", hgn.read_range(f, 0, 8))[0]
    e = json.loads(hgn.read_range(f, 8, 8 + n))[key]
    assert e["dtype"] == "BF16", e["dtype"]
    start, end = e["data_offsets"]
    shape = e["shape"]
    if expert is not None:
        per = int(np.prod(shape[1:])) * 2
        start, end, shape = start + expert * per, start + (expert + 1) * per, shape[1:]
    raw = hgn.read_range(f, 8 + n + start, 8 + n + end)
    return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--expert", type=int, default=0)
    ap.add_argument("--hgn", default=hgn.V2_URL)
    a = ap.parse_args()
    h = hgn.load_header(a.hgn)
    by = {t["name"]: t for t in h["tensors"]}
    t = by[a.stem + ".weight"]
    get = lambda tt, s=0, e=None: hgn.read_range(a.hgn, tt["off"] + s, tt["off"] + (tt["size"] if e is None else e))
    f16 = lambda n: np.frombuffer(get(by[n]), np.float16).astype(np.float32)
    fmt = hgn.FORMATS.get(t["fmt"])
    if fmt == "ht4":
        out_f, in_f = t["dims"]
        w = decode.unrotate(decode.ht4(get(t), out_f, in_f), f16(a.stem + ".suh"), f16(a.stem + ".svh"))
        ref = base_tensor(a.stem)
    elif fmt == "i4r":
        _, out_f, in_f = t["dims"]
        cs, ce, ss, se = decode.i4r_ranges(t, a.expert)
        m = decode.i4r_expert(get(t, cs, ce), get(t, ss, se), out_f, in_f)
        prefix = a.stem.rsplit(".", 1)[0]
        suh = f16(a.stem + ".suh") if a.stem + ".suh" in by else f16(prefix + ".su")
        w = decode.unrotate(m, suh, f16(a.stem + ".svh"))
        ref = base_tensor(a.stem, a.expert)
    elif fmt == "q6g64":
        rows, cols = t["dims"]
        w = decode.q6g64(get(t), rows, cols)
        ref = base_tensor(a.stem)
    else:
        raise SystemExit(f"{a.stem}: format {fmt} has no decoder here")
    c = np.corrcoef(w.ravel(), ref.ravel())[0, 1]
    e = np.linalg.norm(w - ref) / np.linalg.norm(ref)
    print(f"{a.stem} [{fmt}] corr={c:.5f} relerr={e:.4f} checksum_ok={hgn.checksum(get(t)) == t['checksum'] if t['size'] < 1 << 28 else 'skipped (large)'}")


if __name__ == "__main__":
    main()
