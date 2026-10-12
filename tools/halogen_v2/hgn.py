"""Read halogen .hgn v2 checkpoint headers (HGN1, version 2).

Header: magic "HGN1", u32 version, u64 n_tensors, u64 table_off, u64 data_off,
u64 total_size, char[64] model_id. Table: n x 160-byte entries of
char[96] name, u32 fmt, u32 ndim, u64 dims[4], u64 offset, u64 size,
u32 checksum, u32 qparam. checksum is the XOR of the payload's little-endian
u32 words (zero-padded to a multiple of 4 bytes).
"""
import collections
import re
import struct
import sys
import urllib.request

import numpy as np

FORMATS = {0: "bf16", 2: "f16", 4: "i64", 5: "q4c", 7: "q8g64", 10: "fp8",
           11: "q8", 16: "ht4", 23: "i4r", 24: "q6g64"}
V2_URL = ("https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next/"
          "resolve/main/qwen38-flash-next-v2.hgn")


def parse(buf):
    magic, ver, n, tab, data_off, total = struct.unpack_from("<4sIQQQQ", buf, 0)
    assert magic == b"HGN1", magic
    model = buf[0x28:0x68].split(b"\0")[0].decode()
    tensors = []
    for i in range(n):
        o = tab + i * 160
        fmt, nd = struct.unpack_from("<II", buf, o + 96)
        off, size, csum, qparam = struct.unpack_from("<QQII", buf, o + 136)
        tensors.append(dict(
            name=buf[o:o + 96].split(b"\0")[0].decode(), fmt=fmt,
            dims=struct.unpack_from("<4Q", buf, o + 104)[:nd],
            off=off, size=size, checksum=csum, qparam=qparam))
    return dict(version=ver, model=model, data_off=data_off, total=total,
                tensors=tensors)


def checksum(payload):
    pad = payload + b"\0" * (-len(payload) % 4)
    return int(np.bitwise_xor.reduce(np.frombuffer(pad, "<u4")))


def read_range(src, start, end):
    """Bytes [start, end) of a local path or an http(s) URL."""
    if src.startswith("http"):
        req = urllib.request.Request(src, headers={"Range": f"bytes={start}-{end - 1}"})
        return urllib.request.urlopen(req).read()
    with open(src, "rb") as f:
        f.seek(start)
        return f.read(end - start)


def load_header(src, nbytes=1 << 20):
    return parse(read_range(src, 0, nbytes))


def summary(hdr):
    fam = lambda n: re.sub(r"layers\.\d+\.", "L.", n)
    agg = collections.OrderedDict()
    for t in hdr["tensors"]:
        k = (fam(t["name"]), FORMATS.get(t["fmt"], t["fmt"]))
        a = agg.setdefault(k, [0, 0, 0])
        a[0] += 1
        a[1] += int(np.prod(t["dims"]))
        a[2] += t["size"]
    for (f, fm), (c, ne, sz) in agg.items():
        print(f"  {f:64s} {fm!s:6s} n={c:4d} bpw={8 * sz / max(ne, 1):6.2f} {sz / 2**30:8.3f} GiB")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else V2_URL
    h = load_header(src)
    print(f"{h['model']}: v{h['version']}, {len(h['tensors'])} tensors, {h['total'] / 2**30:.2f} GiB")
    summary(h)
