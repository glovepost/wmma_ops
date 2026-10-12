"""Reference (numpy, CPU) decoders for the halogen v2 weight formats.

Derived from the gfx1151 kernels k_ht_deq_t<4>, k_i4r_rows<N> and
k_deq_q6g64 in halogen-flash-server 0.18.1; see
docs/quantized_weight_decode_gfx1151.md. Slow but exact enough for
validation against BF16 references.
"""
import numpy as np

HAD = 128


def hadamard(n=HAD):
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def had_in(m):
    """128-block Hadamard along the last (input) axis."""
    return (m.reshape(m.shape[0], -1, HAD) @ hadamard()).reshape(m.shape)


def had_out(m):
    """128-block Hadamard along the first (output) axis."""
    return (hadamard() @ m.reshape(-1, HAD, m.shape[1])).reshape(m.shape)


def unrotate(m, suh, svh):
    """Rotated-basis (out, in) matrix -> original basis."""
    return had_out(had_in(m) * suh[None, :]) * svh[:, None]


def rotate(w, suh, svh):
    """Original (out, in) matrix -> rotated basis (inverse of unrotate)."""
    return had_in(had_out(w / svh[:, None]) / suh[None, :])


def mul1(state):
    """exllamav3 'mul1' procedural codebook, evaluated in fp16 like the kernel."""
    x = (state.astype(np.uint64) * 0x83DCD12D) & 0xFFFFFFFF
    s = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    h = (1024 + s).astype(np.float16)
    k = np.uint16(0x1EEE).view(np.float16)
    b = np.uint16(0xC931).view(np.float16)
    return (h * k + b).astype(np.float32)


def ht4(payload, out_f, in_f, K=4):
    """ht4 payload -> rotated-basis (out_f, in_f) matrix."""
    nt = (out_f // 16) * (in_f // 16)
    words = np.frombuffer(payload, "<u4").reshape(nt, 8 * K)
    bits = np.unpackbits(words.astype(">u4").view(np.uint8).reshape(nt, -1), axis=1)
    L = 256 * K
    idx = ((np.arange(256)[:, None] + 1) * K - 16 + np.arange(16)[None, :]) % L
    state = (bits[:, idx].astype(np.uint32) << np.arange(15, -1, -1, dtype=np.uint32)).sum(-1)
    v = mul1(state)
    i = np.arange(256)
    tiles = np.zeros((nt, 16, 16), np.float32)
    tiles[:, (i // 8) % 16, (i // 128) * 8 + i % 8] = v
    # memory order: 128-row band, then column tile, then the band's 8 row tiles
    t = tiles.reshape(out_f // 128, in_f // 16, 8, 16, 16)
    return t.transpose(0, 2, 3, 1, 4).reshape(out_f, in_f)


def i4r_expert(codes, scales, out_f, in_f):
    """One expert's i4r codes (out_f*in_f/2 bytes) and fp16 scales -> rotated (out_f, in_f)."""
    c = np.frombuffer(codes, np.uint8)
    q = np.stack([c & 15, c >> 4], 1).reshape(out_f, in_f).astype(np.float32) - 8
    s = np.frombuffer(scales, np.float16).astype(np.float32).reshape(out_f, in_f // 128)
    return (q.reshape(out_f, -1, 128) * s[:, :, None]).reshape(out_f, in_f)


def i4r_ranges(t, expert):
    """(codes_start, codes_end, scales_start, scales_end) within an i4r tensor payload."""
    E, out_f, in_f = t["dims"]
    cb = out_f * in_f // 2
    sb = out_f * (in_f // 128) * 2
    return (expert * cb, (expert + 1) * cb, E * cb + expert * sb, E * cb + (expert + 1) * sb)


def q6g64(payload, rows, cols):
    """q6g64 payload -> (rows, cols) matrix (not rotated)."""
    stride = (cols // 2 + cols // 4 + cols // 16 + 15) & ~15
    r = np.frombuffer(payload, np.uint8).reshape(rows, stride)
    lo = r[:, :cols // 2]
    hi = r[:, cols // 2:cols // 2 + cols // 4]
    dm = r[:, cols // 2 + cols // 4:cols // 2 + cols // 4 + cols // 16].copy()
    dm = dm.view(np.float16).astype(np.float32).reshape(rows, cols // 64, 2)
    l = np.stack([lo & 15, lo >> 4], 2).reshape(rows, cols)
    h = np.stack([(hi >> s) & 3 for s in (0, 2, 4, 6)], 2).reshape(rows, cols)
    q = (l | (h << 4)).astype(np.float32).reshape(rows, cols // 64, 64)
    return (q * dm[:, :, 0:1] + dm[:, :, 1:2]).reshape(rows, cols)
