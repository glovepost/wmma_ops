#!/usr/bin/env python3
"""Trace the historical PyTorch/rocBLAS FP16-output comparison path."""

import torch


size = 4096
warmup = 10
iterations = 50
torch.manual_seed(20260822)
a = torch.randn((size, size), device="cuda", dtype=torch.float16)
b = torch.randn((size, size), device="cuda", dtype=torch.float16)
for _ in range(warmup):
    torch.mm(a, b)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(iterations):
    torch.mm(a, b)
stop.record()
stop.synchronize()
time_ms = start.elapsed_time(stop) / iterations
operations = 2.0 * size * size * size
print(
    f"torch.mm FP16xFP16->FP16: {time_ms:.6f} ms, "
    f"{operations / (time_ms * 1.0e-3) / 1.0e12:.3f} TFLOPS"
)
