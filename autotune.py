#!/usr/bin/env python3
"""
WMMA Kernel Auto-Tuner using Optuna
===================================

Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler to find
optimal tile configurations for different matrix sizes on gfx1151.

Similar to AMD's Tensile approach but focused on our WMMA kernel variants.

Usage:
    python autotune.py                    # Run full auto-tuning
    python autotune.py --quick            # Quick search (50 trials)
    python autotune.py --size 4096 4096 2048  # Tune specific size

References:
    - Tensile: https://github.com/ROCmSoftwarePlatform/Tensile
    - Optuna: https://optuna.org/
    - ROCm WMMA: https://github.com/ROCm/ROCm/issues/2640
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Install with: pip install optuna")

try:
    import wmma_ops
    WMMA_AVAILABLE = True
except ImportError:
    WMMA_AVAILABLE = False
    print("Warning: wmma_ops not available. Build with: pip install -e .")


# Tile configurations to search
# Format: (tile_config_id, block_m, block_n, n_warps, description)
TILE_CONFIGS = [
    (0, 64, 64, 4, "Small 64x64"),
    (1, 128, 64, 8, "Medium 128x64"),
    (2, 256, 64, 16, "Large 256x64"),
    (3, 128, 128, 16, "Wide 128x128"),
]

# Kernel variants to search
KERNEL_VARIANTS = [
    ("standard", lambda A, B: wmma_ops.matmul(A, B), "Standard (LDS pad)"),
    ("adaptive", lambda A, B: wmma_ops.matmul_adaptive(A, B), "Adaptive tile"),
    ("noPrefetch", lambda A, B: wmma_ops.matmul_noPrefetch(A, B), "No prefetch"),
]

# Common matrix sizes for ML workloads
COMMON_SIZES = [
    # Small (batch/embedding operations)
    (256, 256, 256),
    (512, 512, 512),
    (768, 768, 768),
    # Medium (typical layer sizes)
    (1024, 1024, 1024),
    (2048, 2048, 512),
    (2048, 2048, 1024),
    (2048, 2048, 2048),
    # Large (big models)
    (4096, 4096, 1024),
    (4096, 4096, 2048),
    (4096, 4096, 4096),
    # LLM shapes (tall/wide)
    (8192, 1024, 2048),
    (1024, 8192, 2048),
    (4096, 11008, 4096),  # LLaMA MLP shape
]


def benchmark_kernel(
    M: int, N: int, K: int,
    kernel_fn,
    n_warmup: int = 5,
    n_iter: int = 20,
) -> Tuple[Optional[float], Optional[float], str]:
    """Benchmark a kernel and return (TFLOPS, time_ms, status)."""
    try:
        A = torch.randn(M, K, dtype=torch.float16, device="cuda")
        B = torch.randn(K, N, dtype=torch.float16, device="cuda")
        
        # Verify correctness
        C = kernel_fn(A, B)
        C_ref = torch.matmul(A, B)
        rel_err = (C - C_ref).abs().max().item() / (C_ref.abs().max().item() + 1e-8)
        
        if rel_err > 0.01:
            return None, None, f"FAIL (err={rel_err:.4f})"
        
        # Warmup
        for _ in range(n_warmup):
            _ = kernel_fn(A, B)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = kernel_fn(A, B)
        torch.cuda.synchronize()
        
        elapsed_ms = (time.perf_counter() - start) / n_iter * 1000
        tflops = 2 * M * N * K / elapsed_ms / 1e9
        
        return tflops, elapsed_ms, "OK"
        
    except Exception as e:
        return None, None, f"ERROR: {str(e)[:30]}"


def get_kernel_for_tile(tile_config: int):
    """Return kernel function for given tile configuration."""
    def kernel_fn(A, B):
        return wmma_ops.matmul_tiled(A, B, tile_config)
    return kernel_fn


class WMMAAutoTuner:
    """Optuna-based auto-tuner for WMMA kernel tile selection."""
    
    def __init__(self, cache_file: str = "tuning_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache: Dict[str, Dict] = {}
        self.load_cache()
    
    def load_cache(self):
        """Load cached tuning results."""
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self.cache = json.load(f)
            print(f"Loaded {len(self.cache)} cached configurations")
    
    def save_cache(self):
        """Save tuning results to cache."""
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)
    
    def get_cache_key(self, M: int, N: int, K: int) -> str:
        """Generate cache key for matrix dimensions."""
        return f"{M}x{N}x{K}"
    
    def create_objective(self, M: int, N: int, K: int):
        """Create Optuna objective function for given matrix size."""
        
        def objective(trial: optuna.Trial) -> float:
            # Sample tile configuration
            tile_config = trial.suggest_categorical("tile_config", [0, 1, 2, 3])
            
            # Get kernel function
            kernel_fn = get_kernel_for_tile(tile_config)
            
            # Benchmark
            tflops, time_ms, status = benchmark_kernel(M, N, K, kernel_fn)
            
            if tflops is None:
                # Return very low value for failed runs
                return 0.0
            
            # Store additional info
            trial.set_user_attr("time_ms", time_ms)
            trial.set_user_attr("status", status)
            trial.set_user_attr("tile_desc", TILE_CONFIGS[tile_config][4])
            
            return tflops
        
        return objective
    
    def tune_size(
        self,
        M: int, N: int, K: int,
        n_trials: int = 20,
        timeout: Optional[int] = None,
    ) -> Dict:
        """Tune for a specific matrix size."""
        
        cache_key = self.get_cache_key(M, N, K)
        
        # Check cache
        if cache_key in self.cache:
            print(f"  Using cached result for {cache_key}")
            return self.cache[cache_key]
        
        print(f"  Tuning {cache_key}...")
        
        # Create study with TPE sampler (like Tensile)
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"wmma_{cache_key}",
        )
        
        # Optimize
        study.optimize(
            self.create_objective(M, N, K),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False,
        )
        
        # Get best result
        best_trial = study.best_trial
        best_config = best_trial.params["tile_config"]
        best_tflops = best_trial.value
        
        result = {
            "M": M,
            "N": N,
            "K": K,
            "best_tile_config": best_config,
            "best_tile_desc": TILE_CONFIGS[best_config][4],
            "best_tflops": best_tflops,
            "best_time_ms": best_trial.user_attrs.get("time_ms"),
            "n_trials": n_trials,
        }
        
        # Cache result
        self.cache[cache_key] = result
        self.save_cache()
        
        return result
    
    def tune_common_sizes(self, n_trials: int = 20):
        """Tune all common matrix sizes."""
        print("=" * 70)
        print("WMMA Auto-Tuning (Optuna TPE Sampler)")
        print("=" * 70)
        print()
        
        results = []
        
        for M, N, K in COMMON_SIZES:
            result = self.tune_size(M, N, K, n_trials=n_trials)
            results.append(result)
            
            print(f"  {M}x{N}x{K}: {result['best_tflops']:.1f} TFLOPS "
                  f"({result['best_tile_desc']})")
        
        return results
    
    def generate_dispatch_code(self) -> str:
        """Generate C++ dispatch code based on tuning results."""
        
        code = """
// Auto-generated by autotune.py
// Optimal tile selection based on Optuna tuning

__host__ inline TileConfig select_tuned_tile(int M, int N, int K) {
"""
        
        # Sort by size for efficient branching
        sorted_cache = sorted(
            self.cache.items(),
            key=lambda x: (x[1]["M"], x[1]["N"], x[1]["K"])
        )
        
        for key, result in sorted_cache:
            M, N, K = result["M"], result["N"], result["K"]
            config = result["best_tile_config"]
            config_name = ["SMALL_64x64", "MEDIUM_128x64", "LARGE_256x64", "WIDE_128x128"][config]
            
            code += f"    // {key}: {result['best_tflops']:.1f} TFLOPS\n"
            code += f"    if (M == {M} && N == {N} && K == {K})\n"
            code += f"        return TileConfig::{config_name};\n\n"
        
        code += """    // Default: use heuristic
    return select_optimal_tile(M, N, K);
}
"""
        return code


def run_quick_benchmark():
    """Run quick benchmark comparing all tile configs and kernel variants."""
    print("=" * 70)
    print("Quick Benchmark: All Kernel Variants")
    print("=" * 70)
    print()
    
    sizes = [(2048, 2048, 2048), (4096, 4096, 4096)]
    
    for M, N, K in sizes:
        print(f"\nMatrix: {M}x{N}x{K}")
        print("-" * 55)
        
        # Benchmark tile configs
        print("Tile Configurations:")
        for tile_id, block_m, block_n, n_warps, desc in TILE_CONFIGS:
            kernel_fn = get_kernel_for_tile(tile_id)
            tflops, time_ms, status = benchmark_kernel(M, N, K, kernel_fn)
            
            if tflops:
                print(f"  {desc:<20} {tflops:6.1f} TFLOPS  {time_ms:6.2f}ms  {status}")
            else:
                print(f"  {desc:<20} {status}")
        
        # Benchmark kernel variants
        print("\nKernel Variants:")
        variants = [
            ("Standard", lambda A, B: wmma_ops.matmul(A, B)),
            ("Adaptive", lambda A, B: wmma_ops.matmul_adaptive(A, B)),
            ("NoPrefetch", lambda A, B: wmma_ops.matmul_noPrefetch(A, B)),
            ("rocBLAS", lambda A, B: torch.matmul(A, B)),
        ]
        
        for name, fn in variants:
            tflops, time_ms, status = benchmark_kernel(M, N, K, fn)
            if tflops:
                pct = tflops / 59.4 * 100
                print(f"  {name:<20} {tflops:6.1f} TFLOPS  {pct:5.1f}% peak  {status}")
            else:
                print(f"  {name:<20} {status}")


def main():
    parser = argparse.ArgumentParser(description="WMMA Kernel Auto-Tuner")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark only")
    parser.add_argument("--trials", type=int, default=20, help="Trials per size")
    parser.add_argument("--size", type=int, nargs=3, metavar=("M", "N", "K"),
                        help="Tune specific size")
    parser.add_argument("--generate", action="store_true", 
                        help="Generate C++ dispatch code")
    args = parser.parse_args()
    
    if not WMMA_AVAILABLE:
        print("Error: wmma_ops not available. Build first.")
        return
    
    if args.quick:
        run_quick_benchmark()
        return
    
    if not OPTUNA_AVAILABLE:
        print("Error: Optuna required for auto-tuning.")
        print("Install with: pip install optuna")
        return
    
    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    tuner = WMMAAutoTuner()
    
    if args.size:
        M, N, K = args.size
        result = tuner.tune_size(M, N, K, n_trials=args.trials)
        print(f"\nBest for {M}x{N}x{K}:")
        print(f"  Tile: {result['best_tile_desc']}")
        print(f"  TFLOPS: {result['best_tflops']:.1f}")
    else:
        results = tuner.tune_common_sizes(n_trials=args.trials)
        
        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)
        
        # Count tile config wins
        config_wins = {}
        for r in results:
            desc = r["best_tile_desc"]
            config_wins[desc] = config_wins.get(desc, 0) + 1
        
        print("\nOptimal tile by matrix size count:")
        for desc, count in sorted(config_wins.items(), key=lambda x: -x[1]):
            print(f"  {desc}: {count} sizes")
    
    if args.generate:
        code = tuner.generate_dispatch_code()
        print("\n" + "=" * 70)
        print("Generated C++ Dispatch Code")
        print("=" * 70)
        print(code)


if __name__ == "__main__":
    main()

