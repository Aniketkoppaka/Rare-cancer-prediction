"""
Master Reproduction Suite: Active Phenotype Acquisition Benchmark

Executes the primary pipeline checks and benchmarks sequentially:
1. Data integrity and schema audit
2. Primary acquisition policy control (Q=5)
3. Fast vectorized adaptive margin acquisition
4. Cross-validation and clinical noise stress test
"""

import subprocess
import sys
import time

STEPS = [
    ("1. Data Cleanliness & Null Value Audit", "src/audit_data_integrity.py"),
    ("2. Primary Acquisition Policy Control (Q=5)", "src/benchmark_primary_control.py"),
    ("3. Vectorized Adaptive Margin Policy (Fast Inverted-Index)", "src/benchmark_adaptive_margin.py"),
    ("4. 5-Fold Cross-Validation & Clinical EHR Noise Stress Test", "src/audit_crossval_and_noise.py"),
]

def main():
    print("=" * 80)
    print("ACTIVE PHENOTYPE ACQUISITION: MASTER REPRODUCIBILITY SUITE")
    print("=" * 80)
    
    start_total = time.time()
    for idx, (title, script) in enumerate(STEPS, 1):
        print(f"\n[{idx}/{len(STEPS)}] Running {title} ({script})...")
        t0 = time.time()
        ret = subprocess.run([sys.executable, script])
        dt = time.time() - t0
        if ret.returncode != 0:
            print(f"FAILED: {script} exited with error code {ret.returncode}")
            sys.exit(ret.returncode)
        print(f"COMPLETED in {dt:.2f} seconds.")
        
    total_time = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"ALL BENCHMARKS VERIFIED SUCCESSFULLY (Total elapsed: {total_time:.2f}s)")
    print("=" * 80)

if __name__ == "__main__":
    main()
