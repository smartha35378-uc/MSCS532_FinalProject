from __future__ import annotations

from pathlib import Path

from benchmark import run_benchmarks
from utils import BenchmarkConfig


def main() -> None:
    config = BenchmarkConfig()
    project_root = Path(__file__).resolve().parent.parent
    output_csv = project_root / "results" / "benchmark_results.csv"

    # Run the benchmark and print a simple summary table.
    rows = run_benchmarks(config=config, output_csv=output_csv)

    print("HPC Optimization Prototype: AoS vs SoA")
    print("-" * 55)
    print(f"{'Particles':>12} | {'AoS (s)':>12} | {'SoA (s)':>12} | {'Speedup':>10} | {'Match':>5}")
    print("-" * 55)
    for row in rows:
        match = "Yes" if int(row["equivalent"]) == 1 else "No"
        print(
            f"{int(row['n']):>12} | "
            f"{row['aos_median_seconds']:>12.6f} | "
            f"{row['soa_median_seconds']:>12.6f} | "
            f"{row['speedup_x']:>10.2f}x | "
            f"{match:>5}"
        )
    print("-" * 55)
    print(f"Results saved to: {output_csv}")


if __name__ == "__main__":
    main()
