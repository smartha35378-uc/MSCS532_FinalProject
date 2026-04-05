# Data Structure Optimization in HPC: AoS vs. SoA in Python

## Overview
This project is a small prototype for an assignment on data structure optimization in high-performance computing (HPC). It examines how changing the memory layout of a particle simulation workload can improve performance.

The optimization technique demonstrated here is the replacement of an **Array of Structures (AoS)** layout with a **Structure of Arrays (SoA)** layout. In this repository:

- `AoS` stores each particle as a Python dictionary inside a list.
- `SoA` stores each particle attribute in a separate NumPy array.

Both implementations perform the same particle-update computation, but the SoA version is designed to improve memory locality and reduce Python interpreter overhead.

## Assignment Focus
The goal of this project is to connect HPC optimization theory with a practical implementation. The repository supports the following assignment objectives:

- examine an optimization technique used for data structures in HPC
- justify why the technique is useful and relevant
- implement a small prototype that demonstrates the technique
- compare behavior before and after optimization
- discuss strengths, weaknesses, lessons learned, and observed results

## Chosen Optimization Technique
The selected optimization technique is **data layout optimization through Structure of Arrays (SoA)**.

### Why this technique was chosen
This technique is especially relevant in HPC because performance often depends on:

- memory locality
- cache efficiency
- reduced pointer chasing and object indirection
- the ability to use vectorized or SIMD-style operations

An AoS layout is easy to understand, but it is often inefficient for numerical workloads because each element may contain multiple fields stored through separate Python objects. A SoA layout groups each field into a contiguous block of memory, which is friendlier to modern processors and numerical libraries such as NumPy.

In this project, SoA was chosen because it is both conceptually important in HPC and practical to demonstrate clearly in Python.

## Strengths and Weaknesses of the Technique
### Strengths
- Improves spatial locality by storing like values together.
- Reduces Python object overhead compared with a list of dictionaries.
- Works naturally with NumPy vectorized operations.
- Scales better as dataset size increases.
- Makes bulk updates over one field much more efficient.

### Weaknesses
- Can be less intuitive than AoS for beginners.
- Managing multiple arrays requires more care to keep data aligned.
- Not every algorithm benefits equally from SoA.
- In Python, part of the improvement comes from NumPy vectorization as well as layout changes, so the effects are related rather than perfectly isolated.

## Implementation Summary
The prototype simulates repeated updates to a set of particles with the following values:

- position: `x`, `y`
- velocity: `vx`, `vy`
- mass: `mass`

During each time step, velocity is updated using acceleration and damping, then position is updated from velocity.

### AoS version
The AoS implementation in [src/aos.py](/Users/prithviraj/Documents/Jobs/GitHub/MSCS532_FinalProject%20/MSCS532_FinalProject/src/aos.py) stores particles as:

```python
[
    {"x": ..., "y": ..., "vx": ..., "vy": ..., "mass": ...},
    ...
]
```

This version uses nested Python loops and dictionary lookups for every particle and every step.

### SoA version
The SoA implementation in [src/soa.py](/Users/prithviraj/Documents/Jobs/GitHub/MSCS532_FinalProject%20/MSCS532_FinalProject/src/soa.py) stores particles as:

```python
{
    "x": np.array(...),
    "y": np.array(...),
    "vx": np.array(...),
    "vy": np.array(...),
    "mass": np.array(...),
}
```

This version performs the same updates using NumPy arrays and vectorized arithmetic.

## Project Structure
```text
MSCS532_FinalProject/
├── README.md
├── requirements.txt
├── results/
│   └── benchmark_results.csv
├── src/
│   ├── aos.py
│   ├── benchmark.py
│   ├── main.py
│   ├── soa.py
│   └── utils.py
└── tests/
    └── test_correctness.py
```

## Requirements
- Python 3.10+
- NumPy 1.26+

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run
Run the benchmark:

```bash
python src/main.py
```

Run the test suite:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Benchmark Method
The benchmark logic is implemented in [src/benchmark.py](/Users/prithviraj/Documents/Jobs/GitHub/MSCS532_FinalProject%20/MSCS532_FinalProject/src/benchmark.py).

For each input size, the benchmark:

- generates equivalent AoS and SoA particle datasets
- performs repeated update steps on each version
- measures execution time with `time.perf_counter()`
- uses the median of repeated runs for stability
- validates that both implementations produce equivalent numerical results
- writes results to [results/benchmark_results.csv](/Users/prithviraj/Documents/Jobs/GitHub/MSCS532_FinalProject%20/MSCS532_FinalProject/results/benchmark_results.csv)

Default benchmark settings from [src/utils.py](/Users/prithviraj/Documents/Jobs/GitHub/MSCS532_FinalProject%20/MSCS532_FinalProject/src/utils.py):

- particle sizes: `10,000`, `50,000`, `100,000`, `250,000`
- update steps: `20`
- repeats: `5`
- seed: `42`

## Results
The current saved benchmark results are:

| Particles | AoS Median (s) | SoA Median (s) | Speedup | Equivalent |
| --- | ---: | ---: | ---: | ---: |
| 10,000 | 0.081366 | 0.001414 | 57.55x | Yes |
| 50,000 | 0.400884 | 0.005382 | 74.49x | Yes |
| 100,000 | 0.846086 | 0.011019 | 76.78x | Yes |
| 250,000 | 2.186994 | 0.040673 | 53.77x | Yes |

### Interpretation
The SoA implementation is consistently much faster than the AoS version for every tested dataset size. This supports the main HPC idea behind the assignment: **data layout matters**. When data is stored in a contiguous, array-oriented form, numerical updates can be performed far more efficiently than when data is stored as many small Python objects.

## Discussion
### Problems encountered
- Ensuring that both implementations started from exactly equivalent data.
- Separating correctness verification from performance measurement.
- Making sure the benchmark used repeated trials so results were not based on a single noisy run.
- Interpreting the improvement carefully, since Python performance depends heavily on both memory layout and the use of optimized NumPy kernels.

### How the technique was applied
The main optimization was changing the representation of particle data:

- from a list of dictionaries with per-particle updates
- to parallel NumPy arrays with field-wise vectorized updates

This change reduced repeated dictionary access, reduced Python loop overhead, and enabled more cache-friendly computation.

### Observed performance improvements
The benchmark shows very large speedups, generally between about `54x` and `77x`. In practical terms, the optimized version completes the same workload in a small fraction of the time required by the baseline version.

## Lessons Learned
- Theoretical HPC expectations about memory locality and contiguous storage were strongly reflected in the experimental results.
- In Python, optimization is closely tied to choosing the right data structure for the right execution model.
- The gap between theory and practice is that the speedup here is not caused by layout alone; it is also amplified by NumPy's compiled vectorized operations.
- Even a simple prototype can clearly demonstrate why data structure design is critical for performance-sensitive workloads.

## Conclusion
This project demonstrates that replacing an AoS layout with an SoA layout can produce major performance improvements for numerical workloads in Python. The prototype supports the broader HPC principle that data structure organization directly affects runtime efficiency, especially when the optimized layout enables contiguous storage and vectorized computation.
