#!/usr/bin/env python3
"""
Benchmark Plotting Script for FVM Solver.

Reads benchmark/results.csv and generates 8 performance graphs.

Usage:
    cd /path/to/nir
    python3 benchmark/plot_results.py

CSV format:
    scenario,device,physics,nx,ny,steps,mpi_ranks,omp_threads,time_s
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "results.csv")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Visual style
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Run benchmark/run_all.sh first.")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["nx"] = pd.to_numeric(df["nx"], errors="coerce")
    df["ny"] = pd.to_numeric(df["ny"], errors="coerce")
    df["mpi_ranks"] = pd.to_numeric(df["mpi_ranks"], errors="coerce")
    df["omp_threads"] = pd.to_numeric(df["omp_threads"], errors="coerce")
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce")
    df["cells"] = df["nx"] * df["ny"]
    df["grid_label"] = df["nx"].astype(int).astype(str) + "x" + df["ny"].astype(int).astype(str)
    return df.dropna(subset=["time_s"])


# ── Plot 1: OpenMP Strong Scaling (Speedup) ────────────────────────
def plot_omp_scaling(df):
    data = df[df["scenario"] == "omp_scaling"].copy()
    if data.empty:
        print("  Skip: omp_scaling (no data)")
        return

    fig, ax = plt.subplots()
    grids = sorted(data["cells"].unique())

    for i, cells in enumerate(grids):
        sub = data[data["cells"] == cells].sort_values("omp_threads")
        threads = sub["omp_threads"].values
        times = sub["time_s"].values
        t1 = times[0]  # time with 1 thread
        speedup = t1 / times
        label = sub["grid_label"].iloc[0]
        ax.plot(threads, speedup, "o-", color=COLORS[i % len(COLORS)], label=label)

    # Ideal scaling line
    max_t = int(data["omp_threads"].max())
    ideal = np.arange(1, max_t + 1)
    ax.plot(ideal, ideal, "k--", alpha=0.4, label="Ideal")

    ax.set_xlabel("OpenMP Threads")
    ax.set_ylabel("Speedup S(p) = T(1) / T(p)")
    ax.set_title("OpenMP Strong Scaling (Euler, single MPI rank)")
    ax.legend()
    ax.set_xticks(sorted(data["omp_threads"].unique()))
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "1_omp_strong_scaling.png"))
    plt.close(fig)
    print("  [1] omp_strong_scaling.png")


# ── Plot 2: OpenMP Efficiency ──────────────────────────────────────
def plot_omp_efficiency(df):
    data = df[df["scenario"] == "omp_scaling"].copy()
    if data.empty:
        print("  Skip: omp_efficiency (no data)")
        return

    fig, ax = plt.subplots()
    grids = sorted(data["cells"].unique())

    for i, cells in enumerate(grids):
        sub = data[data["cells"] == cells].sort_values("omp_threads")
        threads = sub["omp_threads"].values
        times = sub["time_s"].values
        t1 = times[0]
        speedup = t1 / times
        efficiency = speedup / threads
        label = sub["grid_label"].iloc[0]
        ax.plot(threads, efficiency, "s-", color=COLORS[i % len(COLORS)], label=label)

    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.4, label="Ideal (100%)")
    ax.set_xlabel("OpenMP Threads")
    ax.set_ylabel("Efficiency E(p) = S(p) / p")
    ax.set_title("OpenMP Parallel Efficiency (Euler)")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.set_xticks(sorted(data["omp_threads"].unique()))
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "2_omp_efficiency.png"))
    plt.close(fig)
    print("  [2] omp_efficiency.png")


# ── Plot 3: GPU vs CPU Speedup (bar chart) ─────────────────────────
def plot_gpu_vs_cpu_speedup(df):
    data = df[df["scenario"] == "grid_scaling"].copy()
    if data.empty:
        print("  Skip: gpu_vs_cpu_speedup (no data)")
        return

    cpu = data[data["device"] == "cpu"].set_index("cells")["time_s"]
    gpu = data[data["device"] == "gpu"].set_index("cells")["time_s"]
    common = cpu.index.intersection(gpu.index)
    if common.empty:
        print("  Skip: gpu_vs_cpu_speedup (no matching grids)")
        return

    speedup = cpu.loc[common] / gpu.loc[common]
    labels = [f"{int(np.sqrt(c))}x{int(np.sqrt(c))}" for c in common]

    fig, ax = plt.subplots()
    bars = ax.bar(range(len(common)), speedup.values, color=COLORS[0], edgecolor="black", alpha=0.85)
    ax.set_xticks(range(len(common)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Speedup (GPU over CPU-48thr)")
    ax.set_title("GPU Speedup over CPU (Euler, 1x V100 vs 48 cores)")
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Break-even")

    # Add value labels on bars
    for bar, val in zip(bars, speedup.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}x", ha="center", va="bottom", fontsize=9)

    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "3_gpu_vs_cpu_speedup.png"))
    plt.close(fig)
    print("  [3] gpu_vs_cpu_speedup.png")


# ── Plot 4: Absolute Time vs Grid Size (log-log) ───────────────────
def plot_time_vs_grid(df):
    # Combine omp_scaling (1-thread) and grid_scaling data
    omp1 = df[(df["scenario"] == "omp_scaling") & (df["omp_threads"] == 1)].copy()
    grid = df[df["scenario"] == "grid_scaling"].copy()

    if grid.empty and omp1.empty:
        print("  Skip: time_vs_grid (no data)")
        return

    fig, ax = plt.subplots()

    # CPU 1-thread from omp_scaling
    if not omp1.empty:
        sub = omp1.sort_values("cells")
        ax.plot(sub["cells"], sub["time_s"], "o-", color=COLORS[3], label="CPU 1 thread")

    # CPU 48-thread from grid_scaling
    cpu48 = grid[grid["device"] == "cpu"].sort_values("cells")
    if not cpu48.empty:
        ax.plot(cpu48["cells"], cpu48["time_s"], "s-", color=COLORS[0], label="CPU 48 threads")

    # GPU from grid_scaling
    gpu = grid[grid["device"] == "gpu"].sort_values("cells")
    if not gpu.empty:
        ax.plot(gpu["cells"], gpu["time_s"], "^-", color=COLORS[1], label="GPU (V100)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Grid Size (cells)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Execution Time vs Grid Size (Euler)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "4_time_vs_grid.png"))
    plt.close(fig)
    print("  [4] time_vs_grid.png")


# ── Plot 5: Throughput (cells*steps/s) ──────────────────────────────
def plot_throughput(df):
    grid = df[df["scenario"] == "grid_scaling"].copy()
    if grid.empty:
        print("  Skip: throughput (no data)")
        return

    grid["throughput"] = grid["cells"] * grid["steps"] / grid["time_s"]

    fig, ax = plt.subplots()

    cpu = grid[grid["device"] == "cpu"].sort_values("cells")
    if not cpu.empty:
        ax.plot(cpu["cells"], cpu["throughput"], "s-", color=COLORS[0], label="CPU 48 threads")

    gpu = grid[grid["device"] == "gpu"].sort_values("cells")
    if not gpu.empty:
        ax.plot(gpu["cells"], gpu["throughput"], "^-", color=COLORS[1], label="GPU (V100)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Grid Size (cells)")
    ax.set_ylabel("Throughput (cell-updates / s)")
    ax.set_title("Computational Throughput (Euler)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "5_throughput.png"))
    plt.close(fig)
    print("  [5] throughput.png")


# ── Plot 6: MPI Strong Scaling (CPU) ───────────────────────────────
def plot_mpi_cpu_scaling(df):
    data = df[df["scenario"] == "mpi_cpu"].copy()
    if data.empty:
        print("  Skip: mpi_cpu_scaling (no data)")
        return

    data = data.sort_values("mpi_ranks")
    ranks = data["mpi_ranks"].values
    times = data["time_s"].values
    t1 = times[0]
    speedup = t1 / times

    fig, ax = plt.subplots()
    ax.plot(ranks, speedup, "o-", color=COLORS[0], label="Measured", markersize=8)
    ax.plot(ranks, ranks / ranks[0], "k--", alpha=0.4, label="Ideal")

    ax.set_xlabel("MPI Ranks")
    ax.set_ylabel("Speedup S(N) = T(1) / T(N)")
    ax.set_title("MPI Strong Scaling, CPU (2000x2000 Euler, 12 OMP threads/rank)")
    ax.set_xticks(ranks)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "6_mpi_cpu_scaling.png"))
    plt.close(fig)
    print("  [6] mpi_cpu_scaling.png")


# ── Plot 7: MPI + GPU Scaling ──────────────────────────────────────
def plot_mpi_gpu_scaling(df):
    data = df[df["scenario"] == "mpi_gpu"].copy()
    if data.empty:
        print("  Skip: mpi_gpu_scaling (no data)")
        return

    data = data.sort_values("mpi_ranks")
    ranks = data["mpi_ranks"].values
    times = data["time_s"].values
    t1 = times[0]
    speedup = t1 / times

    fig, ax = plt.subplots()
    ax.bar(range(len(ranks)), speedup, tick_label=[f"{int(r)} GPU" for r in ranks],
           color=COLORS[1], edgecolor="black", alpha=0.85)

    for i, (r, s) in enumerate(zip(ranks, speedup)):
        ax.text(i, s + 0.02, f"{s:.2f}x", ha="center", va="bottom", fontsize=10)

    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Speedup")
    ax.set_title("Multi-GPU Scaling (2000x2000 Euler, MPI + CUDA)")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "7_mpi_gpu_scaling.png"))
    plt.close(fig)
    print("  [7] mpi_gpu_scaling.png")


# ── Plot 8: Weak Scaling (CPU) ─────────────────────────────────────
def plot_weak_scaling(df):
    data = df[df["scenario"] == "weak_scaling"].copy()
    if data.empty:
        print("  Skip: weak_scaling (no data)")
        return

    data = data.sort_values("mpi_ranks")
    ranks = data["mpi_ranks"].values
    times = data["time_s"].values
    t1 = times[0]
    efficiency = t1 / times  # ideal weak scaling: time stays constant

    fig, ax = plt.subplots()
    ax.plot(ranks, efficiency, "o-", color=COLORS[2], markersize=8, label="Measured")
    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.4, label="Ideal")

    ax.set_xlabel("MPI Ranks (grid grows proportionally)")
    ax.set_ylabel("Weak Scaling Efficiency T(1) / T(N)")
    ax.set_title("Weak Scaling, CPU (500x500 per rank, Euler)")
    ax.set_xticks(ranks)
    ax.set_ylim(0, 1.3)
    ax.legend()

    # Add grid labels
    for i, row in data.iterrows():
        ax.annotate(f"{int(row['nx'])}x{int(row['ny'])}",
                     (row["mpi_ranks"], efficiency[list(ranks).index(row["mpi_ranks"])]),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "8_weak_scaling.png"))
    plt.close(fig)
    print("  [8] weak_scaling.png")


# ── Bonus: Physics Comparison ───────────────────────────────────────
def plot_physics_comparison(df):
    data = df[df["scenario"] == "physics"].copy()
    if data.empty:
        print("  Skip: physics_comparison (no data)")
        return

    fig, ax = plt.subplots()
    x_labels = []
    cpu_times = []
    gpu_times = []

    for phys in ["heat", "euler"]:
        sub = data[data["physics"] == phys]
        cpu_t = sub[sub["device"] == "cpu"]["time_s"].values
        gpu_t = sub[sub["device"] == "gpu"]["time_s"].values
        if len(cpu_t) > 0 and len(gpu_t) > 0:
            x_labels.append(phys.capitalize())
            cpu_times.append(cpu_t[0])
            gpu_times.append(gpu_t[0])

    if not x_labels:
        print("  Skip: physics_comparison (incomplete data)")
        return

    x = np.arange(len(x_labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, cpu_times, w, label="CPU 48 threads", color=COLORS[0], edgecolor="black")
    bars2 = ax.bar(x + w/2, gpu_times, w, label="GPU (V100)", color=COLORS[1], edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Time (s)")
    ax.set_title("Physics Type Comparison (1000x1000 grid)")
    ax.legend()

    # Speedup annotations
    for i in range(len(x_labels)):
        spd = cpu_times[i] / gpu_times[i]
        ax.annotate(f"GPU {spd:.1f}x faster",
                     xy=(x[i] + w/2, gpu_times[i]),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=8, color=COLORS[1])

    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "9_physics_comparison.png"))
    plt.close(fig)
    print("  [9] physics_comparison.png")


# ── Main ────────────────────────────────────────────────────────────
def main():
    print(f"Reading: {CSV_PATH}")
    df = load_data()
    print(f"Loaded {len(df)} benchmark records")
    print(f"Scenarios: {sorted(df['scenario'].unique())}")
    print(f"Generating plots -> {PLOT_DIR}/")
    print()

    plot_omp_scaling(df)
    plot_omp_efficiency(df)
    plot_gpu_vs_cpu_speedup(df)
    plot_time_vs_grid(df)
    plot_throughput(df)
    plot_mpi_cpu_scaling(df)
    plot_mpi_gpu_scaling(df)
    plot_weak_scaling(df)
    plot_physics_comparison(df)

    print(f"\nDone! All plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
