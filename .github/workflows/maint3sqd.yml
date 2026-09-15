"""
T3-SQD Synthetic Benchmark
--------------------------
Toy, falsifiable benchmark for testing the T3 hypothesis:

"In a dynamically changing constrained search space, preserve unaffected
high-value structure and locally repair only affected regions."

This is NOT a quantum-chemistry or Qiskit SQD implementation.
It is a synthetic graph benchmark intended to test the algorithmic idea
before porting it to QGSS 2026 Lab 4c / qiskit-addon-sqd.

Outputs:
  - per_run_summary.csv
  - locality_sweep_summary.csv

Requires:
  numpy
"""

import csv
from pathlib import Path
import numpy as np


def make_knn_graph(n=2000, k=8, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    nbrs = np.argpartition(d2, k, axis=1)[:, :k]

    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in nbrs[i]:
            j = int(j)
            adj[i].add(j)
            adj[j].add(i)
    return pts, adj


def neighbors_within(adj, seeds, radius=1):
    current = set(seeds)
    seen = set(seeds)
    for _ in range(radius):
        nxt = set()
        for u in current:
            nxt.update(adj[u])
        nxt -= seen
        seen |= nxt
        current = nxt
    return seen


def jaccard_disruption(a, b):
    union = len(a | b)
    if union == 0:
        return 0.0
    return 1.0 - len(a & b) / union


def run_benchmark(
    seed=0,
    n=2000,
    budget=200,
    iterations=30,
    k=8,
    local_fraction=0.03,
    repair_radius=1,
    exploration_budget=30,
    observation_noise=0.03,
    terrain_change_scale=0.25,
):
    rng = np.random.default_rng(seed)
    pts, adj = make_knn_graph(n=n, k=k, seed=seed)

    # Smooth "quality landscape" + determinant-specific variation.
    centers = np.array([[0.25, 0.25], [0.72, 0.65], [0.45, 0.82]])
    amplitudes = np.array([1.0, 0.85, 0.7])
    widths = np.array([0.10, 0.13, 0.08])

    base = np.zeros(n)
    for c, a, w in zip(centers, amplitudes, widths):
        base += a * np.exp(-((pts - c) ** 2).sum(1) / (2 * w * w))
    base += 0.10 * rng.normal(size=n)

    terrain = np.zeros(n)
    true_score = base + terrain

    initial_obs = true_score + rng.normal(0, observation_noise, n)
    baseline_sel = set(np.argpartition(initial_obs, -budget)[-budget:])

    # T3 keeps a cached estimate for unaffected nodes.
    t3_cache = initial_obs.copy()
    t3_sel = set(baseline_sel)

    rows = []
    prev_baseline = baseline_sel.copy()
    prev_t3 = t3_sel.copy()

    for t in range(1, iterations + 1):
        # Local terrain disturbance.
        center = pts[rng.integers(n)]
        d2 = ((pts - center) ** 2).sum(axis=1)
        changed_count = max(1, int(local_fraction * n))
        changed = set(np.argpartition(d2, changed_count)[:changed_count])

        delta = rng.normal(0, terrain_change_scale)
        idx = np.array(list(changed), dtype=int)
        distances = np.sqrt(d2[idx])
        scale = np.exp(-(distances / (distances.max() + 1e-12)) ** 2)
        terrain[idx] += delta * scale
        true_score = base + terrain

        # Baseline: global re-evaluation and global selection.
        baseline_obs = true_score + rng.normal(0, observation_noise, n)
        baseline_sel = set(np.argpartition(baseline_obs, -budget)[-budget:])

        # T3: identify affected local region, repair it, and use a small
        # exploration budget to reduce the chance of stale blind spots.
        affected = neighbors_within(adj, changed, radius=repair_radius)

        frontier = set()
        for u in affected:
            frontier.update(adj[u])
        frontier -= affected

        half = exploration_budget // 2
        explore_frontier = set()
        if frontier:
            take = min(half, len(frontier))
            explore_frontier = set(
                rng.choice(list(frontier), size=take, replace=False)
            )

        remaining = list(set(range(n)) - affected - explore_frontier)
        remaining_budget = exploration_budget - len(explore_frontier)
        explore_random = set()
        if remaining and remaining_budget > 0:
            take = min(remaining_budget, len(remaining))
            explore_random = set(
                rng.choice(remaining, size=take, replace=False)
            )

        rescored = affected | explore_frontier | explore_random
        r = np.array(list(rescored), dtype=int)
        t3_cache[r] = true_score[r] + rng.normal(0, observation_noise, len(r))
        t3_sel = set(np.argpartition(t3_cache, -budget)[-budget:])

        # Exact oracle is used ONLY for evaluation in the synthetic benchmark.
        exact = set(np.argpartition(true_score, -budget)[-budget:])
        exact_value = true_score[list(exact)].sum()
        baseline_value = true_score[list(baseline_sel)].sum()
        t3_value = true_score[list(t3_sel)].sum()

        baseline_regret = (exact_value - baseline_value) / abs(exact_value)
        t3_regret = (exact_value - t3_value) / abs(exact_value)

        rows.append(
            {
                "iteration": t,
                "baseline_regret": baseline_regret,
                "t3_regret": t3_regret,
                "baseline_disruption": jaccard_disruption(
                    prev_baseline, baseline_sel
                ),
                "t3_disruption": jaccard_disruption(prev_t3, t3_sel),
                "t3_rescored_nodes": len(rescored),
                "t3_rescored_fraction": len(rescored) / n,
                "t3_exact_overlap": len(t3_sel & exact) / budget,
            }
        )

        prev_baseline = baseline_sel.copy()
        prev_t3 = t3_sel.copy()

    return rows


def summarize(rows):
    keys = [
        "baseline_regret",
        "t3_regret",
        "baseline_disruption",
        "t3_disruption",
        "t3_rescored_nodes",
        "t3_rescored_fraction",
        "t3_exact_overlap",
    ]
    out = {}
    for key in keys:
        out[key] = float(np.mean([r[key] for r in rows]))
    return out


def main():
    outdir = Path("t3_sqd_results")
    outdir.mkdir(exist_ok=True)

    # Main replication.
    run_summaries = []
    for seed in range(30):
        rows = run_benchmark(seed=seed)
        s = summarize(rows)
        s["seed"] = seed
        run_summaries.append(s)

    with (outdir / "per_run_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_summaries[0].keys())
        writer.writeheader()
        writer.writerows(run_summaries)

    # Stress test: how performance changes as disturbances become less local.
    sweep_rows = []
    for fraction in [0.01, 0.03, 0.10, 0.25, 0.50]:
        summaries = []
        for seed in range(12):
            rows = run_benchmark(
                seed=1000 + seed,
                n=1500,
                budget=150,
                iterations=20,
                local_fraction=fraction,
            )
            summaries.append(summarize(rows))

        row = {"local_fraction": fraction}
        for key in summaries[0]:
            row[key] = float(np.mean([s[key] for s in summaries]))
        sweep_rows.append(row)

    with (outdir / "locality_sweep_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
        writer.writeheader()
        writer.writerows(sweep_rows)

    print("Finished.")
    print("Main 30-seed average:")
    main_avg = {}
    for key in run_summaries[0]:
        if key != "seed":
            main_avg[key] = np.mean([r[key] for r in run_summaries])
            print(f"  {key}: {main_avg[key]:.6f}")

    print("\nLocality sweep:")
    for row in sweep_rows:
        print(
            f"  change={row['local_fraction']:.0%} | "
            f"T3 regret={row['t3_regret']:.5f} | "
            f"disruption={row['t3_disruption']:.3f} | "
            f"rescored={row['t3_rescored_fraction']:.1%}"
        )


if __name__ == "__main__":
    main()
