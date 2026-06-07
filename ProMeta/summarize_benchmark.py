import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["auroc", "auprc", "f1", "accuracy", "precision", "recall"]
PRIMARY_MODELS = ("ProMeta", "TSA-ProMeta")
TSA_CONFIG_COLUMNS = [
    "tsa_selector_source",
    "tsa_assignment_source",
    "tsa_distance_mode",
    "tsa_gate_distance_weight",
    "tsa_selector_l1_lambda",
    "tsa_routing_schedule",
    "tsa_switch_threshold",
    "tsa_min_group_fraction",
    "tsa_max_group_fraction",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize ProMeta vs TSA-ProMeta benchmark JSON files.")
    parser.add_argument("--output_dir", type=str, default="./experiments_output", help="Experiment output directory.")
    parser.add_argument("--benchmark_dir", type=str, default="", help="Directory containing support_*/ result JSON files.")
    parser.add_argument("--summary_dir", type=str, default="", help="Directory for summary CSV and plots.")
    parser.add_argument("--bootstrap_iterations", type=int, default=5000, help="Bootstrap iterations for 95%% CIs.")
    parser.add_argument("--bootstrap_seed", type=int, default=2026, help="Random seed for bootstrap CIs.")
    parser.add_argument("--include_all_runs", action="store_true", help="Include duplicate reruns instead of keeping latest result per model/shot/seed.")
    return parser.parse_args()


def as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def load_json_results(benchmark_dir):
    paths = sorted(Path(benchmark_dir).glob("support_*/*.json"))
    if not paths:
        raise FileNotFoundError(f"No result JSON files found under {benchmark_dir}/support_*")

    run_rows = []
    task_rows = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)

        config = data.get("config", {})
        model = data.get("model", config.get("model", ""))
        support_size = as_int(data.get("support_size", config.get("support_size")))
        max_support_size = as_int(data.get("max_support_size", config.get("max_support_size")))
        seed = as_int(data.get("seed", config.get("random_seed")))
        timestamp = data.get("timestamp", "")
        experiment_name = data.get("experiment_name", config.get("experiment_name", "ProMeta"))
        selector_source = config.get("tsa_selector_source")
        assignment_source = config.get("tsa_assignment_source")
        distance_mode = config.get("tsa_distance_mode")
        gate_distance_weight = config.get("tsa_gate_distance_weight")
        selector_l1_lambda = config.get("tsa_selector_l1_lambda")
        routing_schedule = config.get("tsa_routing_schedule")
        switch_threshold = config.get("tsa_switch_threshold")
        min_group_fraction = config.get("tsa_min_group_fraction")
        max_group_fraction = config.get("tsa_max_group_fraction")

        base = {
            "json_path": str(path),
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            "model": model,
            "support_size": support_size,
            "max_support_size": max_support_size,
            "seed": seed,
            "tsa_selector_source": selector_source,
            "tsa_assignment_source": assignment_source,
            "tsa_distance_mode": distance_mode,
            "tsa_gate_distance_weight": gate_distance_weight,
            "tsa_selector_l1_lambda": selector_l1_lambda,
            "tsa_routing_schedule": routing_schedule,
            "tsa_switch_threshold": switch_threshold,
            "tsa_min_group_fraction": min_group_fraction,
            "tsa_max_group_fraction": max_group_fraction,
        }

        summary = data.get("summary_metrics", {})
        run_row = dict(base)
        for metric in METRICS:
            run_row[f"test_{metric}"] = summary.get(metric)
        run_rows.append(run_row)

        for task in data.get("per_task_details", []):
            task_row = dict(base)
            task_row["disease_term"] = task.get("disease_term")
            task_row["tsa_group"] = task.get("tsa_group")
            task_row["tsa_distance"] = task.get("tsa_distance")
            task_row["tsa_margin"] = task.get("tsa_margin")
            for metric in METRICS:
                task_row[metric] = task.get(metric)
            task_rows.append(task_row)

    return pd.DataFrame(run_rows), pd.DataFrame(task_rows)


def keep_latest_runs(run_df, task_df):
    if run_df.empty:
        return run_df, task_df

    identity_cols = [
        "support_size",
        "seed",
        "model",
        "experiment_name",
        *TSA_CONFIG_COLUMNS,
    ]
    sort_cols = [*identity_cols, "timestamp", "json_path"]
    latest = (
        run_df.sort_values(sort_cols)
        .drop_duplicates(identity_cols, keep="last")
        .copy()
    )
    latest_paths = set(latest["json_path"])
    latest_tasks = task_df[task_df["json_path"].isin(latest_paths)].copy()
    return latest, latest_tasks


def bootstrap_mean_ci(values, iterations=5000, seed=2026):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1 or iterations <= 0:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def rank_abs_values(values):
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def wilcoxon_pvalue(deltas):
    values = np.asarray(deltas, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values != 0]
    if len(values) == 0:
        return 1.0, "all_zero"

    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(values, zero_method="wilcox", alternative="two-sided")
        return float(result.pvalue), "scipy"
    except Exception:
        abs_values = np.abs(values)
        ranks = rank_abs_values(abs_values)
        w_plus = ranks[values > 0].sum()
        w_minus = ranks[values < 0].sum()
        statistic = min(w_plus, w_minus)
        n = len(values)
        mean = n * (n + 1) / 4.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0
        if variance <= 0:
            return np.nan, "normal_approx"
        z = (statistic - mean + 0.5) / math.sqrt(variance)
        pvalue = 2.0 * 0.5 * math.erfc(abs(z) / math.sqrt(2.0))
        return float(min(max(pvalue, 0.0), 1.0)), "normal_approx"


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(pvalues), np.nan, dtype=float)
    valid = np.where(np.isfinite(pvalues))[0]
    if len(valid) == 0:
        return adjusted

    order = valid[np.argsort(pvalues[valid])]
    running_max = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        corrected = (m - rank) * pvalues[idx]
        running_max = max(running_max, corrected)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def build_summary(task_df, iterations, seed):
    if task_df.empty:
        return pd.DataFrame(columns=["support_size", "model", "seed_count", "task_count", "disease_count"])

    rows = []
    group_cols = ["support_size", "model", "experiment_name", *TSA_CONFIG_COLUMNS]
    for group_key, group in task_df.groupby(group_cols, dropna=False):
        group_values = dict(zip(group_cols, group_key))
        row = {
            **group_values,
            "seed_count": group["seed"].nunique(),
            "task_count": len(group),
            "disease_count": group["disease_term"].nunique(),
        }
        for metric in METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            row[f"mean_{metric}"] = float(values.mean()) if len(values) else np.nan
            row[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            low, high = bootstrap_mean_ci(values, iterations=iterations, seed=seed)
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def build_paired_delta(task_df):
    if task_df.empty or "model" not in task_df:
        return pd.DataFrame()

    base = task_df[task_df["model"] == PRIMARY_MODELS[0]].copy()
    tsa = task_df[task_df["model"] == PRIMARY_MODELS[1]].copy()
    keys = ["support_size", "seed", "disease_term"]

    base_cols = keys + [m for m in METRICS if m in base.columns]
    tsa_cols = (
        keys
        + [col for col in ["experiment_name", *TSA_CONFIG_COLUMNS] if col in tsa.columns]
        + [m for m in METRICS if m in tsa.columns]
        + ["tsa_group", "tsa_distance", "tsa_margin"]
    )
    tsa_cols = [col for col in tsa_cols if col in tsa.columns]

    paired = base[base_cols].merge(
        tsa[tsa_cols],
        on=keys,
        how="inner",
        suffixes=("_baseline", "_tsa"),
    )

    for metric in METRICS:
        baseline_col = f"{metric}_baseline"
        tsa_col = f"{metric}_tsa"
        if baseline_col in paired.columns and tsa_col in paired.columns:
            paired[f"delta_{metric}"] = paired[tsa_col] - paired[baseline_col]

    return paired.sort_values(keys)


def build_stat_tests(paired_df, iterations, seed):
    if paired_df.empty:
        return pd.DataFrame()

    rows = []
    for metric in METRICS:
        delta_col = f"delta_{metric}"
        if delta_col not in paired_df:
            continue

        group_cols = [
            "support_size",
            *[col for col in ["experiment_name", *TSA_CONFIG_COLUMNS] if col in paired_df.columns],
        ]
        for group_key, group in paired_df.groupby(group_cols, dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_values = dict(zip(group_cols, group_key))
            values = pd.to_numeric(group[delta_col], errors="coerce").dropna().to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, iterations=iterations, seed=seed)
            pvalue, method = wilcoxon_pvalue(values)
            rows.append({
                "metric": metric,
                **group_values,
                "n_pairs": int(len(values)),
                "mean_delta": float(values.mean()) if len(values) else np.nan,
                "ci95_low": low,
                "ci95_high": high,
                "wilcoxon_p": pvalue,
                "wilcoxon_method": method,
                "positive_pairs": int((values > 0).sum()),
                "negative_pairs": int((values < 0).sum()),
                "zero_pairs": int((values == 0).sum()),
            })

    tests = pd.DataFrame(rows)
    if tests.empty:
        return tests

    tests["holm_p"] = np.nan
    for metric, group in tests.groupby("metric"):
        adjusted = holm_adjust(group["wilcoxon_p"].to_numpy(dtype=float))
        tests.loc[group.index, "holm_p"] = adjusted
    return tests.sort_values(["metric", "support_size", "experiment_name"])


def plot_metric_curve(task_df, metric, output_path, iterations, seed):
    if task_df.empty or metric not in task_df:
        return

    per_seed = (
        task_df.dropna(subset=[metric])
        .groupby(["support_size", "seed", "model"], as_index=False)[metric]
        .mean()
    )
    if per_seed.empty:
        return

    plt.figure(figsize=(7, 5))
    for model in PRIMARY_MODELS:
        model_df = per_seed[per_seed["model"] == model]
        if model_df.empty:
            continue

        xs = []
        ys = []
        lower = []
        upper = []
        for support_size, group in model_df.groupby("support_size"):
            values = group[metric].to_numpy(dtype=float)
            mean_value = float(values.mean())
            low, high = bootstrap_mean_ci(values, iterations=iterations, seed=seed)
            xs.append(support_size)
            ys.append(mean_value)
            lower.append(mean_value - low)
            upper.append(high - mean_value)

        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ys = np.asarray(ys)[order]
        yerr = np.vstack([np.asarray(lower)[order], np.asarray(upper)[order]])
        plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, linewidth=2, label=model)

    plt.xlabel("Support size")
    plt.ylabel(f"Test {metric.upper()}")
    plt.title(f"ProMeta vs TSA-ProMeta: {metric.upper()}")
    plt.xticks(sorted(per_seed["support_size"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    benchmark_dir = Path(args.benchmark_dir) if args.benchmark_dir else output_dir / "benchmark_results"
    summary_dir = Path(args.summary_dir) if args.summary_dir else output_dir / "benchmark_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    run_df, task_df = load_json_results(benchmark_dir)
    if not args.include_all_runs:
        run_df, task_df = keep_latest_runs(run_df, task_df)

    run_df.to_csv(summary_dir / "run_metrics.csv", index=False)
    task_df.to_csv(summary_dir / "task_metrics.csv", index=False)

    summary_df = build_summary(task_df, args.bootstrap_iterations, args.bootstrap_seed)
    paired_df = build_paired_delta(task_df)
    tests_df = build_stat_tests(paired_df, args.bootstrap_iterations, args.bootstrap_seed)

    summary_df.to_csv(summary_dir / "summary_metrics.csv", index=False)
    paired_df.to_csv(summary_dir / "paired_task_delta.csv", index=False)
    tests_df.to_csv(summary_dir / "statistical_tests.csv", index=False)

    plot_metric_curve(task_df, "auroc", summary_dir / "shot_curve_auroc.png", args.bootstrap_iterations, args.bootstrap_seed)
    plot_metric_curve(task_df, "auprc", summary_dir / "shot_curve_auprc.png", args.bootstrap_iterations, args.bootstrap_seed)

    print(f"Loaded {len(run_df)} run files and {len(task_df)} per-task rows.")
    print(f"Wrote benchmark summaries to: {summary_dir}")


if __name__ == "__main__":
    main()
