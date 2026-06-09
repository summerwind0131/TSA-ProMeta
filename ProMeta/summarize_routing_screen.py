import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize TSA routing-screen runs by validation performance."
    )
    parser.add_argument("--root", required=True, help="Root containing variant subdirectories")
    return parser.parse_args()


def value_at(history, key, index, default=None):
    values = history.get(key)
    if not isinstance(values, list) or index >= len(values):
        return default
    return values[index]


def latest_run_files(root):
    paths = glob.glob(
        os.path.join(
            root,
            "*",
            "benchmark_results",
            "support_*",
            "TSA-ProMeta_seed*.json",
        )
    )
    latest = {}
    for path in paths:
        with open(path, encoding="utf-8-sig") as handle:
            data = json.load(handle)
        key = (
            data.get("experiment_name"),
            int(data.get("support_size")),
            int(data.get("seed")),
        )
        modified = os.path.getmtime(path)
        if key not in latest or modified > latest[key][0]:
            latest[key] = (modified, path, data)
    return [item[1:] for item in latest.values()]


def build_run_rows(runs):
    rows = []
    for path, data in runs:
        history = data.get("history") or {}
        val_aurocs = history.get("val_auroc") or []
        epoch0_val = history.get("epoch0_val_metrics") or {}
        epoch0_test = history.get("epoch0_test_metrics") or {}
        epoch1_test = history.get("epoch1_test_metrics") or {}
        if not val_aurocs and not epoch0_val:
            continue

        recorded_best_epoch = history.get("best_epoch")
        if recorded_best_epoch is None:
            trained_best_index = max(
                range(len(val_aurocs)),
                key=val_aurocs.__getitem__,
            )
            best_epoch = trained_best_index + 1
        else:
            best_epoch = int(recorded_best_epoch)

        if best_epoch == 0:
            best_index = None
            best_val_auroc = epoch0_val.get("auroc")
            best_val_auprc = epoch0_val.get("auprc")
            drift = []
        else:
            best_index = best_epoch - 1
            best_val_auroc = value_at(
                history,
                "val_auroc",
                best_index,
            )
            best_val_auprc = value_at(
                history,
                "val_auprc",
                best_index,
            )
            drift = value_at(
                history,
                "tsa_group_drift",
                best_index,
                [],
            ) or []

        config = data.get("config") or {}
        metrics = data.get("summary_metrics") or {}

        rows.append({
            "variant": data.get("experiment_name"),
            "support_size": data.get("support_size"),
            "seed": data.get("seed"),
            "best_epoch": best_epoch,
            "epochs_ran": len(val_aurocs),
            "epoch0_val_auroc": epoch0_val.get("auroc"),
            "epoch0_val_auprc": epoch0_val.get("auprc"),
            "epoch0_test_auroc": epoch0_test.get("auroc"),
            "epoch0_test_auprc": epoch0_test.get("auprc"),
            "epoch1_val_auroc": value_at(history, "val_auroc", 0),
            "epoch1_val_auprc": value_at(history, "val_auprc", 0),
            "epoch1_test_auroc": epoch1_test.get("auroc"),
            "epoch1_test_auprc": epoch1_test.get("auprc"),
            "best_val_auroc": best_val_auroc,
            "val_auprc_at_best": best_val_auprc,
            "test_auroc": metrics.get("auroc"),
            "test_auprc": metrics.get("auprc"),
            "group_usage": json.dumps(
                (
                    value_at(history, "tsa_group_usage", best_index, [])
                    if best_index is not None
                    else history.get("tsa_cluster_counts", [])
                ),
                separators=(",", ":"),
            ),
            "switch_rate": value_at(
                history,
                "tsa_group_switch_rate",
                best_index,
            ) if best_index is not None else None,
            "hysteresis_retention_rate": value_at(
                history,
                "tsa_hysteresis_retention_rate",
                best_index,
            ) if best_index is not None else None,
            "forced_rebalance_rate": value_at(
                history,
                "tsa_forced_rebalance_rate",
                best_index,
            ) if best_index is not None else None,
            "mean_margin": value_at(
                history,
                "tsa_mean_assignment_margin",
                best_index,
            ) if best_index is not None else None,
            "max_group_drift": max(drift) if drift else None,
            "mean_block_distances": json.dumps(
                (
                    value_at(
                        history,
                        "tsa_mean_block_distances",
                        best_index,
                        {},
                    )
                    if best_index is not None
                    else {}
                ),
                separators=(",", ":"),
                sort_keys=True,
            ),
            "block_nearest_agreement": json.dumps(
                (
                    value_at(
                        history,
                        "tsa_block_nearest_agreement",
                        best_index,
                        {},
                    )
                    if best_index is not None
                    else {}
                ),
                separators=(",", ":"),
                sort_keys=True,
            ),
            "block_task_variance": json.dumps(
                (
                    value_at(
                        history,
                        "tsa_block_task_variance",
                        best_index,
                        {},
                    )
                    if best_index is not None
                    else {}
                ),
                separators=(",", ":"),
                sort_keys=True,
            ),
            "routing_schedule": config.get("tsa_routing_schedule"),
            "switch_threshold": config.get("tsa_switch_threshold"),
            "min_group_fraction": config.get("tsa_min_group_fraction"),
            "max_group_fraction": config.get("tsa_max_group_fraction"),
            "json_path": path,
        })
    return sorted(
        rows,
        key=lambda row: (
            str(row["variant"]),
            int(row["support_size"]),
            int(row["seed"]),
        ),
    )


def mean_or_none(rows, key):
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and row.get(key) != ""
    ]
    return statistics.mean(values) if values else None


def format_metric(value):
    return "NA" if value is None else f"{value:.4f}"


def build_variant_rows(run_rows):
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["variant"], row["support_size"])].append(row)

    summaries = []
    for (variant, support_size), rows in grouped.items():
        val_values = [float(row["best_val_auroc"]) for row in rows]
        summaries.append({
            "variant": variant,
            "support_size": support_size,
            "seeds": len(rows),
            "mean_best_val_auroc": statistics.mean(val_values),
            "std_best_val_auroc": (
                statistics.stdev(val_values) if len(val_values) > 1 else 0.0
            ),
            "mean_val_auprc_at_best": mean_or_none(rows, "val_auprc_at_best"),
            "mean_epoch0_val_auroc": mean_or_none(rows, "epoch0_val_auroc"),
            "mean_epoch0_val_auprc": mean_or_none(rows, "epoch0_val_auprc"),
            "mean_epoch0_test_auroc": mean_or_none(rows, "epoch0_test_auroc"),
            "mean_epoch0_test_auprc": mean_or_none(rows, "epoch0_test_auprc"),
            "mean_epoch1_val_auroc": mean_or_none(rows, "epoch1_val_auroc"),
            "mean_epoch1_val_auprc": mean_or_none(rows, "epoch1_val_auprc"),
            "mean_epoch1_test_auroc": mean_or_none(rows, "epoch1_test_auroc"),
            "mean_epoch1_test_auprc": mean_or_none(rows, "epoch1_test_auprc"),
            "mean_test_auroc": mean_or_none(rows, "test_auroc"),
            "mean_test_auprc": mean_or_none(rows, "test_auprc"),
            "mean_switch_rate": mean_or_none(rows, "switch_rate"),
            "mean_hysteresis_retention_rate": mean_or_none(
                rows,
                "hysteresis_retention_rate",
            ),
            "mean_forced_rebalance_rate": mean_or_none(
                rows,
                "forced_rebalance_rate",
            ),
            "mean_margin": mean_or_none(rows, "mean_margin"),
            "mean_max_group_drift": mean_or_none(rows, "max_group_drift"),
        })
    return sorted(
        summaries,
        key=lambda row: (
            int(row["support_size"]),
            -float(row["mean_best_val_auroc"]),
        ),
    )


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    runs = latest_run_files(args.root)
    if not runs:
        raise FileNotFoundError(f"No TSA result JSON files found under {args.root}")

    run_rows = build_run_rows(runs)
    variant_rows = build_variant_rows(run_rows)
    run_path = os.path.join(args.root, "routing_run_summary.csv")
    variant_path = os.path.join(args.root, "routing_variant_summary.csv")
    write_csv(run_path, run_rows)
    write_csv(variant_path, variant_rows)

    for row in variant_rows:
        print(
            row["variant"],
            f"Epoch0={format_metric(row['mean_epoch0_val_auroc'])}",
            f"Epoch1={format_metric(row['mean_epoch1_val_auroc'])}",
            f"Val AUROC={row['mean_best_val_auroc']:.4f}",
            f"SD={row['std_best_val_auroc']:.4f}",
            f"Val AUPRC={row['mean_val_auprc_at_best']:.4f}",
            f"Switch={row['mean_switch_rate']}",
            f"Rebalance={row['mean_forced_rebalance_rate']}",
        )
    print(f"Run table: {run_path}")
    print(f"Variant table: {variant_path}")


if __name__ == "__main__":
    main()
