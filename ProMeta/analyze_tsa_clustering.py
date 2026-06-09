#!/usr/bin/env python
"""Rebuild TSA task vectors and quantify clustering quality."""

import argparse
import glob
import json
import math
import os
import pickle as pkl
import random
import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
)
from tqdm import tqdm

from config import Config
from dataset import MetaDataset, generate_pathway_mask
from model import FocalLoss, ProphetBioGateModel


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute support-derived TSA task vectors and evaluate initial, "
            "balanced-refit, and trained-group cluster assignments."
        )
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=[],
        help="Explicit TSA checkpoint paths.",
    )
    parser.add_argument(
        "--checkpoint_glob",
        action="append",
        default=[],
        help="Glob pattern for TSA checkpoints; may be repeated.",
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--proteomics_csv", required=True)
    parser.add_argument("--cpdb_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--support_size", type=int, default=None)
    parser.add_argument("--max_support_size", type=int, default=None)
    parser.add_argument(
        "--selector_source",
        choices=["checkpoint", "frozen_warmup", "live_model"],
        default="checkpoint",
        help="Selector used to rebuild task vectors.",
    )
    parser.add_argument("--null_permutations", type=int, default=100)
    parser.add_argument("--silhouette_sample_size", type=int, default=300)
    parser.add_argument("--kmeans_restarts", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=2026)
    return parser.parse_args()


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoints(explicit_paths, patterns):
    paths = [os.path.abspath(path) for path in explicit_paths]
    for pattern in patterns:
        paths.extend(os.path.abspath(path) for path in glob.glob(pattern))
    unique = sorted(set(paths))
    missing = [path for path in unique if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Checkpoint files not found: {missing}")
    if not unique:
        raise ValueError("No checkpoints found. Use --checkpoints or --checkpoint_glob.")
    return unique


def infer_seed(path, checkpoint):
    config = checkpoint.get("tsa_config") or {}
    if config.get("random_seed") is not None:
        return int(config["random_seed"])
    match = re.search(r"seed(\d+)", os.path.basename(path))
    if match:
        return int(match.group(1))
    raise ValueError(
        f"Could not infer seed from checkpoint metadata or filename: {path}"
    )


def checkpoint_label(path):
    checkpoint_path = Path(path)
    if len(checkpoint_path.parents) >= 3:
        return checkpoint_path.parents[2].name
    return checkpoint_path.stem


def config_from_checkpoint(checkpoint, args):
    saved = checkpoint.get("tsa_config") or {}
    centroids = checkpoint.get("tsa_centroids")
    inferred_groups = int(centroids.shape[0]) if torch.is_tensor(centroids) else 1
    selector_source = saved.get("tsa_selector_source", "frozen_warmup")
    if args.selector_source != "checkpoint":
        selector_source = args.selector_source
    param_keys = saved.get(
        "tsa_param_keys",
        ["classifier", "tokenizer.gate_logits"],
    )
    if not isinstance(param_keys, str):
        param_keys = ",".join(param_keys)

    values = SimpleNamespace(
        batch_size=1,
        max_support_size=(
            args.max_support_size
            if args.max_support_size is not None
            else int(saved.get("max_support_size", 32))
        ),
        inner_lr=0.005,
        outer_lr=1e-4,
        epochs=1,
        patience=0,
        dropout=0.5,
        experiment_name="TSA clustering analysis",
        l1_lambda=1e-3,
        tsa_enable=True,
        num_task_groups=int(saved.get("num_task_groups", inferred_groups)),
        tsa_param_keys=param_keys,
        tsa_selector_steps=int(saved.get("tsa_selector_steps", 10)),
        tsa_assignment_metric=saved.get("tsa_assignment_metric", "l2"),
        tsa_warmup_checkpoint="",
        tsa_selector_source=selector_source,
        tsa_assignment_source=saved.get("tsa_assignment_source", "current_group"),
        tsa_distance_mode=saved.get("tsa_distance_mode", "block_mean_l2"),
        tsa_gate_distance_weight=float(
            saved.get("tsa_gate_distance_weight", 1.0)
        ),
        tsa_selector_l1_lambda=float(
            saved.get("tsa_selector_l1_lambda", 1e-3)
        ),
        tsa_routing_schedule=saved.get("tsa_routing_schedule", "epoch_snapshot"),
        tsa_switch_threshold=float(saved.get("tsa_switch_threshold", 0.0)),
        tsa_min_group_fraction=float(saved.get("tsa_min_group_fraction", 0.0)),
        tsa_max_group_fraction=float(saved.get("tsa_max_group_fraction", 1.0)),
    )
    config = Config(values)
    support_size = (
        args.support_size
        if args.support_size is not None
        else int(saved.get("support_size", 4))
    )
    return config, support_size, values.max_support_size


def load_shared_data(args):
    print(f"[Cluster] Loading proteomics matrix: {args.proteomics_csv}")
    data = pd.read_csv(args.proteomics_csv)
    data["EID"] = data["EID"].apply(
        lambda value: str(value).strip().replace(".0", "")
    )
    eid_to_idx = {eid: idx for idx, eid in enumerate(data["EID"].values)}
    protein_frame = data.drop(columns=["EID"])
    protein_names = protein_frame.columns.tolist()
    proteins = np.nan_to_num(protein_frame.values.astype(np.float32))
    del data, protein_frame

    def load_pkl(filename):
        path = os.path.join(args.data_dir, filename)
        with open(path, "rb") as handle:
            return pkl.load(handle)

    train_cases = load_pkl("term2pre_cases_train.pkl")
    train_controls = load_pkl("term2pre_controls_train.pkl")
    pathway_mask, unknown_indices = generate_pathway_mask(
        protein_names,
        args.cpdb_path,
    )
    return {
        "proteins": proteins,
        "eid_to_idx": eid_to_idx,
        "train_cases": train_cases,
        "train_controls": train_controls,
        "pathway_mask": pathway_mask,
        "unknown_indices": unknown_indices,
    }


def rebuild_task_vectors(
    checkpoint_path,
    checkpoint,
    shared,
    args,
    device,
):
    from main import (
        estimate_task_vector,
        flatten_current_group_params,
        load_model_checkpoint,
    )

    seed = infer_seed(checkpoint_path, checkpoint)
    config, support_size, max_support_size = config_from_checkpoint(
        checkpoint,
        args,
    )
    set_seed(seed)
    dataset = MetaDataset(
        shared["proteins"],
        shared["train_cases"],
        shared["train_controls"],
        shared["eid_to_idx"],
        support_size=support_size,
        max_support_size=max_support_size,
        query_size=config.query_size,
        mode="cluster-analysis",
        random_seed=seed,
    )
    model = ProphetBioGateModel(
        shared["proteins"].shape[1],
        config,
        shared["pathway_mask"].to(device),
        shared["unknown_indices"],
    ).to(device)
    load_model_checkpoint(
        model,
        checkpoint_path,
        device,
        strict=True,
        load_metadata=True,
    )
    model.eval()
    criterion = FocalLoss(alpha=0.5, gamma=config.focal_gamma)
    terms = []
    vectors = []
    for index in tqdm(
        range(len(dataset)),
        desc=f"[Cluster] seed={seed} task vectors",
    ):
        _, _, support_x, support_y, term = dataset[index]
        if len(torch.unique(support_y)) < 2:
            continue
        with torch.enable_grad():
            vector = estimate_task_vector(
                model,
                support_x,
                support_y,
                config,
                criterion,
                device,
            )
        terms.append(str(term))
        vectors.append(vector.detach().cpu().numpy().astype(np.float32))

    if not vectors:
        raise ValueError(f"No valid task vectors rebuilt for {checkpoint_path}")
    raw_vectors = np.stack(vectors)
    vector_mean = model.tsa_vector_mean.detach().cpu().numpy()
    vector_std = model.tsa_vector_std.detach().cpu().numpy()
    normalized_vectors = (raw_vectors - vector_mean) / vector_std
    current_vectors = (
        flatten_current_group_params(model, model.tsa_param_slices)
        .detach()
        .cpu()
        .numpy()
    )
    current_prototypes = (current_vectors - vector_mean) / vector_std
    return {
        "seed": seed,
        "config": config,
        "support_size": support_size,
        "max_support_size": max_support_size,
        "terms": terms,
        "raw_vectors": raw_vectors,
        "normalized_vectors": normalized_vectors,
        "current_prototypes": current_prototypes,
        "param_slices": model.tsa_param_slices,
        "cluster_terms": model.tsa_cluster_terms,
        "checkpoint": checkpoint_path,
        "label": checkpoint_label(checkpoint_path),
    }


def block_indices(param_slices):
    blocks = {}
    for item in param_slices:
        block = item.get("block", item["name"])
        blocks.setdefault(block, []).extend(range(item["start"], item["end"]))
    return {
        block: np.asarray(indices, dtype=np.int64)
        for block, indices in blocks.items()
    }


def block_weight(block, gate_weight):
    return gate_weight if "gate_logits" in block else 1.0


def build_representations(normalized_vectors, param_slices, gate_weight):
    blocks = block_indices(param_slices)
    balanced = np.zeros_like(normalized_vectors)
    representations = {
        "global": normalized_vectors,
    }
    for block, indices in blocks.items():
        weight = block_weight(block, gate_weight)
        balanced[:, indices] = normalized_vectors[:, indices] * math.sqrt(
            weight / len(indices)
        )
        representations[f"block:{block}"] = normalized_vectors[:, indices]
    representations["balanced"] = balanced
    return representations, blocks


def prototype_distance_matrix(
    vectors,
    prototypes,
    param_slices,
    distance_mode,
    gate_weight,
):
    squared = (vectors[:, None, :] - prototypes[None, :, :]) ** 2
    if distance_mode == "global_l2":
        return squared.sum(axis=2), {}

    block_distances = {}
    total = np.zeros(squared.shape[:2], dtype=np.float64)
    for block, indices in block_indices(param_slices).items():
        values = squared[:, :, indices].mean(axis=2)
        block_distances[block] = values
        total += block_weight(block, gate_weight) * values
    return total, block_distances


def labels_from_cluster_terms(terms, cluster_terms):
    term_to_group = {}
    for group, group_terms in (cluster_terms or {}).items():
        for term in group_terms:
            term_to_group[str(term)] = int(group)
    missing = [term for term in terms if term not in term_to_group]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{len(missing)} rebuilt tasks are missing from tsa_cluster_terms: {preview}"
        )
    return np.asarray([term_to_group[term] for term in terms], dtype=np.int64)


def nearest_assignments(distance_matrix):
    order = np.argsort(distance_matrix, axis=1)
    labels = order[:, 0]
    nearest = distance_matrix[np.arange(len(labels)), labels]
    if distance_matrix.shape[1] > 1:
        second = distance_matrix[np.arange(len(labels)), order[:, 1]]
        margins = second - nearest
    else:
        margins = np.full(len(labels), np.nan)
    return labels.astype(np.int64), nearest, margins


def normalized_entropy(labels):
    counts = np.bincount(labels)
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(len(counts)))


def between_variance_fraction(values, labels):
    center = values.mean(axis=0)
    total = float(((values - center) ** 2).sum())
    if total <= 0:
        return 0.0
    within = 0.0
    for group in np.unique(labels):
        group_values = values[labels == group]
        group_center = group_values.mean(axis=0)
        within += float(((group_values - group_center) ** 2).sum())
    return float(max(0.0, 1.0 - within / total))


def choose_sample_indices(label_sets, max_size, seed):
    num_tasks = len(next(iter(label_sets.values())))
    if max_size <= 0 or max_size >= num_tasks:
        return np.arange(num_tasks)
    rng = np.random.RandomState(seed)
    selected = set()
    for labels in label_sets.values():
        for group in np.unique(labels):
            candidates = np.where(labels == group)[0]
            selected.add(int(rng.choice(candidates)))
    remaining = np.asarray(
        [idx for idx in range(num_tasks) if idx not in selected],
        dtype=np.int64,
    )
    needed = max_size - len(selected)
    if needed > 0:
        selected.update(rng.choice(remaining, size=needed, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def safe_silhouette(distance_matrix, labels):
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return None
    return float(silhouette_score(distance_matrix, labels, metric="precomputed"))


def permutation_silhouette(
    distance_matrix,
    labels,
    observed,
    permutations,
    seed,
):
    if observed is None or permutations <= 0:
        return None, None, None, None
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(permutations):
        shuffled = rng.permutation(labels)
        value = safe_silhouette(distance_matrix, shuffled)
        if value is not None:
            values.append(value)
    if not values:
        return None, None, None, None
    values = np.asarray(values)
    p_value = (1 + int(np.sum(values >= observed))) / (len(values) + 1)
    return (
        float(values.mean()),
        float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        float(np.quantile(values, 0.95)),
        float(p_value),
    )


def clustering_metric_rows(
    assignments,
    representations,
    sample_indices,
    permutations,
    seed,
    expected_cluster_count,
):
    rows = []
    distance_cache = {
        name: pairwise_distances(values[sample_indices], metric="euclidean")
        for name, values in representations.items()
    }
    for assignment_name, labels in assignments.items():
        counts = np.bincount(labels, minlength=expected_cluster_count)
        for representation_name, values in representations.items():
            sampled_labels = labels[sample_indices]
            observed = safe_silhouette(
                distance_cache[representation_name],
                sampled_labels,
            )
            null_mean, null_std, null_p95, null_p = permutation_silhouette(
                distance_cache[representation_name],
                sampled_labels,
                observed,
                permutations,
                seed + len(rows) * 101,
            )
            unique = np.unique(labels)
            if 1 < len(unique) < len(labels):
                ch_score = float(calinski_harabasz_score(values, labels))
                db_score = float(davies_bouldin_score(values, labels))
            else:
                ch_score = None
                db_score = None
            rows.append({
                "assignment": assignment_name,
                "representation": representation_name,
                "task_count": int(len(labels)),
                "cluster_count": int(len(unique)),
                "expected_cluster_count": int(expected_cluster_count),
                "cluster_sizes": json.dumps(counts.astype(int).tolist()),
                "min_cluster_fraction": float(counts.min() / len(labels)),
                "max_cluster_fraction": float(counts.max() / len(labels)),
                "normalized_size_entropy": normalized_entropy(labels),
                "silhouette": observed,
                "silhouette_sample_size": int(len(sample_indices)),
                "silhouette_null_mean": null_mean,
                "silhouette_null_std": null_std,
                "silhouette_null_p95": null_p95,
                "silhouette_null_p_value": null_p,
                "calinski_harabasz": ch_score,
                "davies_bouldin": db_score,
                "between_variance_fraction": between_variance_fraction(
                    values,
                    labels,
                ),
            })
    return rows


def save_pca_plot(output_path, balanced_vectors, assignments, seed):
    pca = PCA(n_components=2, random_state=seed)
    coordinates = pca.fit_transform(balanced_vectors)
    names = list(assignments)
    fig, axes = plt.subplots(
        1,
        len(names),
        figsize=(5 * len(names), 4.5),
        squeeze=False,
    )
    for axis, name in zip(axes[0], names):
        labels = assignments[name]
        scatter = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=labels,
            cmap="tab10",
            s=16,
            alpha=0.75,
        )
        axis.set_title(name)
        axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        axis.grid(alpha=0.2)
        axis.legend(
            *scatter.legend_elements(),
            title="Group",
            loc="best",
            fontsize=7,
        )
    fig.suptitle("TSA task vectors in block-balanced parameter space")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return pca.explained_variance_ratio_.tolist()


def save_cluster_size_plot(output_path, assignments, group_count):
    names = list(assignments)
    positions = np.arange(group_count)
    width = 0.8 / len(names)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for index, name in enumerate(names):
        counts = np.bincount(assignments[name], minlength=group_count)
        axis.bar(
            positions + (index - (len(names) - 1) / 2) * width,
            counts,
            width=width,
            label=name,
        )
    axis.set_xticks(positions)
    axis.set_xlabel("Group")
    axis.set_ylabel("Training disease tasks")
    axis.set_title("Cluster-size comparison")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def agreement_rows(assignments):
    rows = []
    names = list(assignments)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            rows.append({
                "left_assignment": left,
                "right_assignment": right,
                "ari": float(
                    adjusted_rand_score(assignments[left], assignments[right])
                ),
                "nmi": float(
                    normalized_mutual_info_score(
                        assignments[left],
                        assignments[right],
                    )
                ),
            })
    return rows


def analyze_checkpoint(rebuilt, args, output_root):
    seed = rebuilt["seed"]
    config = rebuilt["config"]
    terms = rebuilt["terms"]
    vectors = rebuilt["normalized_vectors"]
    representations, blocks = build_representations(
        vectors,
        rebuilt["param_slices"],
        config.tsa_gate_distance_weight,
    )
    initial_labels = labels_from_cluster_terms(
        terms,
        rebuilt["cluster_terms"],
    )
    global_refit = KMeans(
        n_clusters=config.num_task_groups,
        random_state=seed,
        n_init=args.kmeans_restarts,
    ).fit_predict(representations["global"])
    balanced_refit = KMeans(
        n_clusters=config.num_task_groups,
        random_state=seed,
        n_init=args.kmeans_restarts,
    ).fit_predict(representations["balanced"])
    current_distances, block_distances = prototype_distance_matrix(
        vectors,
        rebuilt["current_prototypes"],
        rebuilt["param_slices"],
        config.tsa_distance_mode,
        config.tsa_gate_distance_weight,
    )
    current_labels, current_nearest, current_margins = nearest_assignments(
        current_distances
    )
    assignments = {
        "stored_initial": initial_labels,
        "global_refit": global_refit,
        "balanced_refit": balanced_refit,
        "current_group": current_labels,
    }
    sample_indices = choose_sample_indices(
        assignments,
        args.silhouette_sample_size,
        args.random_seed + seed,
    )
    metric_rows = clustering_metric_rows(
        assignments,
        representations,
        sample_indices,
        args.null_permutations,
        args.random_seed + seed,
        config.num_task_groups,
    )
    run_dir = os.path.join(
        output_root,
        f"{rebuilt['label']}_seed{seed}",
    )
    os.makedirs(run_dir, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(
        os.path.join(run_dir, "cluster_metrics.csv"),
        index=False,
    )
    pd.DataFrame(agreement_rows(assignments)).to_csv(
        os.path.join(run_dir, "assignment_agreement.csv"),
        index=False,
    )

    task_rows = []
    block_nearest = {}
    block_nearest_distance = {}
    for block, distances in block_distances.items():
        labels, nearest, _ = nearest_assignments(distances)
        block_nearest[block] = labels
        block_nearest_distance[block] = nearest
    for index, term in enumerate(terms):
        row = {
            "term": term,
            "stored_initial_group": int(initial_labels[index]),
            "global_refit_group": int(global_refit[index]),
            "balanced_refit_group": int(balanced_refit[index]),
            "current_group": int(current_labels[index]),
            "current_distance": float(current_nearest[index]),
            "current_margin": float(current_margins[index]),
        }
        for block in blocks:
            if block in block_nearest:
                safe_name = block.replace(".", "_")
                row[f"{safe_name}_nearest_group"] = int(
                    block_nearest[block][index]
                )
                row[f"{safe_name}_nearest_distance"] = float(
                    block_nearest_distance[block][index]
                )
        task_rows.append(row)
    pd.DataFrame(task_rows).to_csv(
        os.path.join(run_dir, "task_assignments.csv"),
        index=False,
    )
    np.savez_compressed(
        os.path.join(run_dir, "task_vectors.npz"),
        terms=np.asarray(terms),
        raw_vectors=rebuilt["raw_vectors"],
        normalized_vectors=vectors,
        balanced_vectors=representations["balanced"],
    )
    pca_variance = save_pca_plot(
        os.path.join(run_dir, "task_vector_pca.png"),
        representations["balanced"],
        assignments,
        seed,
    )
    save_cluster_size_plot(
        os.path.join(run_dir, "cluster_sizes.png"),
        assignments,
        config.num_task_groups,
    )
    primary_representation = (
        "balanced"
        if config.tsa_distance_mode == "block_mean_l2"
        else "global"
    )
    primary_metrics = {
        row["assignment"]: row
        for row in metric_rows
        if row["representation"] == primary_representation
    }
    summary = {
        "checkpoint": rebuilt["checkpoint"],
        "label": rebuilt["label"],
        "seed": seed,
        "support_size": rebuilt["support_size"],
        "max_support_size": rebuilt["max_support_size"],
        "task_count": len(terms),
        "selector_source": config.tsa_selector_source,
        "selector_steps": config.tsa_selector_steps,
        "distance_mode": config.tsa_distance_mode,
        "gate_distance_weight": config.tsa_gate_distance_weight,
        "param_blocks": {
            block: int(len(indices)) for block, indices in blocks.items()
        },
        "primary_representation": primary_representation,
        "primary_metrics": primary_metrics,
        "assignment_agreement": agreement_rows(assignments),
        "pca_explained_variance_ratio": pca_variance,
    }
    with open(
        os.path.join(run_dir, "clustering_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)
    return {
        "seed": seed,
        "label": rebuilt["label"],
        "terms": terms,
        "assignments": assignments,
        "metric_rows": metric_rows,
        "summary": summary,
    }


def cross_seed_rows(results):
    rows = []
    assignment_names = [
        "stored_initial",
        "balanced_refit",
        "current_group",
    ]
    for left_index, left in enumerate(results):
        for right in results[left_index + 1:]:
            left_index_by_term = {
                term: index for index, term in enumerate(left["terms"])
            }
            right_index_by_term = {
                term: index for index, term in enumerate(right["terms"])
            }
            common_terms = sorted(
                set(left_index_by_term) & set(right_index_by_term)
            )
            for assignment in assignment_names:
                left_labels = np.asarray([
                    left["assignments"][assignment][left_index_by_term[term]]
                    for term in common_terms
                ])
                right_labels = np.asarray([
                    right["assignments"][assignment][right_index_by_term[term]]
                    for term in common_terms
                ])
                rows.append({
                    "left_label": left["label"],
                    "left_seed": left["seed"],
                    "right_label": right["label"],
                    "right_seed": right["seed"],
                    "assignment": assignment,
                    "common_tasks": len(common_terms),
                    "ari": float(
                        adjusted_rand_score(left_labels, right_labels)
                    ),
                    "nmi": float(
                        normalized_mutual_info_score(
                            left_labels,
                            right_labels,
                        )
                    ),
                })
    return rows


def write_readout(output_dir, results, stability_rows):
    def format_metric(value):
        return "NA" if value is None else f"{value:.4f}"

    lines = [
        "# TSA clustering diagnostic readout",
        "",
        "This report evaluates parameter-space geometry, not biological validity "
        "or predictive superiority.",
        "",
    ]
    for result in results:
        summary = result["summary"]
        lines.append(
            f"## {summary['label']} seed {summary['seed']}"
        )
        for assignment, metrics in summary["primary_metrics"].items():
            silhouette = metrics["silhouette"]
            p_value = metrics["silhouette_null_p_value"]
            lines.append(
                f"- {assignment}: silhouette={format_metric(silhouette)} "
                f"(permutation p={format_metric(p_value)}), "
                f"between-variance={metrics['between_variance_fraction']:.4f}, "
                f"sizes={metrics['cluster_sizes']}."
            )
        lines.append("")
    if stability_rows:
        frame = pd.DataFrame(stability_rows)
        lines.append("## Cross-seed stability")
        for assignment, group in frame.groupby("assignment"):
            lines.append(
                f"- {assignment}: mean ARI={group['ari'].mean():.4f}, "
                f"mean NMI={group['nmi'].mean():.4f}."
            )
        lines.append("")
    lines.extend([
        "## Interpretation rules",
        "- A silhouette above its shuffled-label null indicates non-random "
        "geometric separation in the evaluated representation.",
        "- ARI/NMI near zero across seeds indicates unstable disease grouping, "
        "even when within-seed separation looks good.",
        "- A very small minimum cluster fraction or very large maximum fraction "
        "indicates collapse or severe imbalance.",
        "- A low ARI between stored_initial and balanced_refit means the original "
        "global-L2 K-means is misaligned with block-mean routing geometry.",
        "- Biological coherence requires external disease-family annotations; "
        "these files alone cannot establish it.",
    ])
    with open(
        os.path.join(output_dir, "CLUSTERING_READOUT.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    checkpoints = resolve_checkpoints(
        args.checkpoints,
        args.checkpoint_glob,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    print(f"[Cluster] Device: {device}")
    print(f"[Cluster] Checkpoints: {len(checkpoints)}")
    shared = load_shared_data(args)
    results = []
    all_metric_rows = []
    for checkpoint_path in checkpoints:
        checkpoint = torch_load(checkpoint_path, "cpu")
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            raise ValueError(
                f"Expected a TSA checkpoint with metadata: {checkpoint_path}"
            )
        rebuilt = rebuild_task_vectors(
            checkpoint_path,
            checkpoint,
            shared,
            args,
            device,
        )
        result = analyze_checkpoint(rebuilt, args, args.output_dir)
        results.append(result)
        for row in result["metric_rows"]:
            all_metric_rows.append({
                "label": result["label"],
                "seed": result["seed"],
                **row,
            })
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(all_metric_rows).to_csv(
        os.path.join(args.output_dir, "cluster_metric_summary.csv"),
        index=False,
    )
    stability = cross_seed_rows(results)
    pd.DataFrame(stability).to_csv(
        os.path.join(args.output_dir, "cross_seed_stability.csv"),
        index=False,
    )
    write_readout(args.output_dir, results, stability)
    print(f"[Cluster] Wrote analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()
