import os
import math
import pickle as pkl
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import parse_args
from utils import set_seed, compute_task_metrics, save_results
from dataset import generate_pathway_mask, MetaDataset
from model import ProphetBioGateModel, FocalLoss


def is_adaptive_param_name(name, config):
    return any(name == key or name.startswith(f"{key}.") for key in config.tsa_param_keys)


def get_base_params(model):
    return {
        name: param
        for name, param in model.named_parameters()
        if not name.startswith("alphas.") and not name.startswith("tsa_group_params.")
    }


def get_adaptive_param_names(model, config):
    if getattr(config, "tsa_enable", False) and model.tsa_param_names:
        return list(model.tsa_param_names)
    return [name for name in get_base_params(model) if is_adaptive_param_name(name, config)]


def get_alpha_dict(model, detach=False):
    if detach:
        return {name: param.detach().clone() for name, param in model.alphas.items()}
    return model.alphas


def make_fast_weights(model, config, group_idx=None, detach=False, source_params=None):
    adaptive_names = set(get_adaptive_param_names(model, config))
    fast_weights = {}
    base_params = source_params if source_params is not None else get_base_params(model)

    for name, param in base_params.items():
        source = param
        if source_params is None and group_idx is not None and name in adaptive_names:
            source = model.get_tsa_group_param(group_idx, name)

        if detach:
            tensor = source.detach().clone()
            if name in adaptive_names:
                tensor.requires_grad_(True)
            fast_weights[name] = tensor
        else:
            fast_weights[name] = source

    return fast_weights


def normalize_task_tensors(x, y):
    if x.dim() == 2:
        model_x = x.unsqueeze(0)
    else:
        model_x = x

    labels = y
    if labels.dim() > 1 and labels.shape[0] == 1:
        labels = labels.squeeze(0)

    return model_x, labels.float()


def adapt_fast_weights(model, support_x, support_y, fast_weights, alphas, grad_keys,
                       criterion, config, steps, create_graph, l1_lambda=None):
    model_x, labels = normalize_task_tensors(support_x, support_y)
    gate_l1_lambda = config.l1_lambda if l1_lambda is None else l1_lambda

    for _ in range(steps):
        preds, gate_vals = model.functional_forward(model_x, fast_weights)
        cls_loss = criterion(preds.squeeze(0), labels.unsqueeze(-1))
        l1_loss = gate_l1_lambda * torch.norm(gate_vals, p=1)
        grads = torch.autograd.grad(
            cls_loss + l1_loss,
            [fast_weights[name] for name in grad_keys],
            create_graph=create_graph,
            allow_unused=True,
        )

        for name, grad in zip(grad_keys, grads):
            if grad is None:
                continue
            alpha_key = name.replace(".", "_")
            fast_weights[name] = fast_weights[name] - alphas[alpha_key] * grad

    return fast_weights


def flatten_adaptive_params(params, param_names):
    return torch.cat([params[name].reshape(-1) for name in param_names], dim=0)


def get_param_block(name, config):
    for key in config.tsa_param_keys:
        if name == key or name.startswith(f"{key}."):
            return key
    return name


def get_param_slices(model, config):
    slices = []
    start = 0
    base_params = get_base_params(model)
    for name in get_adaptive_param_names(model, config):
        size = base_params[name].numel()
        slices.append({
            "name": name,
            "start": start,
            "end": start + size,
            "shape": tuple(base_params[name].shape),
            "block": get_param_block(name, config),
        })
        start += size
    return slices


def capture_selector_snapshot(model):
    model.tsa_selector_params = {
        name: param.detach().clone()
        for name, param in get_base_params(model).items()
    }
    model.tsa_selector_alphas = {
        name: param.detach().clone()
        for name, param in model.alphas.items()
    }


def estimate_task_vector(model, support_x, support_y, config, criterion, device):
    grad_keys = get_adaptive_param_names(model, config)
    if config.tsa_selector_source == "frozen_warmup":
        if model.tsa_selector_params is None or model.tsa_selector_alphas is None:
            raise ValueError(
                "Frozen TSA selector snapshot is missing. Rebuild TSA from a ProMeta "
                "warmup checkpoint or load a TSA v2 checkpoint."
            )
        source_params = model.tsa_selector_params
        alphas = model.tsa_selector_alphas
    else:
        source_params = None
        alphas = get_alpha_dict(model, detach=True)

    fast_weights = make_fast_weights(
        model,
        config,
        detach=True,
        source_params=source_params,
    )
    fast_weights = adapt_fast_weights(
        model,
        support_x.to(device),
        support_y.to(device),
        fast_weights,
        alphas,
        grad_keys,
        criterion,
        config,
        config.tsa_selector_steps,
        create_graph=False,
        l1_lambda=config.tsa_selector_l1_lambda,
    )
    return flatten_adaptive_params(fast_weights, grad_keys).detach()


def kmeans_numpy(x, num_clusters, seed, max_iter=100):
    if x.shape[0] < num_clusters:
        raise ValueError(
            f"TSA requires at least {num_clusters} valid training tasks, got {x.shape[0]}."
        )

    rng = np.random.RandomState(seed)
    centers = np.empty((num_clusters, x.shape[1]), dtype=np.float32)
    first_idx = rng.randint(x.shape[0])
    centers[0] = x[first_idx]
    closest_dist = ((x - centers[0]) ** 2).sum(axis=1)

    for cluster_idx in range(1, num_clusters):
        total = closest_dist.sum()
        if total <= 0:
            next_idx = rng.randint(x.shape[0])
        else:
            next_idx = rng.choice(x.shape[0], p=closest_dist / total)
        centers[cluster_idx] = x[next_idx]
        closest_dist = np.minimum(closest_dist, ((x - centers[cluster_idx]) ** 2).sum(axis=1))

    labels = np.full(x.shape[0], -1, dtype=np.int64)
    for _ in range(max_iter):
        distances = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)

        new_centers = centers.copy()
        for cluster_idx in range(num_clusters):
            mask = new_labels == cluster_idx
            if mask.any():
                new_centers[cluster_idx] = x[mask].mean(axis=0)
            else:
                farthest_idx = distances.min(axis=1).argmax()
                new_centers[cluster_idx] = x[farthest_idx]

        if np.array_equal(labels, new_labels):
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers

    return labels, centers


def assign_tsa_group_params_from_vectors(model, group_vectors, param_slices):
    with torch.no_grad():
        for group_idx, vector in enumerate(group_vectors):
            tensor_vector = torch.as_tensor(
                vector,
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device,
            )
            for item in param_slices:
                value = tensor_vector[item["start"]:item["end"]].view(item["shape"])
                model.get_tsa_group_param(group_idx, item["name"]).copy_(value)


def tensors_to_cpu(values):
    if values is None:
        return None
    return {name: value.detach().cpu() for name, value in values.items()}


def tensors_to_device(values, device):
    if values is None:
        return None
    return {name: value.detach().to(device) for name, value in values.items()}


def load_tsa_metadata(model, checkpoint, device):
    for key in (
        "tsa_centroids",
        "tsa_vector_mean",
        "tsa_vector_std",
        "tsa_initial_group_vectors",
    ):
        value = checkpoint.get(key)
        setattr(model, key, value.to(device) if torch.is_tensor(value) else value)
    model.tsa_cluster_counts = checkpoint.get("tsa_cluster_counts")
    model.tsa_cluster_terms = checkpoint.get("tsa_cluster_terms")
    model.tsa_param_slices = checkpoint.get("tsa_param_slices")
    if model.tsa_param_slices is not None:
        for item in model.tsa_param_slices:
            item.setdefault("block", get_param_block(item["name"], model.config))
    model.tsa_selector_params = tensors_to_device(
        checkpoint.get("tsa_selector_params"),
        device,
    )
    model.tsa_selector_alphas = tensors_to_device(
        checkpoint.get("tsa_selector_alphas"),
        device,
    )

    if (
        getattr(model.config, "tsa_selector_source", "live_model") == "frozen_warmup"
        and (model.tsa_selector_params is None or model.tsa_selector_alphas is None)
    ):
        raise ValueError(
            "This TSA checkpoint does not contain a frozen selector snapshot. "
            "Use --tsa_selector_source live_model for legacy checkpoints or rebuild "
            "the checkpoint from its ProMeta warmup model."
        )


def load_model_checkpoint(model, path, device, strict=True, load_metadata=True):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=strict)
        if load_metadata:
            load_tsa_metadata(model, checkpoint, device)
        return checkpoint

    model.load_state_dict(checkpoint, strict=strict)
    return checkpoint


def build_model_checkpoint(model, config, args):
    if not config.tsa_enable:
        return model.state_dict()

    return {
        "model_state": model.state_dict(),
        "tsa_centroids": model.tsa_centroids.detach().cpu() if model.tsa_centroids is not None else None,
        "tsa_vector_mean": model.tsa_vector_mean.detach().cpu() if model.tsa_vector_mean is not None else None,
        "tsa_vector_std": model.tsa_vector_std.detach().cpu() if model.tsa_vector_std is not None else None,
        "tsa_cluster_counts": model.tsa_cluster_counts,
        "tsa_cluster_terms": model.tsa_cluster_terms,
        "tsa_param_slices": model.tsa_param_slices,
        "tsa_selector_params": tensors_to_cpu(model.tsa_selector_params),
        "tsa_selector_alphas": tensors_to_cpu(model.tsa_selector_alphas),
        "tsa_initial_group_vectors": (
            model.tsa_initial_group_vectors.detach().cpu()
            if model.tsa_initial_group_vectors is not None
            else None
        ),
        "tsa_config": {
            "num_task_groups": config.num_task_groups,
            "tsa_param_keys": config.tsa_param_keys,
            "tsa_selector_steps": config.tsa_selector_steps,
            "tsa_assignment_metric": config.tsa_assignment_metric,
            "tsa_selector_source": config.tsa_selector_source,
            "tsa_assignment_source": config.tsa_assignment_source,
            "tsa_distance_mode": config.tsa_distance_mode,
            "tsa_gate_distance_weight": config.tsa_gate_distance_weight,
            "tsa_selector_l1_lambda": config.tsa_selector_l1_lambda,
            "tsa_routing_schedule": config.tsa_routing_schedule,
            "tsa_switch_threshold": config.tsa_switch_threshold,
            "tsa_min_group_fraction": config.tsa_min_group_fraction,
            "tsa_max_group_fraction": config.tsa_max_group_fraction,
            "support_size": args.support_size,
            "max_support_size": args.max_support_size,
        },
    }


def prepare_tsa_model(model, train_loader, config, args, device):
    if not config.tsa_enable:
        return
    if not config.tsa_warmup_checkpoint:
        raise ValueError("--tsa_warmup_checkpoint is required when --tsa_enable is used.")
    if not os.path.exists(config.tsa_warmup_checkpoint):
        raise FileNotFoundError(f"TSA warmup checkpoint not found: {config.tsa_warmup_checkpoint}")

    print(f"[TSA] Loading warmup checkpoint: {config.tsa_warmup_checkpoint}")
    load_model_checkpoint(model, config.tsa_warmup_checkpoint, device, strict=False, load_metadata=False)
    capture_selector_snapshot(model)
    model.reset_tsa_group_params_to_base()

    criterion = FocalLoss(alpha=0.5, gamma=config.focal_gamma)
    vectors = []
    terms = []
    was_training = model.training
    model.eval()

    for batch in tqdm(train_loader, desc="[TSA] Estimating task parameters", leave=False):
        s_in, s_lb = batch[2].to(device), batch[3].to(device)
        term_batch = batch[4]

        for task_idx in range(s_in.shape[0]):
            labels = s_lb[task_idx]
            if len(torch.unique(labels)) < 2:
                continue
            with torch.enable_grad():
                vector = estimate_task_vector(model, s_in[task_idx], labels, config, criterion, device)
            vectors.append(vector.cpu().numpy().astype(np.float32))
            terms.append(str(term_batch[task_idx]))

    if was_training:
        model.train()

    if not vectors:
        raise ValueError("No valid training tasks were available for TSA clustering.")

    raw_vectors = np.stack(vectors, axis=0)
    vector_mean = raw_vectors.mean(axis=0)
    vector_std = raw_vectors.std(axis=0)
    vector_std[vector_std < 1e-6] = 1.0
    normalized_vectors = (raw_vectors - vector_mean) / vector_std

    labels, centroids = kmeans_numpy(
        normalized_vectors,
        config.num_task_groups,
        args.random_seed,
    )

    param_slices = get_param_slices(model, config)
    group_vectors = []
    cluster_terms = {}
    cluster_counts = []
    for group_idx in range(config.num_task_groups):
        mask = labels == group_idx
        cluster_counts.append(int(mask.sum()))
        cluster_terms[str(group_idx)] = [terms[i] for i in np.where(mask)[0].tolist()]
        if mask.any():
            group_vectors.append(raw_vectors[mask].mean(axis=0))
        else:
            group_vectors.append(raw_vectors.mean(axis=0))

    assign_tsa_group_params_from_vectors(model, group_vectors, param_slices)
    model.tsa_centroids = torch.tensor(centroids, dtype=torch.float32, device=device)
    model.tsa_vector_mean = torch.tensor(vector_mean, dtype=torch.float32, device=device)
    model.tsa_vector_std = torch.tensor(vector_std, dtype=torch.float32, device=device)
    model.tsa_cluster_counts = cluster_counts
    model.tsa_cluster_terms = cluster_terms
    model.tsa_param_slices = param_slices
    model.tsa_initial_group_vectors = torch.tensor(
        np.stack(group_vectors, axis=0),
        dtype=torch.float32,
        device=device,
    )

    print(f"[TSA] Built {config.num_task_groups} task groups from {len(vectors)} tasks.")
    print(f"[TSA] Cluster sizes: {cluster_counts}")


def flatten_current_group_params(model, param_slices):
    vectors = []
    for group_idx in range(model.num_task_groups):
        vectors.append(torch.cat([
            model.get_tsa_group_param(group_idx, item["name"]).reshape(-1)
            for item in param_slices
        ]))
    return torch.stack(vectors, dim=0)


def compute_tsa_distance_components(normalized_vector, normalized_prototypes, param_slices):
    squared = (normalized_prototypes - normalized_vector.unsqueeze(0)) ** 2
    block_ranges = {}
    for item in param_slices:
        block = item.get("block", item["name"])
        block_ranges.setdefault(block, []).append((item["start"], item["end"]))

    components = {}
    for block, ranges in block_ranges.items():
        block_values = torch.cat(
            [squared[:, start:end] for start, end in ranges],
            dim=1,
        )
        components[block] = block_values.mean(dim=1)
    return squared, components


def compute_tsa_distances(normalized_vector, normalized_prototypes, param_slices,
                          distance_mode, gate_distance_weight):
    squared, components = compute_tsa_distance_components(
        normalized_vector,
        normalized_prototypes,
        param_slices,
    )
    if distance_mode == "global_l2":
        return squared.sum(dim=1)

    distances = torch.zeros(
        normalized_prototypes.shape[0],
        dtype=normalized_prototypes.dtype,
        device=normalized_prototypes.device,
    )
    for block, block_values in components.items():
        weight = gate_distance_weight if "gate_logits" in block else 1.0
        distances = distances + weight * block_values
    return distances


def get_tsa_assignment_prototypes(model, config):
    if config.tsa_assignment_source == "fixed_centroid":
        return model.tsa_centroids

    current_vectors = flatten_current_group_params(model, model.tsa_param_slices).detach()
    return (current_vectors - model.tsa_vector_mean.unsqueeze(0)) / model.tsa_vector_std.unsqueeze(0)


def select_tsa_group(model, support_x, support_y, config, criterion, device,
                     prototypes=None):
    if model.tsa_centroids is None or model.tsa_vector_mean is None or model.tsa_vector_std is None:
        raise ValueError("TSA metadata is missing. Run prepare_tsa_model or load a TSA checkpoint first.")

    was_training = model.training
    model.eval()
    with torch.enable_grad():
        vector = estimate_task_vector(model, support_x, support_y, config, criterion, device)
    if was_training:
        model.train()

    normalized = (vector.to(device) - model.tsa_vector_mean) / model.tsa_vector_std
    if prototypes is None:
        prototypes = get_tsa_assignment_prototypes(model, config)
    distances = compute_tsa_distances(
        normalized,
        prototypes,
        model.tsa_param_slices,
        config.tsa_distance_mode,
        config.tsa_gate_distance_weight,
    )
    _, block_components = compute_tsa_distance_components(
        normalized,
        prototypes,
        model.tsa_param_slices,
    )
    sorted_distances, sorted_indices = torch.sort(distances)
    margin = None
    if len(sorted_distances) > 1:
        margin = float((sorted_distances[1] - sorted_distances[0]).detach().cpu())
    group_idx = int(sorted_indices[0].item())
    return {
        "group": group_idx,
        "nearest_group": group_idx,
        "distance": float(sorted_distances[0].detach().cpu()),
        "margin": margin,
        "block_distances": {
            block: float(values[group_idx].detach().cpu())
            for block, values in block_components.items()
        },
        "block_nearest_groups": {
            block: int(torch.argmin(values).item())
            for block, values in block_components.items()
        },
    }


def apply_switch_hysteresis(distance_matrix, terms, previous_assignments,
                            switch_threshold):
    nearest = distance_matrix.argmin(axis=1).astype(np.int64)
    selected = nearest.copy()
    retained = np.zeros(len(terms), dtype=bool)
    relative_improvements = np.full(len(terms), np.nan, dtype=np.float64)

    if previous_assignments is None:
        return selected, nearest, retained, relative_improvements

    num_groups = distance_matrix.shape[1]
    for idx, term in enumerate(terms):
        previous_group = previous_assignments.get(term)
        if previous_group is None or not 0 <= previous_group < num_groups:
            continue
        best_group = int(nearest[idx])
        if best_group == previous_group:
            relative_improvements[idx] = 0.0
            continue

        previous_distance = float(distance_matrix[idx, previous_group])
        best_distance = float(distance_matrix[idx, best_group])
        denominator = max(abs(previous_distance), 1e-12)
        improvement = (previous_distance - best_distance) / denominator
        relative_improvements[idx] = improvement
        if improvement < switch_threshold:
            selected[idx] = previous_group
            retained[idx] = True

    return selected, nearest, retained, relative_improvements


def enforce_group_size_constraints(distance_matrix, assignments, min_count,
                                   max_count):
    assignments = np.asarray(assignments, dtype=np.int64).copy()
    num_tasks, num_groups = distance_matrix.shape
    forced = np.zeros(num_tasks, dtype=bool)
    counts = np.bincount(assignments, minlength=num_groups).astype(np.int64)

    if min_count * num_groups > num_tasks or max_count * num_groups < num_tasks:
        raise ValueError("Infeasible TSA group size constraints")

    while np.any(counts > max_count):
        best_move = None
        underfilled = set(np.where(counts < min_count)[0].tolist())
        for source in np.where(counts > max_count)[0]:
            source_tasks = np.where(assignments == source)[0]
            for task_idx in source_tasks:
                for target in np.where(counts < max_count)[0]:
                    if target == source:
                        continue
                    penalty = (
                        float(distance_matrix[task_idx, target])
                        - float(distance_matrix[task_idx, source])
                    )
                    priority = 0 if target in underfilled else 1
                    candidate = (priority, penalty, int(task_idx), int(target))
                    if best_move is None or candidate < best_move:
                        best_move = candidate
        if best_move is None:
            raise ValueError("Could not satisfy TSA maximum group size")
        _, _, task_idx, target = best_move
        source = int(assignments[task_idx])
        assignments[task_idx] = target
        counts[source] -= 1
        counts[target] += 1
        forced[task_idx] = True

    while np.any(counts < min_count):
        target = int(np.where(counts < min_count)[0][0])
        best_move = None
        for task_idx in range(num_tasks):
            source = int(assignments[task_idx])
            if source == target or counts[source] <= min_count:
                continue
            penalty = (
                float(distance_matrix[task_idx, target])
                - float(distance_matrix[task_idx, source])
            )
            candidate = (penalty, int(task_idx))
            if best_move is None or candidate < best_move:
                best_move = candidate
        if best_move is None:
            raise ValueError("Could not satisfy TSA minimum group size")
        _, task_idx = best_move
        source = int(assignments[task_idx])
        assignments[task_idx] = target
        counts[source] -= 1
        counts[target] += 1
        forced[task_idx] = True

    return assignments, forced, counts.tolist()


def get_group_count_bounds(num_tasks, num_groups, min_fraction, max_fraction):
    min_count = int(math.ceil(num_tasks * min_fraction))
    max_count = int(math.floor(num_tasks * max_fraction))
    min_count = min(min_count, num_tasks // num_groups)
    max_count = max(max_count, int(math.ceil(num_tasks / num_groups)))
    return min_count, max_count


def build_epoch_assignment_map(model, train_loader, config, device,
                               previous_assignments):
    routing_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=False,
    )
    criterion = FocalLoss(alpha=0.5, gamma=config.focal_gamma)
    prototypes = get_tsa_assignment_prototypes(model, config).detach().clone()
    terms = []
    normalized_vectors = []
    distance_rows = []
    block_rows = {}

    was_training = model.training
    model.eval()
    for batch in tqdm(routing_loader, desc="[TSA] Routing epoch", leave=False):
        support_x, support_y = batch[2].to(device), batch[3].to(device)
        for task_idx in range(support_x.shape[0]):
            labels = support_y[task_idx]
            if len(torch.unique(labels)) < 2:
                continue
            with torch.enable_grad():
                vector = estimate_task_vector(
                    model,
                    support_x[task_idx],
                    labels,
                    config,
                    criterion,
                    device,
                )
            normalized = (vector.to(device) - model.tsa_vector_mean) / model.tsa_vector_std
            distances = compute_tsa_distances(
                normalized,
                prototypes,
                model.tsa_param_slices,
                config.tsa_distance_mode,
                config.tsa_gate_distance_weight,
            )
            _, components = compute_tsa_distance_components(
                normalized,
                prototypes,
                model.tsa_param_slices,
            )
            terms.append(str(batch[4][task_idx]))
            normalized_vectors.append(normalized.detach())
            distance_rows.append(distances.detach())
            for block, values in components.items():
                block_rows.setdefault(block, []).append(values.detach())
    if was_training:
        model.train()

    if not terms:
        raise ValueError("No valid training tasks were available for TSA routing.")

    distance_matrix = torch.stack(distance_rows).cpu().numpy()
    selected, nearest, retained, improvements = apply_switch_hysteresis(
        distance_matrix,
        terms,
        previous_assignments,
        config.tsa_switch_threshold,
    )
    min_count, max_count = get_group_count_bounds(
        len(terms),
        config.num_task_groups,
        config.tsa_min_group_fraction,
        config.tsa_max_group_fraction,
    )
    selected, forced, counts = enforce_group_size_constraints(
        distance_matrix,
        selected,
        min_count,
        max_count,
    )

    block_matrices = {
        block: torch.stack(rows).cpu().numpy()
        for block, rows in block_rows.items()
    }
    normalized_matrix = torch.stack(normalized_vectors)
    block_task_variance = {}
    for block in block_matrices:
        ranges = [
            (item["start"], item["end"])
            for item in model.tsa_param_slices
            if item.get("block", item["name"]) == block
        ]
        block_values = torch.cat(
            [normalized_matrix[:, start:end] for start, end in ranges],
            dim=1,
        )
        block_task_variance[block] = float(
            block_values.var(dim=0, unbiased=False).mean().cpu()
        )

    details = []
    assignment_map = {}
    for idx, term in enumerate(terms):
        group_idx = int(selected[idx])
        sorted_distances = np.sort(distance_matrix[idx])
        margin = (
            float(sorted_distances[1] - sorted_distances[0])
            if len(sorted_distances) > 1
            else None
        )
        previous_group = (
            previous_assignments.get(term)
            if previous_assignments is not None
            else None
        )
        detail = {
            "disease_term": term,
            "group": group_idx,
            "nearest_group": int(nearest[idx]),
            "distance": float(distance_matrix[idx, group_idx]),
            "nearest_distance": float(distance_matrix[idx, nearest[idx]]),
            "margin": margin,
            "previous_group": previous_group,
            "switched": (
                previous_group is not None and group_idx != previous_group
            ),
            "relative_improvement": (
                None if np.isnan(improvements[idx]) else float(improvements[idx])
            ),
            "hysteresis_retained": bool(retained[idx]),
            "forced_rebalance": bool(forced[idx]),
            "block_distances": {
                block: float(values[idx, group_idx])
                for block, values in block_matrices.items()
            },
            "block_nearest_groups": {
                block: int(np.argmin(values[idx]))
                for block, values in block_matrices.items()
            },
        }
        details.append(detail)
        assignment_map[term] = detail

    diagnostics = {
        "group_counts": counts,
        "min_group_count": min_count,
        "max_group_count": max_count,
        "block_task_variance": block_task_variance,
    }
    return assignment_map, details, diagnostics


def train_step_v2(model, batch, outer_optimizer, config, device,
                  assignment_map=None):
    """Executes one meta-training step (Inner Loop + Outer Loop)."""
    q_in, q_lb, s_in, s_lb = batch[0], batch[1], batch[2], batch[3]
    term_batch = batch[4]
    s_in, s_lb = s_in.to(device), s_lb.to(device)
    q_in, q_lb = q_in.to(device), q_lb.to(device)

    inner_criterion = FocalLoss(alpha=0.5, gamma=config.focal_gamma)
    outer_criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)

    outer_optimizer.zero_grad()

    meta_loss_sum = 0.0
    focal_loss_sum = 0.0
    batch_preds_collect = []
    batch_labels_collect = []
    tsa_assignments = []

    B_tasks = s_in.shape[0]
    grad_keys = get_adaptive_param_names(model, config)

    for i in range(B_tasks):
        t_s_in, t_s_lb = s_in[i], s_lb[i]
        t_q_in, t_q_lb = q_in[i], q_lb[i]
        group_idx = None

        if config.tsa_enable:
            term = str(term_batch[i])
            if assignment_map is None:
                assignment = select_tsa_group(
                    model,
                    t_s_in,
                    t_s_lb,
                    config,
                    inner_criterion,
                    device,
                )
            else:
                if term not in assignment_map:
                    raise KeyError(f"Missing epoch TSA assignment for task {term}")
                assignment = dict(assignment_map[term])
            group_idx = assignment["group"]
            tsa_assignments.append({
                "disease_term": term,
                **assignment,
            })

        fast_weights = make_fast_weights(model, config, group_idx=group_idx, detach=False)
        alphas = get_alpha_dict(model, detach=False)

        fast_weights = adapt_fast_weights(
            model,
            t_s_in,
            t_s_lb,
            fast_weights,
            alphas,
            grad_keys,
            inner_criterion,
            config,
            config.inner_step,
            create_graph=True,
        )

        q_logits, _ = model.functional_forward(t_q_in.unsqueeze(0), fast_weights)
        q_logits = q_logits.squeeze(0)

        batch_preds_collect.append(torch.sigmoid(q_logits).detach().cpu())
        batch_labels_collect.append(t_q_lb.detach().cpu())

        task_loss = outer_criterion(q_logits, t_q_lb.unsqueeze(-1))
        meta_loss_sum += task_loss
        focal_loss_sum += task_loss.item()

    loss_for_backward = meta_loss_sum / B_tasks
    loss_for_backward.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    outer_optimizer.step()

    return (
        loss_for_backward.item(),
        focal_loss_sum / B_tasks,
        torch.cat(batch_preds_collect),
        torch.cat(batch_labels_collect),
        tsa_assignments,
    )


def evaluate_v2(model, task_loader, config, device, mode="Valid"):
    model.eval()
    task_results = []
    acc_dict = {k: [] for k in ["auroc", "auprc", "f1", "accuracy"]}
    eval_criterion = FocalLoss(alpha=0.5, gamma=config.focal_gamma)
    grad_keys = get_adaptive_param_names(model, config)

    for batch in tqdm(task_loader, desc=f"Eval ({mode})", leave=False):
        q_in, q_lb, s_in, s_lb = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        term_name = batch[4][0]

        if len(torch.unique(s_lb[0])) < 2:
            continue

        group_idx = None
        assignment = None
        if config.tsa_enable:
            assignment = select_tsa_group(
                model,
                s_in,
                s_lb,
                config,
                eval_criterion,
                device,
            )
            group_idx = assignment["group"]

        fast_weights = make_fast_weights(model, config, group_idx=group_idx, detach=True)
        alphas = get_alpha_dict(model, detach=True)

        fast_weights = adapt_fast_weights(
            model,
            s_in,
            s_lb,
            fast_weights,
            alphas,
            grad_keys,
            eval_criterion,
            config,
            config.inner_step + 5,
            create_graph=False,
        )

        with torch.no_grad():
            q_logits, _ = model.functional_forward(q_in, fast_weights)
            q_probs = torch.sigmoid(q_logits).squeeze(0)
            if torch.isnan(q_probs).any() or len(torch.unique(q_lb.squeeze(0))) < 2:
                continue

        try:
            metrics = compute_task_metrics(q_probs, q_lb.squeeze(0))
            if metrics:
                metrics["disease_term"] = str(term_name)
                if assignment is not None:
                    metrics["tsa_group"] = int(assignment["group"])
                    metrics["tsa_distance"] = assignment["distance"]
                    metrics["tsa_margin"] = assignment["margin"]
                    metrics["tsa_block_distances"] = assignment.get(
                        "block_distances",
                        {},
                    )
                    metrics["tsa_block_nearest_groups"] = assignment.get(
                        "block_nearest_groups",
                        {},
                    )
                task_results.append(metrics)
                for k in acc_dict:
                    if k in metrics:
                        acc_dict[k].append(metrics[k])
        except Exception:
            pass

    summary = {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in acc_dict.items()}
    return summary, task_results


def compute_group_drifts(model):
    if model.tsa_initial_group_vectors is None:
        return []
    current = flatten_current_group_params(model, model.tsa_param_slices).detach()
    normalized_delta = (
        current - model.tsa_initial_group_vectors
    ) / model.tsa_vector_std.unsqueeze(0)
    return torch.sqrt((normalized_delta ** 2).mean(dim=1)).cpu().tolist()


def summarize_epoch_assignments(assignments, previous_assignments, num_task_groups):
    if not assignments:
        return {
            "group_counts": [0] * num_task_groups,
            "mean_distance": None,
            "mean_margin": None,
            "switch_rate": None,
            "hysteresis_retention_rate": None,
            "forced_rebalance_rate": None,
            "mean_block_distances": {},
            "block_nearest_agreement": {},
            "current_assignments": {},
            "details": [],
        }

    current_assignments = {
        item["disease_term"]: int(item["group"])
        for item in assignments
    }
    distances = [item["distance"] for item in assignments]
    margins = [item["margin"] for item in assignments if item["margin"] is not None]
    retained = [
        bool(item.get("hysteresis_retained", False))
        for item in assignments
    ]
    forced = [
        bool(item.get("forced_rebalance", False))
        for item in assignments
    ]

    block_names = sorted({
        block
        for item in assignments
        for block in item.get("block_distances", {})
    })
    mean_block_distances = {}
    block_nearest_agreement = {}
    for block in block_names:
        block_values = [
            item["block_distances"][block]
            for item in assignments
            if block in item.get("block_distances", {})
        ]
        agreements = [
            item["block_nearest_groups"][block] == item["group"]
            for item in assignments
            if block in item.get("block_nearest_groups", {})
        ]
        mean_block_distances[block] = (
            float(np.mean(block_values)) if block_values else None
        )
        block_nearest_agreement[block] = (
            float(np.mean(agreements)) if agreements else None
        )

    switch_rate = None
    if previous_assignments is not None:
        common_terms = set(previous_assignments).intersection(current_assignments)
        if common_terms:
            switches = sum(
                previous_assignments[term] != current_assignments[term]
                for term in common_terms
            )
            switch_rate = switches / len(common_terms)

    details = []
    for item in assignments:
        previous_group = None
        switched = None
        if previous_assignments is not None:
            previous_group = previous_assignments.get(item["disease_term"])
            if previous_group is not None:
                switched = previous_group != item["group"]
        details.append({
            **item,
            "previous_group": previous_group,
            "switched": switched,
        })

    return {
        "group_counts": np.bincount(
            [item["group"] for item in assignments],
            minlength=num_task_groups,
        ).astype(int).tolist(),
        "mean_distance": float(np.mean(distances)),
        "mean_margin": float(np.mean(margins)) if margins else None,
        "switch_rate": switch_rate,
        "hysteresis_retention_rate": float(np.mean(retained)),
        "forced_rebalance_rate": float(np.mean(forced)),
        "mean_block_distances": mean_block_distances,
        "block_nearest_agreement": block_nearest_agreement,
        "current_assignments": current_assignments,
        "details": details,
    }


def main():
    args, config = parse_args()
    set_seed(args.random_seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    mode_name = "TSA-ProMeta" if config.tsa_enable else "ProMeta"
    print(f"Running {mode_name} on Device: {device} | Config: OutLR={args.outer_lr}, InLR={args.inner_lr}")

    p_data_df = pd.read_csv(args.proteomics_csv)
    p_data_df["EID"] = p_data_df["EID"].apply(lambda x: str(x).strip().replace(".0", ""))
    eid_to_idx = {e: i for i, e in enumerate(p_data_df["EID"].values)}
    protein_names = p_data_df.drop(columns=["EID"]).columns.tolist()
    proteins = np.nan_to_num(p_data_df.drop(columns=["EID"]).values.astype(np.float32))

    def load_pkl(n):
        return pkl.load(open(os.path.join(args.data_dir, n), "rb"))

    train_case, train_ctrl = load_pkl("term2pre_cases_train.pkl"), load_pkl("term2pre_controls_train.pkl")
    valid_case, valid_ctrl = load_pkl("term2pre_cases_valid.pkl"), load_pkl("term2pre_controls_valid.pkl")
    test_case, test_ctrl = load_pkl("term2pre_cases_test.pkl"), load_pkl("term2pre_controls_test.pkl")

    pathway_mask, unknown_indices = generate_pathway_mask(protein_names, args.cpdb_path)
    pathway_mask = pathway_mask.to(device)

    train_loader = DataLoader(
        MetaDataset(
            proteins,
            train_case,
            train_ctrl,
            eid_to_idx,
            args.support_size,
            max_support_size=args.max_support_size,
            query_size=config.query_size,
            mode="train",
            random_seed=args.random_seed,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        MetaDataset(
            proteins,
            valid_case,
            valid_ctrl,
            eid_to_idx,
            args.support_size,
            max_support_size=args.max_support_size,
            query_size=config.query_size,
            mode="valid",
            random_seed=args.random_seed,
        ),
        batch_size=1,
        shuffle=False,
    )
    test_loader = DataLoader(
        MetaDataset(
            proteins,
            test_case,
            test_ctrl,
            eid_to_idx,
            args.support_size,
            max_support_size=args.max_support_size,
            query_size=config.query_size,
            mode="test",
            random_seed=args.random_seed,
        ),
        batch_size=1,
        shuffle=False,
    )

    model = ProphetBioGateModel(proteins.shape[1], config, pathway_mask, unknown_indices).to(device)
    prepare_tsa_model(model, train_loader, config, args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.outer_lr)

    best_auroc = 0.0
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "val_auroc": [],
        "val_auprc": [],
        "early_stopped_epoch": None,
    }
    if config.tsa_enable:
        history["tsa_cluster_counts"] = model.tsa_cluster_counts
        history["tsa_group_usage"] = []
        history["tsa_mean_assignment_distance"] = []
        history["tsa_mean_assignment_margin"] = []
        history["tsa_group_switch_rate"] = []
        history["tsa_hysteresis_retention_rate"] = []
        history["tsa_forced_rebalance_rate"] = []
        history["tsa_group_drift"] = []
        history["tsa_mean_block_distances"] = []
        history["tsa_block_nearest_agreement"] = []
        history["tsa_block_task_variance"] = []
        history["tsa_group_size_bounds"] = []
        history["tsa_assignment_details"] = []
        history["tsa_assignment_config"] = {
            "selector_source": config.tsa_selector_source,
            "assignment_source": config.tsa_assignment_source,
            "distance_mode": config.tsa_distance_mode,
            "gate_distance_weight": config.tsa_gate_distance_weight,
            "selector_l1_lambda": config.tsa_selector_l1_lambda,
            "routing_schedule": config.tsa_routing_schedule,
            "switch_threshold": config.tsa_switch_threshold,
            "min_group_fraction": config.tsa_min_group_fraction,
            "max_group_fraction": config.tsa_max_group_fraction,
        }
    previous_epoch_assignments = None

    save_dir = os.path.join(args.output_dir, "checkpoints", f"support_{args.support_size}")
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f"{mode_name}_best_seed{args.random_seed}.pth")

    for epoch in range(config.epochs):
        epoch_assignment_map = None
        routing_details = None
        routing_diagnostics = {}
        if config.tsa_enable and config.tsa_routing_schedule == "epoch_snapshot":
            (
                epoch_assignment_map,
                routing_details,
                routing_diagnostics,
            ) = build_epoch_assignment_map(
                model,
                train_loader,
                config,
                device,
                previous_epoch_assignments,
            )

        model.train()
        l_sum, focal_sum, train_probs_all, train_labels_all = 0.0, 0.0, [], []
        epoch_assignments = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            loss, focal_val, b_probs, b_labels, groups = train_step_v2(
                model,
                batch,
                optimizer,
                config,
                device,
                assignment_map=epoch_assignment_map,
            )
            l_sum += loss
            focal_sum += focal_val
            train_probs_all.append(b_probs)
            train_labels_all.append(b_labels)
            epoch_assignments.extend(groups)

        train_metrics = compute_task_metrics(torch.cat(train_probs_all), torch.cat(train_labels_all)) or {"auroc": 0.0}
        val_summary, _ = evaluate_v2(model, valid_loader, config, device)

        history["train_loss"].append(l_sum / len(train_loader))
        history["val_auroc"].append(val_summary["auroc"])
        history["val_auprc"].append(val_summary["auprc"])

        group_msg = ""
        if config.tsa_enable:
            assignment_summary = summarize_epoch_assignments(
                routing_details if routing_details is not None else epoch_assignments,
                previous_epoch_assignments,
                config.num_task_groups,
            )
            previous_epoch_assignments = assignment_summary.pop("current_assignments")
            group_counts = assignment_summary["group_counts"]
            history["tsa_group_usage"].append(group_counts)
            history["tsa_mean_assignment_distance"].append(
                assignment_summary["mean_distance"]
            )
            history["tsa_mean_assignment_margin"].append(
                assignment_summary["mean_margin"]
            )
            history["tsa_group_switch_rate"].append(
                assignment_summary["switch_rate"]
            )
            history["tsa_hysteresis_retention_rate"].append(
                assignment_summary["hysteresis_retention_rate"]
            )
            history["tsa_forced_rebalance_rate"].append(
                assignment_summary["forced_rebalance_rate"]
            )
            history["tsa_group_drift"].append(compute_group_drifts(model))
            history["tsa_mean_block_distances"].append(
                assignment_summary["mean_block_distances"]
            )
            history["tsa_block_nearest_agreement"].append(
                assignment_summary["block_nearest_agreement"]
            )
            history["tsa_block_task_variance"].append(
                routing_diagnostics.get("block_task_variance", {})
            )
            history["tsa_group_size_bounds"].append({
                "min": routing_diagnostics.get("min_group_count"),
                "max": routing_diagnostics.get("max_group_count"),
            })
            history["tsa_assignment_details"].append(
                assignment_summary["details"]
            )
            group_msg = f" | TSA Groups: {group_counts}"
            if assignment_summary["mean_distance"] is not None:
                group_msg += f" | Dist: {assignment_summary['mean_distance']:.4f}"
            if assignment_summary["mean_margin"] is not None:
                group_msg += f" | Margin: {assignment_summary['mean_margin']:.4f}"
            if assignment_summary["switch_rate"] is not None:
                group_msg += f" | Switch: {assignment_summary['switch_rate']:.3f}"
            if assignment_summary["forced_rebalance_rate"]:
                group_msg += (
                    f" | Rebalance: "
                    f"{assignment_summary['forced_rebalance_rate']:.3f}"
                )

        print(
            f"Epoch {epoch + 1} | Loss: {l_sum / len(train_loader):.4f} | "
            f"Train AUC: {train_metrics['auroc']:.4f} | Val AUC: {val_summary['auroc']:.4f}"
            f"{group_msg}"
        )

        if val_summary["auroc"] > best_auroc:
            best_auroc = val_summary["auroc"]
            epochs_without_improvement = 0
            torch.save(build_model_checkpoint(model, config, args), best_model_path)
            print(f"[Info] New Best Model Saved (AUROC: {best_auroc:.4f})")
        else:
            epochs_without_improvement += 1
            if config.patience > 0 and epochs_without_improvement >= config.patience:
                history["early_stopped_epoch"] = epoch + 1
                print(
                    f"[Info] Early stopping at epoch {epoch + 1}; "
                    f"best validation AUROC: {best_auroc:.4f}"
                )
                break

    print("\n--- Testing ---")
    if os.path.exists(best_model_path):
        load_model_checkpoint(model, best_model_path, device, strict=True, load_metadata=True)
    test_summary, test_results = evaluate_v2(model, test_loader, config, device, mode="Test")
    print(f"[Result] Final Test AUROC: {test_summary['auroc']:.4f}")
    save_results(test_summary, test_results, args, mode_name, args.output_dir, history)


if __name__ == "__main__":
    main()
