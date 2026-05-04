from __future__ import annotations

from pathlib import Path
import math
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch as GeometricBatch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from .device import RuntimeContext, move_to_device, set_seed
from .metrics import regression_metrics, classification_metrics, summarize_fold_metrics
from .domain import DomainDiscriminator, domain_loss, cdan_representation
from .config import save_json


def _label_of(sample) -> int:
    return int(sample[0].y.view(-1)[0].item())


def _target_of(sample) -> float:
    return float(sample[0].y.view(-1)[0].item())


def _make_loader(dataset, batch_size: int, shuffle: bool, ctx: RuntimeContext, drop_last: bool = False):
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last) if ctx.is_distributed and shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=(shuffle and sampler is None), sampler=sampler, drop_last=drop_last)


def _pad_to_batch_multiple(items: list[Any], batch_size: int) -> list[Any]:
    """Mimic the original repository's batch-size padding for CDAN training.

    The old scripts duplicated the first samples so that train/test sets were a
    multiple of 128 before constructing ``zip(train_dataset, repeated_test)``.
    We keep this behavior only for the CDAN paired loader; the held-out test
    loader itself is not padded for final evaluation.
    """
    padded = list(items)
    if not padded or batch_size <= 0:
        return padded
    remainder = len(padded) % batch_size
    if remainder:
        padded.extend(padded[: batch_size - remainder])
    return padded


def _make_original_cdan_loader(source_set: list[Any], target_set: list[Any], batch_size: int, ctx: RuntimeContext):
    """Build the original-style CDAN loader: zip(source, repeated target)."""
    if not source_set or not target_set:
        return None, 0, 0
    source_padded = _pad_to_batch_multiple(source_set, batch_size)
    target_padded = _pad_to_batch_multiple(target_set, batch_size)
    repeat_factor = math.ceil(len(source_padded) / max(len(target_padded), 1)) + 10
    repeated_target = (target_padded * repeat_factor)[: len(source_padded)]
    paired = list(zip(source_padded, repeated_target))
    return _make_loader(paired, batch_size, True, ctx), len(source_padded), len(target_padded)



def _batch_num_graphs(batch) -> int:
    """Return the number of samples represented by a PyG mini-batch.

    CDAN receives a pair ``(source_batch, target_batch)`` where each element is
    itself a tuple/list of PyG ``Batch`` objects.  The first graph component owns
    the per-sample labels and has the correct ``num_graphs`` value.
    """
    first = batch[0] if isinstance(batch, (list, tuple)) else batch
    if hasattr(first, "num_graphs"):
        return int(first.num_graphs)
    if hasattr(first, "y"):
        return int(first.y.view(-1).size(0))
    raise TypeError(f"Cannot infer batch size from object of type {type(first)!r}")


def _merge_pyg_batches(source_batch, target_batch):
    """Merge source and target PyG batches so the model is called once.

    The previous CDAN implementation called ``model(source_batch)`` and then
    ``model(target_batch)`` before one backward pass.  Models in this project
    contain BatchNorm layers; the second forward updates BatchNorm running
    buffers in-place and invalidates the tensors saved by the first forward,
    which causes PyTorch's ``expected version ...`` RuntimeError.  Concatenating
    the two mini-batches and splitting the outputs keeps BatchNorm in training
    mode while avoiding that in-place version conflict.
    """
    if isinstance(source_batch, (list, tuple)):
        if not isinstance(target_batch, (list, tuple)) or len(source_batch) != len(target_batch):
            raise TypeError("source_batch and target_batch must have the same nested structure")
        merged = []
        for src_part, tgt_part in zip(source_batch, target_batch):
            if hasattr(src_part, "to_data_list") and hasattr(tgt_part, "to_data_list"):
                merged.append(GeometricBatch.from_data_list(src_part.to_data_list() + tgt_part.to_data_list()))
            elif torch.is_tensor(src_part) and torch.is_tensor(tgt_part):
                merged.append(torch.cat([src_part, tgt_part], dim=0))
            else:
                raise TypeError(f"Unsupported CDAN batch component: {type(src_part)!r}")
        return type(source_batch)(merged)
    if hasattr(source_batch, "to_data_list") and hasattr(target_batch, "to_data_list"):
        return GeometricBatch.from_data_list(source_batch.to_data_list() + target_batch.to_data_list())
    if torch.is_tensor(source_batch) and torch.is_tensor(target_batch):
        return torch.cat([source_batch, target_batch], dim=0)
    raise TypeError(f"Unsupported CDAN batch type: {type(source_batch)!r}")


def _split_merged_outputs(output: torch.Tensor, rep: torch.Tensor, raw: torch.Tensor, source_n: int):
    return output[:source_n], rep[:source_n], raw[:source_n], output[source_n:], rep[source_n:], raw[source_n:]


def _supervised_loss_from_y(output, y, task_type: str):
    if task_type == "regression":
        return torch.sqrt(F.mse_loss(output.float(), y.view_as(output).float()) + 1e-12)
    return F.nll_loss(output, y.view(-1).long())

def _forward(model, batch, task_type: str):
    out = model(batch)
    if task_type == "regression":
        pred, rep = out[0], out[1]
        return pred, rep, pred
    log_probs, rep = out[0], out[1]
    raw_logits = out[-1] if isinstance(out, (tuple, list)) and hasattr(out[-1], "shape") and out[-1].ndim == 2 else log_probs
    return log_probs, rep, raw_logits


def _supervised_loss(output, batch, task_type: str):
    if task_type == "regression":
        y = batch[0].y.view_as(output).float()
        return torch.sqrt(F.mse_loss(output.float(), y) + 1e-12)
    y = batch[0].y.view(-1).long()
    return F.nll_loss(output, y)


@torch.no_grad()
def evaluate(model, loader, task_type: str, device: torch.device) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    y_true: list[Any] = []
    y_pred: list[Any] = []
    y_score: list[Any] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        output, rep, raw = _forward(model, batch, task_type)
        loss = _supervised_loss(output, batch, task_type)
        losses.append(float(loss.detach().cpu()))
        if task_type == "regression":
            y = batch[0].y.view_as(output).detach().cpu().numpy().reshape(-1)
            pred = output.detach().cpu().numpy().reshape(-1)
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
        else:
            y = batch[0].y.view(-1).detach().cpu().numpy()
            probs = torch.exp(output).detach().cpu().numpy()
            pred = probs.argmax(axis=1)
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
            y_score.extend(probs.tolist())
    metrics = {"loss": float(np.mean(losses)) if losses else None}
    if task_type == "regression":
        metrics.update(regression_metrics(y_true, y_pred))
    else:
        metrics.update(classification_metrics(y_true, y_pred, y_score))
    return metrics


def _split_train_val(train_indices, dataset, task_type: str, val_ratio: float, seed: int):
    if val_ratio <= 0:
        return list(train_indices), []
    stratify = None
    if task_type == "classification":
        labels = np.array([_label_of(dataset[i]) for i in train_indices])
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) > 1 and counts.min() >= 2:
            stratify = labels
    train_idx, val_idx = train_test_split(list(train_indices), test_size=val_ratio, random_state=seed, stratify=stratify)
    return list(train_idx), list(val_idx)


def _make_outer_splits(dataset, task_type: str, cfg: dict[str, Any]):
    split_cfg = cfg.get("split", {})
    mode = split_cfg.get("mode", "kfold")
    seed = int(cfg.get("seed", 432))
    indices = np.arange(len(dataset))
    if mode == "kfold":
        n_splits = int(split_cfg.get("num_folds", 10))
        if task_type == "classification":
            labels = np.array([_label_of(s) for s in dataset])
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) > 1 and counts.min() >= n_splits:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                yield from splitter.split(indices, labels)
                return
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(indices)
    elif mode == "repeated_holdout":
        n_repeats = int(split_cfg.get("num_repeats", 10))
        test_size = float(split_cfg.get("test_size", 0.1))
        for rep in range(n_repeats):
            stratify = None
            if task_type == "classification":
                labels = np.array([_label_of(s) for s in dataset])
                unique, counts = np.unique(labels, return_counts=True)
                if len(unique) > 1 and counts.min() >= 2:
                    stratify = labels
            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed + rep, stratify=stratify)
            yield train_idx, test_idx
    else:
        raise ValueError(f"Unsupported split mode: {mode}")


def _select_metric(task_type: str, cfg: dict[str, Any]) -> tuple[str, bool]:
    metric = cfg.get("training", {}).get("monitor")
    if metric:
        maximize = bool(cfg.get("training", {}).get("monitor_maximize", metric in {"auc", "f1_macro", "r2", "corr", "accuracy"}))
        return metric, maximize
    return ("r2", True) if task_type == "regression" else ("auc", True)


def _selection_source(cfg: dict[str, Any]) -> str:
    """Return which split is allowed to drive scheduler/best-epoch selection."""
    train_cfg = cfg.get("training", {})
    domain_cfg = cfg.get("domain_adaptation", {})
    value = str(train_cfg.get("selection_source", domain_cfg.get("selection_source", "validation"))).lower()
    if value in {"test", "test_fold", "original", "github"}:
        return "test"
    return "validation"


def _original_test_monitor(task_type: str) -> tuple[str, bool]:
    """Match the GitHub scripts' test-driven model-selection rule.

    Regression keeps the epoch with the smallest test loss.  Classification keeps
    the epoch with the largest test AUC.  In both cases the learning-rate
    scheduler follows test loss, as in the original scripts.
    """
    return ("loss", False) if task_type == "regression" else ("auc", True)


def _better(value, best, maximize: bool) -> bool:
    if value is None:
        return False
    if best is None:
        return True
    return value > best if maximize else value < best


def train_one_fold(model_class, dataset, train_indices, test_indices, fold_id: int, cfg: dict[str, Any], task_type: str, out_dir: Path, ctx: RuntimeContext) -> dict[str, Any]:
    train_cfg = cfg.get("training", {})
    split_cfg = cfg.get("split", {})
    domain_cfg = cfg.get("domain_adaptation", {})
    seed = int(cfg.get("seed", 432)) + fold_id
    set_seed(seed)

    train_inner, val_indices = _split_train_val(train_indices, dataset, task_type, float(split_cfg.get("val_ratio", 0.1)), seed)
    source_indices = list(train_inner)
    use_cdan = bool(domain_cfg.get("enabled", False))

    # Original GitHub CDAN uses the current fold's test set as the unlabeled target domain.
    # Set target_source: train if the leakage-safe train-internal target is desired.
    cdan_target_source = str(domain_cfg.get("target_source", "test" if use_cdan else "none")).lower()
    cdan_target_indices: list[int] = []
    if use_cdan:
        if cdan_target_source in {"test", "test_fold", "original", "github"}:
            cdan_target_indices = list(test_indices)
            cdan_target_source = "test"
        elif cdan_target_source in {"train", "train_internal", "source"} and len(source_indices) >= 4:
            ratio = float(domain_cfg.get("target_ratio", 0.2))
            source_indices, cdan_target_indices = train_test_split(source_indices, test_size=ratio, random_state=seed)
            source_indices = list(source_indices)
            cdan_target_indices = list(cdan_target_indices)
            cdan_target_source = "train_internal"
        else:
            cdan_target_indices = []
            cdan_target_source = "none"

    assert set(source_indices).isdisjoint(test_indices)
    assert set(val_indices).isdisjoint(test_indices)
    assert set(source_indices).isdisjoint(val_indices)
    if cdan_target_source != "test":
        assert set(cdan_target_indices).isdisjoint(test_indices)

    source_set = [dataset[i] for i in source_indices]
    val_set = [dataset[i] for i in val_indices] if val_indices else [dataset[i] for i in source_indices]
    test_set = [dataset[i] for i in test_indices]
    target_set = [dataset[i] for i in cdan_target_indices] if cdan_target_indices else []

    batch_size = int(train_cfg.get("batch_size", 128))
    train_loader = _make_loader(source_set, batch_size, True, ctx)
    val_loader = _make_loader(val_set, batch_size, False, ctx)
    test_loader = _make_loader(test_set, batch_size, False, ctx)
    cdan_loader, num_cdan_source_padded, num_cdan_target_padded = _make_original_cdan_loader(source_set, target_set, batch_size, ctx) if use_cdan else (None, 0, 0)

    hidden = int(cfg.get("model", {}).get("hidden", 32))
    num_layers = int(cfg.get("model", {}).get("num_layers", 2))
    try:
        model = model_class(source_set, num_layers, hidden, cfg=cfg)
    except TypeError:
        model = model_class(source_set, num_layers, hidden)
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    model = model.to(ctx.device)
    if ctx.is_distributed:
        # Ablation switches intentionally suppress feature branches.  Keep DDP
        # robust even if a future ablation disables a branch before it reaches
        # the loss graph.  The graph-preserving zeroing in model.py should make
        # most zeroed branches report zero gradients rather than missing ones;
        # find_unused_parameters is a safe fallback for the remaining cases.
        ablation_cfg = cfg.get("ablation", {})
        ddp_find_unused = bool(cfg.get("distributed", {}).get("find_unused_parameters", any(bool(v) for v in ablation_cfg.values())))
        model = DDP(
            model,
            device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
            find_unused_parameters=ddp_find_unused,
        )

    lr = float(train_cfg.get("lr", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    selection_source = _selection_source(cfg)
    monitor_name, maximize = _original_test_monitor(task_type) if selection_source == "test" else _select_metric(task_type, cfg)
    scheduler_mode = "min" if selection_source == "test" else ("max" if maximize else "min")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=float(train_cfg.get("lr_factor", 0.1)),
        patience=int(train_cfg.get("patience", 50)),
        min_lr=float(train_cfg.get("min_lr", 1e-7)),
    )

    discriminator = None
    opt_disc = None
    scheduler_disc = None
    lambda_domain = float(domain_cfg.get("lambda", 1.0))
    domain_alpha = float(domain_cfg.get("grl_alpha", 1.0))
    entropy_weight = bool(domain_cfg.get("entropy_weight", True))

    best_metric = None
    best_state = None
    history: list[dict[str, Any]] = []
    max_epochs = int(train_cfg.get("epochs", 150))

    for epoch in range(1, max_epochs + 1):
        model.train()
        if ctx.is_distributed and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            train_loader.sampler.set_epoch(epoch)
        if ctx.is_distributed and cdan_loader is not None and hasattr(cdan_loader, "sampler") and cdan_loader.sampler is not None:
            cdan_loader.sampler.set_epoch(epoch)

        epoch_losses = []
        if use_cdan and cdan_loader is not None:
            train_iter = cdan_loader
            for source_batch, target_batch in train_iter:
                source_n = _batch_num_graphs(source_batch)
                merged_batch = _merge_pyg_batches(source_batch, target_batch)
                merged_batch = move_to_device(merged_batch, ctx.device)
                optimizer.zero_grad(set_to_none=True)
                if opt_disc is not None:
                    opt_disc.zero_grad(set_to_none=True)

                output_all, rep_all, raw_all = _forward(model, merged_batch, task_type)
                output, rep, raw, _, rep_t, raw_t = _split_merged_outputs(output_all, rep_all, raw_all, source_n)
                loss = _supervised_loss_from_y(output, merged_batch[0].y[:source_n], task_type)

                if discriminator is None:
                    with torch.no_grad():
                        dim = cdan_representation(rep, raw, task_type).size(1)
                    discriminator = DomainDiscriminator(dim, hidden_dim=int(domain_cfg.get("hidden_dim", 256))).to(ctx.device)
                    if hasattr(discriminator, "reset_parameters"):
                        discriminator.reset_parameters()
                    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=float(domain_cfg.get("lr", 0.001)))
                    scheduler_disc = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        opt_disc,
                        mode="min" if selection_source == "test" else scheduler_mode,
                        factor=float(train_cfg.get("lr_factor", 0.1)),
                        patience=int(train_cfg.get("patience", 50)),
                        min_lr=float(train_cfg.get("min_lr", 1e-7)),
                    )
                    opt_disc.zero_grad(set_to_none=True)

                adv = domain_loss(rep, raw, rep_t, raw_t, task_type, discriminator, alpha=domain_alpha, entropy_weight=entropy_weight)
                total_loss = loss + lambda_domain * adv
                total_loss.backward()
                optimizer.step()
                opt_disc.step()
                epoch_losses.append(float(total_loss.detach().cpu()))
        else:
            for batch in train_loader:
                batch = move_to_device(batch, ctx.device)
                optimizer.zero_grad(set_to_none=True)
                output, rep, raw = _forward(model, batch, task_type)
                loss = _supervised_loss(output, batch, task_type)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

        if ctx.is_main:
            eval_model = model.module if isinstance(model, DDP) else model
            if selection_source == "test":
                val_metrics = {}
                test_epoch_metrics = evaluate(eval_model, test_loader, task_type, ctx.device)
                selection_metrics = test_epoch_metrics
                scheduler_value = test_epoch_metrics.get("loss")
            else:
                val_metrics = evaluate(eval_model, val_loader, task_type, ctx.device)
                test_epoch_metrics = {}
                selection_metrics = val_metrics
                scheduler_value = selection_metrics.get(monitor_name)
            selection_value = selection_metrics.get(monitor_name)
        else:
            val_metrics = {}
            test_epoch_metrics = {}
            selection_value = None
            scheduler_value = None

        if ctx.is_distributed:
            tensor = torch.tensor([-1e30 if scheduler_value is None else float(scheduler_value)], dtype=torch.float, device=ctx.device)
            torch.distributed.broadcast(tensor, src=0)
            scheduler_value_for_step = None if tensor.item() < -1e20 else float(tensor.item())
        else:
            scheduler_value_for_step = scheduler_value
        if scheduler_value_for_step is not None:
            scheduler.step(scheduler_value_for_step)
            if scheduler_disc is not None:
                scheduler_disc.step(scheduler_value_for_step)

        if ctx.is_main:
            row = {
                "fold": fold_id,
                "epoch": epoch,
                "selection_source": selection_source,
                "monitor": monitor_name,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_epoch_metrics.items()},
            }
            history.append(row)
            if _better(selection_value, best_metric, maximize):
                best_metric = selection_value
                eval_model = model.module if isinstance(model, DDP) else model
                best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}

        if optimizer.param_groups[0]["lr"] <= float(train_cfg.get("min_lr", 1e-7)):
            break

    if not ctx.is_main:
        return {"fold": fold_id}

    eval_model = model.module if isinstance(model, DDP) else model
    if best_state is not None:
        eval_model.load_state_dict(best_state)
    test_metrics = evaluate(eval_model, test_loader, task_type, ctx.device)
    fold_dir = out_dir / f"fold_{fold_id:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
    torch.save(best_state if best_state is not None else eval_model.state_dict(), fold_dir / "best_model.pt")
    save_json({
        "fold": fold_id,
        "source_train_indices": list(map(int, source_indices)),
        "validation_indices": list(map(int, val_indices)),
        "cdan_target_indices": list(map(int, cdan_target_indices)),
        "cdan_target_source": cdan_target_source,
        "selection_source": selection_source,
        "selection_monitor": monitor_name,
        "selection_maximize": maximize,
        "test_indices": list(map(int, test_indices)),
        "num_cdan_source_padded": int(num_cdan_source_padded),
        "num_cdan_target_padded": int(num_cdan_target_padded),
        "protocol_note": "CDAN target is the test fold when cdan_target_source='test', matching the original GitHub transductive logic.",
    }, fold_dir / "split_indices.json")
    row = {"fold": fold_id, "num_source_train": len(source_indices), "num_val": len(val_indices), "num_cdan_target": len(cdan_target_indices), "cdan_target_source": cdan_target_source, "selection_source": selection_source, "selection_monitor": monitor_name, "num_test": len(test_indices), **test_metrics}
    return row


def train_cross_validate(model_class, dataset, cfg: dict[str, Any], task_type: str, out_dir: str | Path, ctx: RuntimeContext) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domain_cfg = cfg.get("domain_adaptation", {})
    target_source = str(domain_cfg.get("target_source", "test" if domain_cfg.get("enabled", False) else "none")).lower()
    if ctx.is_main:
        save_json(cfg, out_dir / "config_used.json")
        save_json({
            "protocol": "original_github_transductive_cdan" if domain_cfg.get("enabled", False) and target_source in {"test", "test_fold", "original", "github"} else "reviewer_safe_inductive_or_no_cdan",
            "test_usage": "when CDAN is enabled with target_source='test', test-fold inputs are used as unlabeled target-domain samples during adversarial alignment; labels are not used for CDAN",
            "validation_usage": "when training.selection_source='test', validation is not used for epoch selection; this zip intentionally follows the original GitHub test-driven selection logic",
            "domain_adaptation": domain_cfg,
            "ablation": cfg.get("ablation", {}),
        }, out_dir / "protocol.json")
    rows: list[dict[str, Any]] = []
    for fold_id, (train_idx, test_idx) in enumerate(_make_outer_splits(dataset, task_type, cfg), start=1):
        row = train_one_fold(model_class, dataset, list(map(int, train_idx)), list(map(int, test_idx)), fold_id, cfg, task_type, out_dir, ctx)
        if ctx.is_main:
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "fold_metrics.csv", index=False)
            summary = summarize_fold_metrics(rows)
            save_json(summary, out_dir / "summary_metrics.json")
            pd.DataFrame([summary]).to_csv(out_dir / "summary_metrics.csv", index=False)
    if ctx.is_distributed:
        torch.distributed.barrier()
