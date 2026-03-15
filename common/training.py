"""Shared training loop and evaluation utilities.

Provides train_epoch, evaluate, and run_training for all benchmarks.
Results are logged to results.json for tracking across experiments.
"""

import gc
import json
import os
import time
from datetime import datetime

import torch
import torch.nn.functional as F


def train_epoch(model, loader, optimizer, device, augment_fn=None,
                label_smoothing=0.0, clip_norm=1.0, scaler=None):
    """Train for one epoch.

    Args:
        model: SNN model
        loader: DataLoader
        optimizer: optimizer
        device: torch device
        augment_fn: optional callable(batch_input) -> augmented_input
        label_smoothing: label smoothing factor (0.0 = off)
        clip_norm: gradient clipping norm (None to disable)
        scaler: GradScaler for mixed precision (None to disable AMP)

    Returns:
        (avg_loss, accuracy) tuple
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None
    n_batches = len(loader)
    log_every = max(1, n_batches // 10)  # Print ~10 updates per epoch

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        if augment_fn is not None:
            inputs = augment_fn(inputs)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            output = model(inputs)
            loss = F.cross_entropy(output, labels, label_smoothing=label_smoothing)
            # Add auxiliary losses (e.g. activity regularization)
            if hasattr(model, 'aux_loss') and model.aux_loss is not None:
                loss = loss + model.aux_loss

        if use_amp:
            scaler.scale(loss).backward()
            if clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (output.argmax(1) == labels).sum().item()
        total += inputs.size(0)

        if (batch_idx + 1) % log_every == 0 or batch_idx == n_batches - 1:
            avg_loss = total_loss / total
            acc = correct / total * 100
            print(f"  [{batch_idx+1:4d}/{n_batches}] loss={avg_loss:.4f} acc={acc:.1f}%",
                  flush=True)
            gc.collect()  # Prevent numpy/tensor memory accumulation

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    """Evaluate model on a dataset.

    Returns:
        (avg_loss, accuracy) tuple
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            output = model(inputs)
            loss = F.cross_entropy(output, labels)
        total_loss += loss.item() * inputs.size(0)
        correct += (output.argmax(1) == labels).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


def run_training(model, train_loader, test_loader, config):
    """Full training pipeline with LR scheduling, checkpointing, and logging.

    Args:
        model: SNN model (already on device)
        train_loader: training DataLoader
        test_loader: test DataLoader
        config: dict with keys:
            - device: torch device
            - epochs: int
            - lr: float
            - weight_decay: float
            - save_path: str (checkpoint path)
            - benchmark: str (e.g. "shd", "nmnist")
            - augment_fn: optional callable
            - label_smoothing: float (default 0.0)
            - clip_norm: float (default 1.0)
            - lr_min: float (default 1e-5)
            - results_file: str (default "results.json")
            - resume_from: str (path to .last.pt checkpoint to resume from)
            - checkpoint_every: int (save full state every N epochs, default 10)
            - gc_every: int (run gc.collect() every N batches, default 0=off)

    Returns:
        dict with best_acc, best_epoch, final_acc, training_time
    """
    device = config['device']
    epochs = config['epochs']
    augment_fn = config.get('augment_fn')
    label_smoothing = config.get('label_smoothing', 0.0)
    clip_norm = config.get('clip_norm', 1.0)
    lr_min = config.get('lr_min', 1e-5)
    save_path = config.get('save_path', f"{config['benchmark']}_model.pt")
    results_file = config.get('results_file', 'results.json')
    checkpoint_every = config.get('checkpoint_every', 10)
    gc_every = config.get('gc_every', 0)

    # Allow external optimizer (e.g. for separate delay param LR groups)
    optimizer = config.get('optimizer')
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
        )

    # LR schedule: optional linear warmup + cosine decay
    warmup_epochs = config.get('warmup_epochs', 5)
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs - warmup_epochs, eta_min=lr_min)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, eta_min=lr_min,
        )

    # Mixed precision training
    use_amp = config.get('use_amp', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    patience = config.get('patience', 50)  # Early stopping patience
    print(f"Parameters: {n_params:,}")
    if use_amp:
        print("Mixed precision (AMP): enabled")
    print(f"Early stopping patience: {patience} epochs")

    best_acc = 0.0
    best_epoch = 0
    start_epoch = 0
    start_time = time.time()

    # Resume from checkpoint if specified
    resume_from = config.get('resume_from')
    if resume_from is None:
        last_pt = save_path + '.last.pt'
        if os.path.exists(last_pt):
            resume_from = last_pt
            print(f"Found existing checkpoint: {last_pt}")

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler is not None and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)
        best_epoch = ckpt.get('best_epoch', 0)
        print(f"Resumed at epoch {start_epoch}, best_acc={best_acc*100:.1f}%")

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            augment_fn=augment_fn,
            label_smoothing=label_smoothing,
            clip_norm=clip_norm,
            scaler=scaler,
        )
        test_loss, test_acc = evaluate(model, test_loader, device, use_amp=use_amp)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'config': config.get('model_config', {}),
            }, save_path)

        # Periodic full-state checkpoint for resume
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            last_path = save_path + '.last.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'best_acc': best_acc,
                'best_epoch': best_epoch,
                'config': config.get('model_config', {}),
            }, last_path)

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time
        remaining = epoch_time * (epochs - epoch - 1)
        eta_str = f"{remaining/60:.0f}m" if remaining < 3600 else f"{remaining/3600:.1f}h"

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train: {train_loss:.4f} / {train_acc*100:.1f}% | "
              f"Test: {test_loss:.4f} / {test_acc*100:.1f}% | "
              f"LR={lr:.2e} | Best={best_acc*100:.1f}% | "
              f"{epoch_time:.1f}s/ep | ETA {eta_str}")

        # Periodic garbage collection for long-running jobs
        if gc_every > 0 and (epoch + 1) % gc_every == 0:
            gc.collect()

        # Early stopping
        if epoch - best_epoch >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs")
            break

    training_time = time.time() - start_time

    result = {
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'final_acc': test_acc,
        'n_params': n_params,
        'training_time_s': round(training_time, 1),
    }

    print(f"\nDone. Best test accuracy: {best_acc*100:.1f}% (epoch {best_epoch+1})")
    print(f"Training time: {training_time/60:.1f} min")
    print(f"Model saved to {save_path}")

    # Log result
    _save_result(results_file, config.get('benchmark', 'unknown'), config, result)

    return result


def _save_result(results_file, benchmark, config, result):
    """Append result to results.json."""
    results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                results_file)
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        results = []

    model_config = config.get('model_config', {})
    entry = {
        'benchmark': benchmark,
        'accuracy_float': round(result['best_acc'] * 100, 2),
        'n_params': result['n_params'],
        'epochs': config.get('epochs'),
        'best_epoch': result['best_epoch'] + 1,
        'neuron_type': model_config.get('neuron_type', 'lif'),
        'hidden': model_config.get('hidden', model_config.get('n_hidden')),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'training_time_s': result['training_time_s'],
    }
    results.append(entry)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Result logged to {results_path}")
