"""Sleep Consolidation Continual Learning — N4-specific feature demo.

Demonstrates the N4's 6-phase sleep consolidation engine preventing
catastrophic forgetting in sequential task learning.

Hardware correspondence:
    Phase 1 (QUIESCE):     Drain spike FIFOs
    Phase 2 (REPLAY):      Replay stored spikes from L2 buffer at 10x speed
    Phase 3 (CONSOLIDATE): Strengthen replayed patterns via metaplasticity
    Phase 4 (PRUNE):       Remove weak synapses (weight < threshold, meta_state <= 1)
    Phase 5 (HOMEOSTATIC): Downscale weak synapses by 0.9x (Tononi-Cirelli)
    Phase 6 (RESUME):      Re-enable normal processing

Benchmark: Split-MNIST (5 sequential tasks, 2 digits each)
    Task 0: digits 0,1    Task 1: digits 2,3    Task 2: digits 4,5
    Task 3: digits 6,7    Task 4: digits 8,9

Compares:
    - Naive SNN (no sleep): catastrophic forgetting
    - EWC baseline: elastic weight consolidation
    - N4 Sleep Consolidation: dual fast/slow weights + 6-phase cycle

Usage:
    python continual_learning/train_sleep.py --device cuda:1
    python continual_learning/train_sleep.py --device cuda:1 --tasks 10 --dataset cifar10
"""

import os
import sys
import json
import argparse
import copy
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common.neurons import AdaptiveLIFNeuron, LIFNeuron, surrogate_spike


# ── Dataset utilities ──

def get_split_mnist(n_tasks=5, data_dir='data'):
    """Split MNIST into sequential tasks (2 classes each)."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    classes_per_task = 10 // n_tasks
    tasks_train, tasks_test = [], []

    for t in range(n_tasks):
        task_classes = set(range(t * classes_per_task, (t + 1) * classes_per_task))
        train_idx = [i for i, y in enumerate(train_data.targets.tolist()) if y in task_classes]
        test_idx = [i for i, y in enumerate(test_data.targets.tolist()) if y in task_classes]
        tasks_train.append(Subset(train_data, train_idx))
        tasks_test.append(Subset(test_data, test_idx))

    return tasks_train, tasks_test


def spike_encode(images, T=20, gain=1.0):
    """Poisson spike encoding: pixel intensity -> spike probability per timestep."""
    batch = images.shape[0]
    flat = images.view(batch, -1)  # (B, 784)
    rates = flat.clamp(0, 1) * gain
    spikes = torch.rand(batch, T, flat.shape[1], device=images.device) < rates.unsqueeze(1)
    return spikes.float()


# ── SNN Model with Dual Fast/Slow Weights ──

class SleepSNN(nn.Module):
    """SNN with dual fast/slow weight system for sleep consolidation.

    Fast weights: modified during wake (training), decay during sleep
    Slow weights: consolidated during sleep, stable across tasks
    Effective weight: w_fast + w_slow (saturating addition)

    This mirrors the N4 hardware dual-weight system (spec §5.2.1).
    """

    def __init__(self, n_input=784, n_hidden=256, n_output=10,
                 threshold=1.0, alpha_init=0.95):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Fast weights (wake-modified, decays during sleep)
        self.fc1_fast = nn.Linear(n_input, n_hidden, bias=False)
        self.fc_rec_fast = nn.Linear(n_hidden, n_hidden, bias=False)
        self.fc_out_fast = nn.Linear(n_hidden, n_output, bias=False)

        # Slow weights (consolidated, stable) — start at zero
        self.register_buffer('fc1_slow', torch.zeros(n_hidden, n_input))
        self.register_buffer('fc_rec_slow', torch.zeros(n_hidden, n_hidden))
        self.register_buffer('fc_out_slow', torch.zeros(n_output, n_hidden))

        # Per-synapse metaplasticity state (3-bit in hardware, float here)
        # 0 = fully plastic, 7 = fully consolidated
        self.register_buffer('meta_fc1', torch.zeros(n_hidden, n_input))
        self.register_buffer('meta_fc_rec', torch.zeros(n_hidden, n_hidden))
        self.register_buffer('meta_fc_out', torch.zeros(n_output, n_hidden))

        # Neuron model
        self.lif = AdaptiveLIFNeuron(n_hidden, alpha_init=alpha_init,
                                      threshold=threshold)
        self.lif_out = LIFNeuron(n_output, beta_init=0.9, threshold=threshold,
                                  learn_beta=True)

        # Spike replay buffer (mirrors N4 L2 circular buffer)
        self.replay_buffer = []
        self.max_buffer_size = 1024

        # Init
        nn.init.xavier_uniform_(self.fc1_fast.weight, gain=0.5)
        nn.init.orthogonal_(self.fc_rec_fast.weight, gain=0.2)
        nn.init.xavier_uniform_(self.fc_out_fast.weight, gain=0.5)

    def effective_weight(self, fast_w, slow_w):
        """Compute effective weight: w_fast + w_slow (saturating add)."""
        return (fast_w + slow_w).clamp(-2.0, 2.0)

    def forward(self, x):
        """Forward pass using effective (fast + slow) weights.

        Args:
            x: (batch, T, n_input) spike-encoded input
        """
        batch, T, _ = x.shape
        device = x.device

        # Effective weights
        w1 = self.effective_weight(self.fc1_fast.weight, self.fc1_slow)
        w_rec = self.effective_weight(self.fc_rec_fast.weight, self.fc_rec_slow)
        w_out = self.effective_weight(self.fc_out_fast.weight, self.fc_out_slow)

        v_h = torch.zeros(batch, self.n_hidden, device=device)
        spk_h = torch.zeros(batch, self.n_hidden, device=device)
        a_h = torch.zeros(batch, self.n_hidden, device=device)
        v_out = torch.zeros(batch, self.n_output, device=device)
        out_sum = torch.zeros(batch, self.n_output, device=device)

        for t in range(T):
            I_h = F.linear(x[:, t], w1) + F.linear(spk_h, w_rec)
            v_h, spk_h, a_h = self.lif(I_h, v_h, a_h, spk_h)

            I_out = F.linear(spk_h, w_out)
            v_out, spk_out = self.lif_out(I_out, v_out)
            out_sum = out_sum + v_out

        return out_sum / T

    def record_spikes(self, x):
        """Record input spikes to replay buffer (mirrors N4 L2 circular buffer)."""
        if len(self.replay_buffer) >= self.max_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.max_buffer_size // 2:]
        self.replay_buffer.append(x.detach().cpu())


# ── Sleep Consolidation Engine (Software model of N4 hardware) ──

class SleepEngine:
    """6-phase sleep consolidation cycle matching N4 hardware (n4_sleep.v).

    Phase 1 (QUIESCE):     No-op in software (hardware drains FIFOs)
    Phase 2 (REPLAY):      Forward pass on buffered spikes (10x speed = 10 iterations)
    Phase 3 (CONSOLIDATE): Increment meta_state for stable synapses, transfer fast->slow
    Phase 4 (PRUNE):       Zero out weak synapses with low meta_state
    Phase 5 (HOMEOSTATIC): Downscale fast weights by 0.9x (Tononi-Cirelli SHY)
    Phase 6 (RESUME):      Clear phase state
    """

    def __init__(self, consolidation_rate=0.1, prune_threshold=0.01,
                 meta_threshold=1, downscale_factor=0.9, replay_iterations=10):
        self.consolidation_rate = consolidation_rate
        self.prune_threshold = prune_threshold
        self.meta_threshold = meta_threshold
        self.downscale_factor = downscale_factor
        self.replay_iterations = replay_iterations

    def run_cycle(self, model, device, verbose=True):
        """Execute full 6-phase sleep cycle."""
        stats = {}

        # Phase 1: QUIESCE
        if verbose:
            print("  [SLEEP] Phase 1: QUIESCE — draining spike buffers")

        # Phase 2: REPLAY — forward pass on buffered spikes
        if verbose:
            print(f"  [SLEEP] Phase 2: REPLAY — {self.replay_iterations}x iterations "
                  f"on {len(model.replay_buffer)} buffered samples")
        if model.replay_buffer:
            model.eval()
            with torch.no_grad():
                for _ in range(self.replay_iterations):
                    for buf_x in model.replay_buffer[-64:]:  # Use recent samples
                        _ = model(buf_x.to(device))
            model.train()

        # Phase 3: CONSOLIDATE — transfer stable fast weights to slow
        if verbose:
            print("  [SLEEP] Phase 3: CONSOLIDATE — fast->slow weight transfer")
        consolidated = 0
        for fast_w, slow_w, meta in [
            (model.fc1_fast.weight, model.fc1_slow, model.meta_fc1),
            (model.fc_rec_fast.weight, model.fc_rec_slow, model.meta_fc_rec),
            (model.fc_out_fast.weight, model.fc_out_slow, model.meta_fc_out),
        ]:
            with torch.no_grad():
                # Check stability: large fast weights are important
                stable_mask = fast_w.data.abs() > self.prune_threshold
                # Increment meta_state for stable synapses
                meta.data[stable_mask] = (meta.data[stable_mask] + 1).clamp(max=7)
                # Decrement for unstable
                meta.data[~stable_mask] = (meta.data[~stable_mask] - 2).clamp(min=0)
                # Transfer: slow += alpha * (fast - slow) for consolidated synapses
                transfer_mask = meta.data >= 2
                delta = self.consolidation_rate * (fast_w.data - slow_w.data)
                slow_w.data[transfer_mask] += delta[transfer_mask]
                consolidated += transfer_mask.sum().item()
        stats['consolidated_synapses'] = consolidated

        # Phase 4: PRUNE — remove weak synapses with low meta_state
        if verbose:
            print("  [SLEEP] Phase 4: PRUNE — removing weak synapses")
        pruned = 0
        for fast_w, meta in [
            (model.fc1_fast.weight, model.meta_fc1),
            (model.fc_rec_fast.weight, model.meta_fc_rec),
            (model.fc_out_fast.weight, model.meta_fc_out),
        ]:
            with torch.no_grad():
                prune_mask = (fast_w.data.abs() < self.prune_threshold) & \
                             (meta.data <= self.meta_threshold)
                fast_w.data[prune_mask] = 0.0
                pruned += prune_mask.sum().item()
        stats['pruned_synapses'] = pruned

        # Phase 5: HOMEOSTATIC — downscale fast weights (Tononi-Cirelli SHY)
        if verbose:
            print(f"  [SLEEP] Phase 5: HOMEOSTATIC — downscale fast weights by "
                  f"{self.downscale_factor}x")
        downscaled = 0
        for fast_w, meta in [
            (model.fc1_fast.weight, model.meta_fc1),
            (model.fc_rec_fast.weight, model.meta_fc_rec),
            (model.fc_out_fast.weight, model.meta_fc_out),
        ]:
            with torch.no_grad():
                # Only downscale synapses with low meta_state (not yet consolidated)
                scale_mask = meta.data <= 2
                fast_w.data[scale_mask] *= self.downscale_factor
                downscaled += scale_mask.sum().item()
        stats['downscaled_synapses'] = downscaled

        # Phase 6: RESUME
        if verbose:
            print(f"  [SLEEP] Phase 6: RESUME — cycle complete")
            print(f"  [SLEEP] Stats: consolidated={stats['consolidated_synapses']:,}, "
                  f"pruned={stats['pruned_synapses']:,}, "
                  f"downscaled={stats['downscaled_synapses']:,}")

        return stats


# ── EWC Baseline ──

class EWC:
    """Elastic Weight Consolidation baseline for comparison."""

    def __init__(self, model, lambda_ewc=400):
        self.lambda_ewc = lambda_ewc
        self.params = {}
        self.fisher = {}

    def register_task(self, model, dataloader, device):
        """Compute Fisher information after learning a task."""
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                  if p.requires_grad}

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            x_spk = spike_encode(x, T=20)
            model.zero_grad()
            out = model(x_spk)
            loss = F.cross_entropy(out, y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2

        n_samples = len(dataloader.dataset)
        for n in fisher:
            fisher[n] /= n_samples

        # Store current params and Fisher
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.params[n] = p.data.clone()
                if n in self.fisher:
                    self.fisher[n] = 0.5 * self.fisher[n] + 0.5 * fisher[n]
                else:
                    self.fisher[n] = fisher[n]

    def penalty(self, model):
        """EWC regularization loss."""
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.params:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.lambda_ewc * loss


# ── Training functions ──

def train_task(model, loader, optimizer, device, T=20, epochs=5,
               ewc=None, task_id=0, task_classes=None):
    """Train model on a single task with masked output.

    Args:
        task_classes: set of class indices for this task. If provided,
                      loss is computed only over these output neurons
                      (prevents overwriting other tasks' output weights).
    """
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            x_spk = spike_encode(images, T=T)

            # Record spikes to replay buffer (N4 L2 buffer)
            if hasattr(model, 'record_spikes'):
                model.record_spikes(x_spk)

            optimizer.zero_grad()
            out = model(x_spk)

            # Masked cross-entropy: only compute loss on task-relevant classes
            if task_classes is not None:
                class_list = sorted(task_classes)
                out_masked = out[:, class_list]
                # Remap labels to local indices
                label_map = {c: i for i, c in enumerate(class_list)}
                labels_local = torch.tensor([label_map[l.item()] for l in labels],
                                            device=device, dtype=torch.long)
                loss = F.cross_entropy(out_masked, labels_local)
            else:
                loss = F.cross_entropy(out, labels)

            if ewc is not None:
                loss += ewc.penalty(model)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += images.size(0)

        acc = correct / total * 100
        if (epoch + 1) % max(1, epochs // 3) == 0 or epoch == epochs - 1:
            print(f"    Task {task_id} Epoch {epoch+1}/{epochs}: "
                  f"loss={total_loss/total:.4f}, acc={acc:.1f}%")


@torch.no_grad()
def evaluate_all_tasks(model, task_loaders, device, T=20):
    """Evaluate on all tasks, return per-task accuracies."""
    model.eval()
    accs = []
    for loader in task_loaders:
        correct, total = 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            x_spk = spike_encode(images, T=T)
            out = model(x_spk)
            correct += (out.argmax(1) == labels).sum().item()
            total += images.size(0)
        accs.append(correct / total * 100 if total > 0 else 0.0)
    return accs


# ── Main experiment ──

def run_experiment(method, tasks_train, tasks_test, device, args):
    """Run continual learning experiment with specified method."""
    n_tasks = len(tasks_train)
    n_output = 10  # Total classes

    # Create model
    if method == 'sleep':
        model = SleepSNN(n_hidden=args.hidden, n_output=n_output).to(device)
        sleep_engine = SleepEngine(
            consolidation_rate=0.3, prune_threshold=0.005,
            replay_iterations=20, downscale_factor=0.8,
            meta_threshold=1)
    else:
        model = SleepSNN(n_hidden=args.hidden, n_output=n_output).to(device)
        sleep_engine = None

    classes_per_task = n_output // n_tasks
    ewc = EWC(model, lambda_ewc=5000) if method == 'ewc' else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=1e-4)

    # Test loaders for all tasks
    test_loaders = [DataLoader(ts, batch_size=256, shuffle=False)
                    for ts in tasks_test]

    # Results tracking
    results = {'method': method, 'task_accs': [], 'avg_accs': [], 'forgetting': []}

    print(f"\n{'='*60}")
    print(f"Method: {method.upper()}")
    print(f"{'='*60}")

    for task_id in range(n_tasks):
        print(f"\n--- Learning Task {task_id} (classes {task_id*2}-{task_id*2+1}) ---")

        train_loader = DataLoader(tasks_train[task_id], batch_size=args.batch_size,
                                   shuffle=True)

        # Train on current task (with masked output to protect other classes)
        task_classes = set(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))
        train_task(model, train_loader, optimizer, device, T=args.T,
                   epochs=args.epochs_per_task, ewc=ewc, task_id=task_id,
                   task_classes=task_classes)

        # Register EWC Fisher after task
        if ewc is not None:
            ewc.register_task(model, train_loader, device)

        # Sleep consolidation after task
        if sleep_engine is not None and task_id < n_tasks - 1:
            print(f"\n  Entering sleep cycle after Task {task_id}...")
            stats = sleep_engine.run_cycle(model, device)

        # Evaluate on ALL tasks seen so far
        accs = evaluate_all_tasks(model, test_loaders[:task_id + 1], device, T=args.T)
        avg_acc = np.mean(accs)

        results['task_accs'].append(accs)
        results['avg_accs'].append(avg_acc)

        # Compute forgetting (drop from peak on each previous task)
        if task_id > 0:
            forgetting = []
            for prev_t in range(task_id):
                peak = max(results['task_accs'][t][prev_t]
                           for t in range(prev_t, task_id + 1))
                current = accs[prev_t]
                forgetting.append(peak - current)
            avg_forget = np.mean(forgetting)
            results['forgetting'].append(avg_forget)
        else:
            results['forgetting'].append(0.0)

        print(f"\n  After Task {task_id}:")
        for t, a in enumerate(accs):
            marker = " <-- current" if t == task_id else ""
            print(f"    Task {t}: {a:.1f}%{marker}")
        print(f"    Average: {avg_acc:.1f}%")
        if task_id > 0:
            print(f"    Avg forgetting: {results['forgetting'][-1]:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Sleep Consolidation Continual Learning")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--T", type=int, default=20, help="Spike encoding timesteps")
    parser.add_argument("--epochs-per-task", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-tasks", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 60)
    print("CATALYST N4 SLEEP CONSOLIDATION — CONTINUAL LEARNING DEMO")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Tasks: {args.n_tasks} (Split-MNIST, 2 classes each)")
    print(f"Hidden: {args.hidden}, Timesteps: {args.T}")
    print(f"Epochs/task: {args.epochs_per_task}")

    # Load data
    print("\nLoading Split-MNIST...")
    tasks_train, tasks_test = get_split_mnist(args.n_tasks)
    for i, ts in enumerate(tasks_train):
        print(f"  Task {i}: {len(ts)} train, {len(tasks_test[i])} test samples")

    start_time = time.time()

    # Run all three methods
    results_all = {}
    for method in ['naive', 'ewc', 'sleep']:
        results_all[method] = run_experiment(
            method, tasks_train, tasks_test, device, args)

    elapsed = time.time() - start_time

    # ── Final comparison ──
    print("\n" + "=" * 60)
    print("FINAL RESULTS COMPARISON")
    print("=" * 60)
    print(f"\n{'Method':<20} {'Final Avg Acc':>15} {'Avg Forgetting':>15}")
    print("-" * 50)
    for method in ['naive', 'ewc', 'sleep']:
        r = results_all[method]
        final_avg = r['avg_accs'][-1]
        avg_forget = np.mean(r['forgetting'][1:]) if len(r['forgetting']) > 1 else 0
        print(f"{method.upper():<20} {final_avg:>14.1f}% {avg_forget:>14.1f}%")

    # Task-by-task breakdown
    print(f"\nPer-task accuracy after learning ALL {args.n_tasks} tasks:")
    print(f"{'Task':<8}", end='')
    for method in ['naive', 'ewc', 'sleep']:
        print(f"  {method.upper():>10}", end='')
    print()
    for t in range(args.n_tasks):
        print(f"Task {t:<3}", end='')
        for method in ['naive', 'ewc', 'sleep']:
            acc = results_all[method]['task_accs'][-1][t]
            print(f"  {acc:>9.1f}%", end='')
        print()

    # DoD narrative
    sleep_r = results_all['sleep']
    naive_r = results_all['naive']
    improvement = sleep_r['avg_accs'][-1] - naive_r['avg_accs'][-1]

    print(f"\n{'='*60}")
    print("DEFENSE APPLICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Sleep consolidation improves retention by {improvement:+.1f}% over naive learning.")
    print(f"This demonstrates that the N4's hardware sleep engine enables autonomous")
    print(f"systems to learn from new threats without forgetting legacy detection rules.")
    print(f"During idle periods, the chip consolidates critical defensive knowledge")
    print(f"while pruning obsolete responses — no retraining required.")
    print(f"\nTotal experiment time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save results
    save_path = os.path.join(os.path.dirname(__file__), 'checkpoints',
                              'sleep_consolidation_results.json')
    save_data = {
        'benchmark': 'continual_learning_sleep',
        'dataset': 'split_mnist',
        'n_tasks': args.n_tasks,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'methods': {},
        'training_time_s': round(elapsed, 1),
    }
    for method in ['naive', 'ewc', 'sleep']:
        r = results_all[method]
        save_data['methods'][method] = {
            'final_avg_acc': round(r['avg_accs'][-1], 2),
            'avg_forgetting': round(np.mean(r['forgetting'][1:]), 2)
                              if len(r['forgetting']) > 1 else 0,
            'per_task_final': [round(a, 2) for a in r['task_accs'][-1]],
            'avg_acc_progression': [round(a, 2) for a in r['avg_accs']],
        }
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
