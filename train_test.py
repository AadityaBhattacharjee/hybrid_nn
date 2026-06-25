"""
train_test.py
=============
Trains and evaluates the HybridNN on three classic benchmarks:

  1. Mirror Symmetry (Rumelhart 1986)
     6-input → 2-hidden → 1-output, 64 patterns, binary classification
     Target: learn symmetry feature with NO explicit supervision about what to look for

  2. N-bit Parity (Møller 1993)
     n-input → n-hidden → 1-output, all 2^n bit patterns, binary
     Stress-test on a notoriously hard error surface

  3. MNIST-style Forward-Forward inspired (Hinton 2022)
     Positive = real image + correct label channel
     Negative = same image + wrong label channel
     Local "goodness" objective, no global backprop
     (Runs on raw pixel MNIST if available, otherwise synthetic fallback)

Usage:
  python train_test.py [--dataset all|symmetry|parity|mnist]
                       [--parity_bits 4]
                       [--epochs 200]
                       [--batch_size 32]
                       [--seed 42]
                       [--verbose]
"""

import argparse
import sys
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from hybrid_nn import HybridNN


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(title: str) -> None:
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def print_results(name, train_res, test_res, elapsed, mut_count=0):
    print(f"\n  ┌─ {name} Results {'─'*(40-len(name))}")
    print(f"  │  Train  loss={train_res['loss']:.4f}  acc={train_res['accuracy']*100:.2f}%")
    print(f"  │  Test   loss={test_res['loss']:.4f}  acc={test_res['accuracy']*100:.2f}%")
    print(f"  │  Mutations triggered : {mut_count}")
    print(f"  │  Time               : {elapsed:.1f}s")
    print(f"  └{'─'*50}")


def epoch_log(epoch, total, train_loss, train_acc, val_loss, val_acc, verbose):
    if verbose and (epoch % max(1, total // 20) == 0 or epoch == total):
        print(f"  Epoch {epoch:4d}/{total}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.1f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Generic trainer
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model: HybridNN, X_train, y_train, X_val, y_val,
                epochs: int, batch_size: int, verbose: bool):
    best_val_acc = 0.0
    best_weights = None

    for ep in range(1, epochs + 1):
        train_loss = model.train_epoch(X_train, y_train, batch_size)
        train_res  = model.evaluate(X_train, y_train)
        val_res    = model.evaluate(X_val, y_val)

        epoch_log(ep, epochs,
                  train_loss, train_res["accuracy"],
                  val_res["loss"], val_res["accuracy"],
                  verbose)

        if val_res["accuracy"] > best_val_acc:
            best_val_acc = val_res["accuracy"]

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 1. MIRROR SYMMETRY (Rumelhart 1986)
# ─────────────────────────────────────────────────────────────────────────────

def make_symmetry_data():
    """
    Generate all 64 6-bit binary patterns.
    Label = 1 if pattern is mirror-symmetric, 0 otherwise.
    A 6-bit pattern [a,b,c,d,e,f] is symmetric iff a==f, b==e, c==d.
    """
    X, y = [], []
    for i in range(64):
        bits = [(i >> (5 - j)) & 1 for j in range(6)]
        label = int(bits[0] == bits[5] and bits[1] == bits[4] and bits[2] == bits[3])
        X.append(bits)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def run_symmetry(epochs=500, batch_size=8, seed=42, verbose=True):
    print_banner("Benchmark 1 — Mirror Symmetry (Rumelhart 1986)")
    print("  Architecture : 6 input → 2 hidden → 1 output")
    print("  Dataset      : 64 patterns (all 6-bit combinations)")
    print("  Task         : Learn mirror symmetry with NO explicit feature hints")
    print("  Split        : 80% train / 20% test (stratified)")

    X, y = make_symmetry_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed)

    print(f"  Train size   : {len(X_train)}  (symmetric={y_train.sum()})")
    print(f"  Test  size   : {len(X_test)}   (symmetric={y_test.sum()})")
    print()

    model = HybridNN(
        layer_sizes   = [6, 2, 1],
        activations   = ["relu", "sigmoid"],
        eta_h         = 0.015,
        lambda_decay  = 5e-4,
        eta_fa        = 0.010,
        alpha         = 0.40,
        sigma_0       = 0.01,
        gamma         = 4.0,
        plateau_k     = 10,
        eps_plateau   = 0.01,
        eta_adam      = 5e-3,
        eps_fd        = 1e-3,
        N_opt         = 10,
        top_k         = 1.0,   # keep all neurons (only 2 hidden)
        n_refresh     = 20,
        task          = "binary",
        seed          = seed,
    )

    t0 = time.time()
    train_model(model, X_train, y_train, X_test, y_test,
                epochs, batch_size, verbose)
    elapsed = time.time() - t0

    train_res = model.evaluate(X_train, y_train)
    test_res  = model.evaluate(X_test,  y_test)
    print_results("Mirror Symmetry", train_res, test_res,
                  elapsed, model._mut_count)

    if verbose:
        preds = model.predict(X_test)
        print("\n  Classification report (test set):")
        print(classification_report(y_test, preds,
              target_names=["Asymmetric", "Symmetric"], digits=3))
        print("  Confusion matrix:")
        print("  ", confusion_matrix(y_test, preds))

        # Show which hidden unit activations separate symmetric from not
        print("\n  Hidden unit activations (first 5 test samples):")
        acts = model.forward(X_test[:5])
        h    = acts[0]   # (5, 2)
        print("  Pattern               H1     H2    Label  Pred")
        for i in range(5):
            pat = "".join(str(int(b)) for b in X_test[i])
            print(f"  {pat}    {h[i,0]:.3f}  {h[i,1]:.3f}   "
                  f"{y_test[i]}       {preds[i]}")

    return model, test_res


# ─────────────────────────────────────────────────────────────────────────────
# 2. N-BIT PARITY (Møller 1993)
# ─────────────────────────────────────────────────────────────────────────────

def make_parity_data(n_bits: int):
    """
    All 2^n bit combinations. Label = XOR (sum of bits mod 2).
    Notoriously hard: error surface has deep ravines, saddle points.
    """
    n_patterns = 2 ** n_bits
    X = np.array([([(i >> (n_bits - 1 - j)) & 1 for j in range(n_bits)])
                  for i in range(n_patterns)], dtype=np.float32)
    y = (X.sum(axis=1) % 2).astype(np.int32)
    return X, y


def run_parity(n_bits=4, epochs=1000, batch_size=16, seed=42, verbose=True):
    print_banner(f"Benchmark 2 — {n_bits}-bit Parity (Møller 1993)")
    print(f"  Architecture : {n_bits} input → {n_bits} hidden → 1 output")
    print(f"  Dataset      : {2**n_bits} patterns (all {n_bits}-bit combinations)")
    print(f"  Task         : XOR parity — hard error surface stress test")
    print(f"  Split        : 80% train / 20% test (stratified)")

    X, y = make_parity_data(n_bits)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed)

    print(f"  Train size   : {len(X_train)}")
    print(f"  Test  size   : {len(X_test)}")
    print()

    model = HybridNN(
        layer_sizes   = [n_bits, n_bits, 1],
        activations   = ["relu", "sigmoid"],
        eta_h         = 0.012,
        lambda_decay  = 3e-4,
        eta_fa        = 0.008,
        alpha         = 0.35,
        sigma_0       = 0.015,
        gamma         = 5.0,
        plateau_k     = 8,
        eps_plateau   = 0.02,
        eta_adam      = 3e-3,
        eps_fd        = 1e-3,
        N_opt         = 5,
        top_k         = 0.80,  # keep 80% for small hidden
        n_refresh     = 15,
        task          = "binary",
        seed          = seed,
    )

    t0 = time.time()
    train_model(model, X_train, y_train, X_test, y_test,
                epochs, batch_size, verbose)
    elapsed = time.time() - t0

    train_res = model.evaluate(X_train, y_train)
    test_res  = model.evaluate(X_test,  y_test)
    print_results(f"{n_bits}-bit Parity", train_res, test_res,
                  elapsed, model._mut_count)

    if verbose:
        preds = model.predict(X_test)
        print("\n  Classification report (test set):")
        print(classification_report(y_test, preds,
              target_names=["Even parity", "Odd parity"], digits=3))
        print("  Confusion matrix:")
        print("  ", confusion_matrix(y_test, preds))

        print(f"\n  Loss at convergence : {model.loss_history[-1]:.6f}")
        print(f"  Epochs to reach loss < 0.10 : ", end="")
        reached = next((i+1 for i, l in enumerate(model.loss_history) if l < 0.10), None)
        print(reached if reached else "not reached")

    return model, test_res


# ─────────────────────────────────────────────────────────────────────────────
# 3. MNIST — Forward-Forward inspired (Hinton 2022)
# ─────────────────────────────────────────────────────────────────────────────

def make_mnist_ff_data(seed=42):
    """
    Hinton's Forward-Forward positive/negative construction:
      Positive: real image + correct one-hot label appended to pixel vector
      Negative: real image + WRONG one-hot label (randomly chosen ≠ true)

    This doubles the dataset size. Label for our binary classifier:
      1 = positive (real + correct label)
      0 = negative (real + wrong label)

    We attempt to load MNIST via sklearn / keras.
    Falls back to a synthetic 28×28 dataset if unavailable.
    """
    try:
        # Try sklearn fetch
        from sklearn.datasets import fetch_openml
        print("  Loading MNIST via sklearn (may download ~12MB first time)...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False,
                             parser="auto")
        X_raw = mnist.data.astype(np.float32) / 255.0   # (70000, 784)
        y_raw = mnist.target.astype(np.int32)
        print("  MNIST loaded: 70,000 samples")
        source = "MNIST (sklearn/OpenML)"
    except Exception as e:
        print(f"  MNIST unavailable ({e}). Using synthetic 28×28 data.")
        rng = np.random.default_rng(seed)
        n   = 5000
        X_raw = rng.uniform(0, 1, (n, 784)).astype(np.float32)
        y_raw = rng.integers(0, 10, n).astype(np.int32)
        source = "Synthetic (28×28, 10 classes)"

    rng = np.random.default_rng(seed)
    n_samples = len(X_raw)

    # Build positive samples: image ‖ one-hot(true label)
    one_hot_true = np.zeros((n_samples, 10), dtype=np.float32)
    one_hot_true[np.arange(n_samples), y_raw] = 1.0
    X_pos = np.concatenate([X_raw, one_hot_true], axis=1)  # (n, 794)

    # Build negative samples: image ‖ one-hot(wrong label)
    wrong_labels = (y_raw + rng.integers(1, 10, n_samples)) % 10
    one_hot_wrong = np.zeros((n_samples, 10), dtype=np.float32)
    one_hot_wrong[np.arange(n_samples), wrong_labels] = 1.0
    X_neg = np.concatenate([X_raw, one_hot_wrong], axis=1)  # (n, 794)

    X = np.concatenate([X_pos, X_neg], axis=0)             # (2n, 794)
    y = np.concatenate([np.ones(n_samples, dtype=np.int32),
                        np.zeros(n_samples, dtype=np.int32)], axis=0)

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm], source


def run_mnist(epochs=30, batch_size=256, seed=42, verbose=True):
    print_banner("Benchmark 3 — MNIST Forward-Forward (Hinton 2022)")
    print("  Architecture : 794 input → 512 → 256 → 1 output")
    print("  Input        : pixel vector (784) ‖ one-hot label (10)")
    print("  Positive     : image + correct label  → target 1")
    print("  Negative     : image + WRONG label    → target 0")
    print("  Task         : learn 'goodness' — no global backprop")
    print("  Split        : 80% train / 20% test")
    print()

    X, y, source = make_mnist_ff_data(seed)
    print(f"  Source       : {source}")
    print(f"  Total samples: {len(X)} (pos+neg pairs)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed)

    print(f"  Train size   : {len(X_train)}")
    print(f"  Test  size   : {len(X_test)}")
    print()

    model = HybridNN(
        layer_sizes   = [794, 512, 256, 1],
        activations   = ["relu", "relu", "sigmoid"],
        eta_h         = 0.008,
        lambda_decay  = 2e-4,
        eta_fa        = 0.005,
        alpha         = 0.35,
        sigma_0       = 0.005,
        gamma         = 6.0,
        plateau_k     = 5,
        eps_plateau   = 0.03,
        eta_adam      = 1e-3,
        eps_fd        = 1e-3,
        N_opt         = 5,
        top_k         = 0.25,
        n_refresh     = 5,
        task          = "binary",
        seed          = seed,
    )

    t0 = time.time()
    train_model(model, X_train, y_train, X_test, y_test,
                epochs, batch_size, verbose)
    elapsed = time.time() - t0

    train_res = model.evaluate(X_train, y_train)
    test_res  = model.evaluate(X_test,  y_test)
    print_results("MNIST Forward-Forward", train_res, test_res,
                  elapsed, model._mut_count)

    if verbose:
        preds = model.predict(X_test[:2000])
        print("\n  Classification report (first 2000 test samples):")
        print(classification_report(y_test[:2000], preds,
              target_names=["Negative (wrong label)", "Positive (correct label)"],
              digits=3))

    return model, test_res


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict):
    print("\n" + "═" * 60)
    print("  FINAL SUMMARY — Hybrid Non-Backprop Framework")
    print("═" * 60)
    print(f"  {'Benchmark':<32} {'Test Accuracy':>14}")
    print(f"  {'-'*32} {'-'*14}")
    for name, res in results.items():
        acc_str = f"{res['accuracy']*100:.2f}%"
        flag = " ✓" if res["accuracy"] >= 0.90 else " ✗ (<90%)"
        print(f"  {name:<32} {acc_str:>14}{flag}")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train & test HybridNN on classic benchmarks")
    parser.add_argument("--dataset",    default="all",
                        choices=["all", "symmetry", "parity", "mnist"])
    parser.add_argument("--parity_bits", type=int, default=4)
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--verbose",    action="store_true", default=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    results = {}

    # Default epochs per benchmark
    default_epochs = {
        "symmetry": 500,
        "parity":   1000,
        "mnist":    30,
    }
    default_batch = {
        "symmetry": 8,
        "parity":   16,
        "mnist":    256,
    }

    run_sym   = args.dataset in ("all", "symmetry")
    run_par   = args.dataset in ("all", "parity")
    run_mnist = args.dataset in ("all", "mnist")

    if run_sym:
        ep = args.epochs or default_epochs["symmetry"]
        bs = args.batch_size or default_batch["symmetry"]
        _, res = run_symmetry(epochs=ep, batch_size=bs,
                              seed=args.seed, verbose=args.verbose)
        results["Mirror Symmetry (Rumelhart 1986)"] = res

    if run_par:
        ep = args.epochs or default_epochs["parity"]
        bs = args.batch_size or default_batch["parity"]
        _, res = run_parity(n_bits=args.parity_bits, epochs=ep,
                            batch_size=bs, seed=args.seed, verbose=args.verbose)
        results[f"{args.parity_bits}-bit Parity (Møller 1993)"] = res

    if run_mnist:
        ep = args.epochs or default_epochs["mnist"]
        bs = args.batch_size or default_batch["mnist"]
        _, res = run_mnist(epochs=ep, batch_size=bs,
                           seed=args.seed, verbose=args.verbose)
        results["MNIST Forward-Forward (Hinton 2022)"] = res

    if len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
