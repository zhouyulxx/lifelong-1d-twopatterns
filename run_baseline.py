"""Run a reproducible sequential A -> B MLP experiment on original UCR 1D data."""
import argparse
from continual_1d.experiment import run

def main():
    p = argparse.ArgumentParser(description="TwoPatterns A->B catastrophic forgetting baseline")
    p.add_argument("--epochs-a", type=int, default=20)
    p.add_argument("--epochs-b", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="results/baseline_seed42")
    a = p.parse_args()
    run(output_dir=a.output_dir, data_dir=a.data_dir, seed=a.seed,
        epochs_a=a.epochs_a, epochs_b=a.epochs_b, lr=a.lr,
        batch_size=a.batch_size, hidden_dim=a.hidden_dim, device=a.device)

if __name__ == "__main__":
    main()
