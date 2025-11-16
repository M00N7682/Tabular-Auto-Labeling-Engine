"""Command-line entry point for running the TabDDPM auto-labeling pipeline."""

from __future__ import annotations

from src.pipeline import run_full_pipeline


def main() -> None:
    metrics = run_full_pipeline()
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
