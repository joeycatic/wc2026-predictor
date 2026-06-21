"""Single command entry point for the World Cup 2026 predictor."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.features import build_feature_dataset
from src.model import train_and_evaluate
from src.preprocessing import configure_logging, load_and_preprocess
from src.simulate import (
    generate_visualizations,
    run_monte_carlo,
    save_simulation_results,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train and simulate FIFA World Cup 2026."
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=10_000,
        help="Number of Monte Carlo tournament simulations.",
    )
    return parser.parse_args()


def main() -> None:
    """Run preprocessing, feature engineering, training, simulation, and plotting."""
    configure_logging()
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    outputs_dir = project_root / "outputs"
    visualizations_dir = project_root / "visualizations"

    print("1/5 Loading and validating raw CSVs...")
    results, elo = load_and_preprocess(raw_dir)

    print("2/5 Engineering leakage-safe features...")
    features = build_feature_dataset(results, elo)
    processed_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(processed_dir / "match_features.csv", index=False)
    print(f"Feature rows: {len(features):,}")

    print("3/5 Training and evaluating ensemble...")
    model, label_encoder, _ = train_and_evaluate(
        features, outputs_dir, visualizations_dir
    )

    print(f"4/5 Running {args.simulations:,} Monte Carlo simulations...")
    summary, round_counts, context = run_monte_carlo(
        model, label_encoder, results, elo, simulations=args.simulations
    )
    save_simulation_results(summary, outputs_dir / "simulation_results.csv")

    print("Top 10 predicted finalists:")
    finalist_view = summary.sort_values("final_probability", ascending=False).head(10)
    for _, row in finalist_view.iterrows():
        print(f"  {row['team']}: {row['final_probability'] * 100:.2f}%")

    print("5/5 Generating visualizations...")
    generate_visualizations(
        model,
        label_encoder,
        summary,
        round_counts,
        context,
        visualizations_dir,
        simulations=args.simulations,
    )
    print("Done. Check /visualizations and /outputs.")


if __name__ == "__main__":
    main()
