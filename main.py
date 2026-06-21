"""Single command entry point for the World Cup 2026 predictor."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.features import build_feature_dataset
from src.model import train_and_evaluate
from src.player_features import load_team_player_features, save_team_player_features
from src.preprocessing import configure_logging, load_and_preprocess
from src.simulate import (
    generate_visualizations,
    run_monte_carlo,
    save_simulation_results,
)
from src.tournament import load_tournament_config

LOGGER = logging.getLogger(__name__)


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
    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument(
        "--use-player-features",
        action="store_true",
        help="Train and simulate with Kaggle-derived team player features.",
    )
    feature_group.add_argument(
        "--skip-player-features",
        action="store_true",
        help="Train and simulate with results and ELO features only.",
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
    tournament_config = load_tournament_config()
    use_player_features = bool(args.use_player_features)
    player_features = None
    player_feature_path = processed_dir / "team_player_features.csv"
    if use_player_features:
        if not player_feature_path.exists():
            save_team_player_features(raw_dir, processed_dir)
        player_features = load_team_player_features(player_feature_path)
        if player_features is None:
            LOGGER.warning(
                "Continuing with neutral player features because Kaggle data is absent"
            )

    print("2/5 Engineering leakage-safe features...")
    features = build_feature_dataset(results, elo, player_features=player_features)
    processed_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(processed_dir / "match_features.csv", index=False)
    print(f"Feature rows: {len(features):,}")

    print("3/5 Training and evaluating ensemble...")
    model, label_encoder, _ = train_and_evaluate(
        features,
        outputs_dir,
        visualizations_dir,
        use_player_features=use_player_features,
    )

    print(f"4/5 Running {args.simulations:,} Monte Carlo simulations...")
    summary, round_counts, context, path_counts = run_monte_carlo(
        model,
        label_encoder,
        results,
        elo,
        simulations=args.simulations,
        player_features=player_features,
        use_player_features=use_player_features,
        tournament_config=tournament_config,
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
        path_counts=path_counts,
    )
    print("Done. Check /visualizations and /outputs.")


if __name__ == "__main__":
    main()
