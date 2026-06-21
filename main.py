"""Single command entry point for the World Cup 2026 predictor."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.backtest import run_world_cup_backtest
from src.features import build_feature_dataset, get_feature_columns
from src.model import load_cached_model, train_and_evaluate
from src.optional_data import build_data_status, optional_source_status
from src.player_features import load_team_player_features, save_team_player_features
from src.preprocessing import configure_logging, load_and_preprocess
from src.simulate import (
    build_group_most_likely_tables,
    generate_visualizations,
    load_path_counts,
    load_round_counts,
    run_monte_carlo,
    save_bracket_slot_probabilities,
    save_path_counts,
    save_round_counts,
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible Monte Carlo simulations.",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Load cached model artifacts from outputs instead of training.",
    )
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Regenerate PNGs from saved simulation CSV/JSON artifacts.",
    )
    parser.add_argument(
        "--data-status",
        action="store_true",
        help="Print input-data coverage and exit.",
    )
    parser.add_argument(
        "--scoreline-model",
        choices=("poisson", "legacy"),
        default="poisson",
        help="Scoreline sampler used by tournament simulations.",
    )
    feature_group = parser.add_mutually_exclusive_group()
    feature_group.add_argument(
        "--use-player-features",
        dest="use_player_features",
        action="store_true",
        help="Train and simulate with Kaggle-derived team player features.",
    )
    feature_group.add_argument(
        "--skip-player-features",
        dest="use_player_features",
        action="store_false",
        help="Train and simulate with results and ELO features only.",
    )
    parser.set_defaults(use_player_features=None)
    return parser.parse_args()


def _load_player_features(
    raw_dir: Path,
    processed_dir: Path,
    use_player_features: bool,
) -> pd.DataFrame | None:
    player_feature_path = processed_dir / "team_player_features.csv"
    if not use_player_features:
        return None
    if not player_feature_path.exists():
        save_team_player_features(raw_dir, processed_dir)
    player_features = load_team_player_features(player_feature_path)
    if player_features is None:
        LOGGER.warning(
            "Continuing with neutral player features because Kaggle data is absent"
        )
        print("Player features: neutral fallback values are being used.")
    return player_features


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def _infer_cached_feature_flag(outputs_dir: Path) -> bool:
    metrics_path = outputs_dir / "metrics.json"
    if not metrics_path.exists():
        return False
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return metrics.get("feature_set") == "player_enhanced"


def _run_visualize_only(
    outputs_dir: Path,
    visualizations_dir: Path,
    results: pd.DataFrame,
    elo: pd.DataFrame,
    player_features: pd.DataFrame | None,
    use_player_features: bool,
    tournament_config,
) -> None:
    model, label_encoder = load_cached_model(outputs_dir)
    summary = _read_required_csv(outputs_dir / "simulation_results.csv")
    group_stage_summary = _read_required_csv(outputs_dir / "group_stage_predictions.csv")
    round_counts = load_round_counts(outputs_dir / "round_counts.json")
    path_counts = load_path_counts(outputs_dir / "path_counts.json")
    simulations = int(sum(round_counts["Champion"].values()))
    if simulations <= 0:
        raise ValueError("Saved round_counts.json has no champion simulations")
    from src.simulate import build_prediction_context

    context = build_prediction_context(
        results,
        elo,
        player_features=player_features,
        use_player_features=use_player_features,
        tournament_config=tournament_config,
    )
    generate_visualizations(
        model,
        label_encoder,
        summary,
        round_counts,
        context,
        group_stage_summary,
        visualizations_dir,
        simulations=simulations,
        path_counts=path_counts,
    )


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
    if args.use_player_features is None and (args.no_train or args.visualize_only):
        use_player_features = _infer_cached_feature_flag(outputs_dir)
    else:
        use_player_features = bool(args.use_player_features)
    player_features = _load_player_features(raw_dir, processed_dir, use_player_features)
    data_status = build_data_status(
        results,
        elo,
        tournament_config,
        raw_dir,
        player_features,
    )
    if args.data_status:
        print(json.dumps(data_status, indent=2))
        return

    if args.visualize_only:
        print("Regenerating visualizations from saved artifacts...")
        _run_visualize_only(
            outputs_dir,
            visualizations_dir,
            results,
            elo,
            player_features,
            use_player_features,
            tournament_config,
        )
        print("Done. Check /visualizations.")
        return

    if args.no_train:
        print("2/5 Loading cached model artifacts...")
        model, label_encoder = load_cached_model(outputs_dir)
        features = None
    else:
        print("2/5 Engineering leakage-safe features...")
        features = build_feature_dataset(results, elo, player_features=player_features)
        processed_dir.mkdir(parents=True, exist_ok=True)
        features.to_csv(processed_dir / "match_features.csv", index=False)
        print(f"Feature rows: {len(features):,}")

        print("3/5 Training, calibrating, and evaluating ensemble...")
        model, label_encoder, _ = train_and_evaluate(
            features,
            outputs_dir,
            visualizations_dir,
            use_player_features=use_player_features,
            player_feature_status=data_status["player_feature_coverage"],
            optional_source_status=optional_source_status(raw_dir, tournament_config),
        )
        run_world_cup_backtest(
            model,
            label_encoder,
            features,
            get_feature_columns(use_player_features),
            outputs_dir,
            visualizations_dir,
        )

    print(f"4/5 Running {args.simulations:,} Monte Carlo simulations...")
    summary, round_counts, context, group_stage_summary, path_counts = run_monte_carlo(
        model,
        label_encoder,
        results,
        elo,
        simulations=args.simulations,
        player_features=player_features,
        use_player_features=use_player_features,
        tournament_config=tournament_config,
        seed=args.seed,
        scoreline_model=args.scoreline_model,
    )
    save_simulation_results(summary, outputs_dir / "simulation_results.csv")
    save_simulation_results(
        group_stage_summary,
        outputs_dir / "group_stage_predictions.csv",
    )
    save_simulation_results(
        build_group_most_likely_tables(group_stage_summary),
        outputs_dir / "group_most_likely_tables.csv",
    )
    save_bracket_slot_probabilities(
        path_counts,
        round_counts,
        args.simulations,
        tournament_config,
        outputs_dir / "bracket_slot_probabilities.csv",
    )
    save_round_counts(round_counts, outputs_dir / "round_counts.json")
    save_path_counts(path_counts, outputs_dir / "path_counts.json")

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
        group_stage_summary,
        visualizations_dir,
        simulations=args.simulations,
        path_counts=path_counts,
    )
    print("Done. Check /visualizations and /outputs.")


if __name__ == "__main__":
    main()
