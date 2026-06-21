"""Download supported Kaggle datasets into data/raw."""

from __future__ import annotations

import argparse
from pathlib import Path

DATASETS = {
    "fifa_players": "stefanoleone992/fifa-23-complete-player-dataset",
    "wc2022_player_stats": "rhugvedbhojane/fifa-world-cup-2022-players-statistics",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Kaggle datasets.")
    parser.add_argument(
        "--datasets",
        choices=[*DATASETS.keys(), "all"],
        default="all",
        help="Dataset key to download.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
        help="Destination raw data directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import kaggle
    except ImportError as exc:
        raise SystemExit(
            "Install the kaggle package and configure ~/.kaggle/kaggle.json first."
        ) from exc

    selected = (
        DATASETS if args.datasets == "all" else {args.datasets: DATASETS[args.datasets]}
    )
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    for key, dataset in selected.items():
        destination = args.raw_dir / key
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {dataset} -> {destination}")
        kaggle.api.dataset_download_files(
            dataset,
            path=str(destination),
            unzip=True,
            quiet=False,
        )


if __name__ == "__main__":
    main()
