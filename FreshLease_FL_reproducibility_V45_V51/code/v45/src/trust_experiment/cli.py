from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .plotting import create_figures
from .runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="运行第一阶段节点历史信任实验")
    parser.add_argument("--config", default="config/default.yaml", help="YAML配置文件")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_dir = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    paths = run_experiment(config, project_dir)
    figures = create_figures(
        paths["rounds"], paths["summary"], paths["summary"].parent,
        dpi=int(config["reporting"]["figure_dpi"]),
        threshold_curve_path=paths["threshold_curve"],
    )
    print("Experiment completed")
    for name, path in paths.items():
        if path.exists():
            print(f"{name}: {path}")
    for path in figures:
        print(f"figure: {path}")


if __name__ == "__main__":
    main()
