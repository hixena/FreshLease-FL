from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .robustness_validation import run_robustness_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="运行强基线与复杂攻击验证")
    parser.add_argument("--config", default="config/robustness.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    project_dir = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = run_robustness_validation(config, project_dir)
    print("Robustness validation completed")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
