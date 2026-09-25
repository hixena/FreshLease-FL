from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .sensitivity_validation import run_sensitivity_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="运行受控观察期参数敏感性实验")
    parser.add_argument("--config", default="config/sensitivity.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    project_dir = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = run_sensitivity_validation(config, project_dir)
    print("Sensitivity validation completed")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
