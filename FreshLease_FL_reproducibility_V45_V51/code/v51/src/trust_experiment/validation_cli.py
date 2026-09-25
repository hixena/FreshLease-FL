from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .validation_suite import run_validation_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="运行四组增强Dirichlet验证实验")
    parser.add_argument(
        "--config", default="config/validation_suite.yaml", help="YAML配置文件"
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    project_dir = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = run_validation_suite(config, project_dir)
    print("Validation suite completed")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
