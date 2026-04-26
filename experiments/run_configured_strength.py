import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.configured_eval_utils import load_json_config, run_configured_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_json_config(args.config)
    run_configured_experiment(
        config,
        include_quality=False,
        include_u_score=True,
    )


if __name__ == "__main__":
    main()
