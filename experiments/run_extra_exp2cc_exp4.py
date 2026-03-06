import json
import os
import _jsonnet

from experiments.__main__ import run


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.jsonnet")
    configs = json.loads(_jsonnet.evaluate_file(config_path))
    configs = [
        c
        for c in configs
        if c.get("data_folder") == "extra_exp2cc_exp4_data"
        and c.get("task") == "summarization_scan_n"
    ]
    if not configs:
        raise SystemExit("No exp2cc exp4 configs found")
    for config in configs:
        run(config)


if __name__ == "__main__":
    main()
