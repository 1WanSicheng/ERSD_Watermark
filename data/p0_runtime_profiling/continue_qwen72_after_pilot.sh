#!/usr/bin/env bash
set -euo pipefail

pilot_pid=$1
python_bin=/data/wansicheng3/envdllm/bin/python
pilot_dir=/data/opc/mpfr_p0/qwen72_pilot
repo=/data/opc/mpfr_p0/repo

while kill -0 "$pilot_pid" 2>/dev/null; do
  sleep 30
done

"$python_bin" - "$pilot_dir" <<'PY'
import json
import sys
from pathlib import Path

pilot_dir = Path(sys.argv[1])
single = json.loads((pilot_dir / "single_n10.json").read_text())
multi = json.loads((pilot_dir / "multi_n10.json").read_text())

assert single["config"]["samples"] == 10
assert multi["config"]["samples"] == 10
assert len(single["results"]) == 3
assert len(multi["results"]) == 4
for payload in (single, multi):
    mapping = payload["config"]["target_device_map_actual"]
    assert set(mapping.values()) == {str(index) for index in range(8)}
    for result in payload["results"]:
        assert result["end_to_end"]["tokens"] > 0
        assert result["instrumented"]["tokens"] > 0
PY

exec "$repo/data/p0_runtime_profiling/run_qwen72_n100.sh"
