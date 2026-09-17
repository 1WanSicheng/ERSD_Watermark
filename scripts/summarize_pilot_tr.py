#!/usr/bin/env python3
"""Print MPFR-vs-INVARIANT TR/AATPS deltas per (cell, B) from runner outputs.

Usage: python scripts/summarize_pilot_tr.py outputs/pilot/*.json
"""
import json
import sys
from pathlib import Path


def main() -> None:
    print(f"{'file':<32s} {'B':>2s} {'MPFR TR':>8s} {'INV TR':>8s} {'dTR%':>7s} "
          f"{'MPFR AATPS':>10s} {'INV AATPS':>10s} {'ANLPPT-U':>8s}")
    for path in sys.argv[1:]:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        summary = d["summary"]
        cells = {}
        for key, agg in summary.items():
            dec, rest = key.rsplit("_L", 1)
            L, B = rest.split("_B")
            cells.setdefault((int(L), int(B)), {})[dec] = agg
        for (L, B), by_dec in sorted(cells.items()):
            m = by_dec.get("mpfr_torchgen_cached", {})
            i = by_dec.get("invariant_multi", {})
            m_tr, i_tr = m.get("token_rate"), i.get("token_rate")
            delta = (100.0 * (m_tr - i_tr) / i_tr) if (m_tr and i_tr) else float("nan")
            print(f"{Path(path).stem:<32s} {B:>2d} "
                  f"{m_tr or float('nan'):>8.2f} {i_tr or float('nan'):>8.2f} "
                  f"{delta:>+6.1f}% "
                  f"{m.get('AATPS', float('nan')):>10.3f} "
                  f"{i.get('AATPS', float('nan')):>10.3f} "
                  f"{m.get('ANLPPT_U', float('nan')):>8.3f}")


if __name__ == "__main__":
    main()
