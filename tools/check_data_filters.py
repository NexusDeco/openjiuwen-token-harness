#!/usr/bin/env python3
"""Offline check (0 tokens) of the composable data_analysis questions in det_solver.solve_data.

Runs on the organisers' REAL sales.csv files from the sample tasks. The ground truth is computed here with the csv module
(no code shared with det_solver). Every combination of: metric (units / revenue = units * price), product (or none),
region (or none), period (none / one month / months A through B inclusive) is asked.

    python3 tools/check_data_filters.py <directory with the sample task dirs>

Exit status 1 on any WRONG answer or any question the solver should have recognised but sent to the LLM.
"""
from __future__ import annotations

import csv
import itertools
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import det_solver  # noqa: E402

BODY = ("The file sales.csv has columns order_id,region,product,month,units,price ({n} rows). {q} Use bash with python3 to "
        "compute it. Work only inside the current directory. When you are done, write ONLY the final answer to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = sys.argv[1]
    wss = [os.path.join(root, d, "workspace") for d in sorted(os.listdir(root))
           if d.startswith("data_analysis_") and os.path.isfile(os.path.join(root, d, "workspace", "sales.csv"))]
    asked = wrong = nomatch = 0
    bad = []
    for ws in wss:
        rows = list(csv.DictReader(open(os.path.join(ws, "sales.csv"), newline="")))
        products = sorted({r["product"] for r in rows}) + [None]
        regions = sorted({r["region"] for r in rows}) + [None]
        months = sorted({int(r["month"]) for r in rows})
        periods = [None] + [("m", m) for m in months] + [("r", a, b) for a, b in itertools.combinations(range(1, 13), 2)
                                                         if (a + b) % 3 == 0 or b - a == 1]
        for revenue, p, reg, per in itertools.product((False, True), products, regions, periods):
            head = "What is the total revenue (units * price)" if revenue else "How many units"
            q = head + (f" of product '{p}'" if p else "") + ("" if revenue else " were sold")
            q += f" in region '{reg}'" if reg else ""
            if per and per[0] == "m":
                q += f" in month {per[1]}"
            elif per:
                q += f" in months {per[1]} through {per[2]} inclusive"
            q += "?"
            lo, hi = (per[1], per[1]) if per and per[0] == "m" else ((per[1], per[2]) if per else (0, 99))
            exp = sum(int(r["units"]) * (int(r["price"]) if revenue else 1) for r in rows
                      if (not p or r["product"] == p) and (not reg or r["region"] == reg) and lo <= int(r["month"]) <= hi)
            res = det_solver.solve_data(det_solver._norm(BODY.format(n=len(rows), q=q)), ws)
            asked += 1
            if res.status == "nomatch":
                nomatch += 1
                bad.append(("NOMATCH", q))
            elif res.status != "answer" or res.answer != str(exp):
                wrong += 1
                bad.append(("WRONG", q, exp, res.status, res.answer))
    print(f"CSV files: {len(wss)}; questions: {asked}; correct: {asked - wrong - nomatch}; WRONG: {wrong}; to LLM: {nomatch}")
    for b in bad[:8]:
        print("  ", b)

    # safety: phrasings that must NOT be claimed, and a missing column must be reported as unsolvable, not guessed
    ws = wss[0]
    rows = list(csv.DictReader(open(os.path.join(ws, "sales.csv"), newline="")))
    must_not = [
        "How many units of product 'widget' were sold in months 7 through 8?",                      # no 'inclusive'
        "How many units of product 'widget' were sold in months 7 to 8 inclusive?",                 # other wording
        "How many units of product 'widget' were sold in months 7 through 8 inclusive in region 'east'?",  # order
        "What is the average revenue (units * price) of product 'widget'?",                          # other metric
        "How many orders of product 'widget' were sold?",                                            # other metric
    ]
    for q in must_not:
        res = det_solver.solve_data(det_solver._norm(BODY.format(n=len(rows), q=q)), ws)
        ok = res.status == "nomatch"
        print(f"must stay with the LLM: {q[:78]!r:82s} -> {res.status} {'ok' if ok else 'BAD'}")
        bad += [] if ok else [("claimed", q)]
    with tempfile.TemporaryDirectory() as tmp:  # sales.csv without the region column
        os.makedirs(tmp, exist_ok=True)
        cols = ["order_id", "product", "month", "units", "price"]
        with open(os.path.join(tmp, "sales.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r[c] for c in cols])
        q = "How many units of product 'widget' were sold in region 'east' in months 1 through 12 inclusive?"
        body = BODY.replace("order_id,region,product", "order_id,product").format(n=len(rows), q=q)
        res = det_solver.solve_data(det_solver._norm(body), tmp)
        ok = res.status == "unsolvable"
        print(f"missing 'region' column -> {res.status} {'ok' if ok else 'BAD'}")
        bad += [] if ok else [("missing-column", res.status)]
    print("RESULT:", "OK" if not bad else f"{len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
