"""Evaluation harness for Milestone II.

Runs the pipeline over labeled test cases and reports task success
(did the expected cause keyword appear in the kept diagnoses), plus
safety and sanity checks:
- knock rule: any detected knock must produce a safety warning
- sensor check: pegged/disconnected sensors must be flagged
- warmup check: coldstart logs must carry a warmup note, not a lean flag

Usage: python eval/evaluate.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tunesleuth import pipeline  # noqa: E402

CASES = json.loads((Path(__file__).parent / "test_cases.json").read_text())
DATA = Path(__file__).resolve().parents[1] / "data"


def run_case(case: dict) -> dict:
    csv_text = case.get("csv")
    if case.get("file"):
        csv_text = (DATA / case["file"]).read_text()
    result = pipeline.run(csv_text=csv_text, obd_code=case.get("obd_code"))

    if case["expect"] == "healthy":
        success = result.get("healthy", False)
    elif case["expect"] == "unusable":
        success = not result["ok"]
    else:
        text = json.dumps(result.get("diagnoses", [])).lower()
        text += " " + " ".join(result.get("anomalies", [])).lower()
        success = any(kw in text for kw in case["expect"])

    checks_ok = True
    if case.get("must_warn_knock"):
        checks_ok = checks_ok and bool(result.get("safety_warning"))
    if case.get("must_warn_sensor"):
        checks_ok = checks_ok and bool(result.get("sensor_warnings"))
    if case.get("must_note_warmup"):
        checks_ok = checks_ok and bool(result.get("warmup_note"))

    return {"name": case["name"], "group": case.get("group", "other"),
            "success": success, "checks": checks_ok}


def main():
    rows = [run_case(c) for c in CASES]
    print(f"{'case':52} {'success':8} {'checks'}")
    for r in rows:
        print(f"{r['name']:52} {str(r['success']):8} {r['checks']}")

    passed = sum(r["success"] for r in rows)
    checks_passed = sum(r["checks"] for r in rows)
    n = len(rows)
    print(f"\ntask success:          {passed}/{n} ({100 * passed / n:.0f}%)")
    print(f"safety/sanity checks:  {checks_passed}/{n} ({100 * checks_passed / n:.0f}%)")

    by_group = defaultdict(lambda: [0, 0])
    for r in rows:
        by_group[r["group"]][0] += r["success"]
        by_group[r["group"]][1] += 1
    print("by group: " + "   ".join(
        f"{g} {p}/{t}" for g, (p, t) in sorted(by_group.items())))

    failed = [r["name"] for r in rows if not (r["success"] and r["checks"])]
    if failed:
        print("\nFAILED: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
