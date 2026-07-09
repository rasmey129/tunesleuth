"""Evaluation harness for Milestone II.

Runs the pipeline over labeled test cases and reports task success
(did the expected cause keyword appear in the kept diagnoses), plus
basic safety checks (knock always produces a warning).

Usage: python eval/evaluate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tunesleuth import pipeline  # noqa: E402

CASES = json.loads((Path(__file__).parent / "test_cases.json").read_text())
DATA = Path(__file__).resolve().parents[1] / "data"


def run_case(case: dict) -> dict:
    csv_text = None
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

    knock_ok = True
    if case.get("must_warn_knock"):
        knock_ok = bool(result.get("safety_warning"))

    return {"name": case["name"], "success": success, "knock_rule": knock_ok}


def main():
    rows = [run_case(c) for c in CASES]
    passed = sum(r["success"] for r in rows)
    knock_passed = sum(r["knock_rule"] for r in rows)
    print(f"{'case':40} {'success':8} {'knock rule'}")
    for r in rows:
        print(f"{r['name']:40} {str(r['success']):8} {r['knock_rule']}")
    print(f"\ntask success: {passed}/{len(rows)}   knock rule held: {knock_passed}/{len(rows)}")


if __name__ == "__main__":
    main()
