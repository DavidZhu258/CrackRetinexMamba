#!/usr/bin/env python3
"""Validate the repository's seed gold evaluation cases.

This validator is intentionally dependency-free so it can run in GitHub Actions
for every A-core repository without installing the project's full ML/app stack.
It verifies that gold seed cases are concrete, traceable to tracked repo files,
and produce a GitHub-visible result artifact.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any

PLACEHOLDER_MARKERS = [
    "replace_with_real_fixture_or_record_id",
    "placeholder",
    "todo",
    "tbd",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            obj["_line_no"] = line_no
            cases.append(obj)
    return cases


def contains_placeholder(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False).lower()
    return any(marker in text for marker in PLACEHOLDER_MARKERS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(root: Path) -> dict[str, Any]:
    seed_path = root / "evals" / "gold" / "seed_gold.jsonl"
    if not seed_path.exists():
        raise SystemExit(f"Missing seed file: {seed_path}")
    cases = load_jsonl(seed_path)
    failures: list[str] = []
    file_refs: dict[str, dict[str, Any]] = {}
    hard_negative_count = 0
    for case in cases:
        cid = case.get("id", f"line-{case.get('_line_no')}")
        for field in ["id", "repo", "eval_type", "split", "case_type", "input", "expected_behavior", "metrics", "input_files", "expected_evidence_files"]:
            if field not in case:
                failures.append(f"{cid}: missing required field {field}")
        if contains_placeholder(case):
            failures.append(f"{cid}: contains placeholder/TODO marker")
        if case.get("hard_negative") is True or "negative" in str(case.get("case_type", "")).lower() or "unanswerable" in str(case.get("case_type", "")).lower() or "missing" in str(case.get("case_type", "")).lower():
            hard_negative_count += 1
        for key in ["input_files", "expected_evidence_files"]:
            for rel in case.get(key, []) or []:
                path = root / rel
                if not path.exists():
                    failures.append(f"{cid}: referenced file missing: {rel}")
                elif path.is_file() and rel not in file_refs:
                    file_refs[rel] = {
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
    if len(cases) < 3:
        failures.append(f"expected at least 3 seed cases, found {len(cases)}")
    if hard_negative_count < 1:
        failures.append("expected at least one hard-negative/safety/refusal seed case")
    status = "pass" if not failures else "fail"
    return {
        "status": status,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "repo": cases[0].get("repo") if cases else root.name,
        "seed_path": str(seed_path.relative_to(root)),
        "seed_case_count": len(cases),
        "hard_negative_count": hard_negative_count,
        "referenced_file_count": len(file_refs),
        "referenced_files": file_refs,
        "failures": failures,
        "note": "This validates gold seed traceability and feasibility scaffolding; it does not claim model-quality scores yet.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="evals/results/latest_seed_validation.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = validate(root)
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "seed_case_count": result["seed_case_count"], "hard_negative_count": result["hard_negative_count"], "result": args.out}, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
