#!/usr/bin/env python3
"""
pip-audit CI policy enforcement.

Exit behaviour:
  0 — all detected CVEs have no upstream fix (accepted residual risk);
      a warning table is written to the GitHub Actions step summary.
  1 — one or more CVEs have an available fix version; the PR must be
      remediated before merging.

This approach avoids hard-coding specific CVE IDs. When an upstream fix is
released for a previously no-fix CVE, the check automatically starts failing
without any workflow changes, prompting the dependency upgrade.

Future extension: add a CLASS_C_PACKAGES list. For safety-critical components
classified as IEC 62304 Class C, even no-fix CVEs may be unacceptable and
should cause exit 1 regardless of fix availability.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys


def _run_audit() -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", "pip_audit", "--format=json", "--progress-spinner=off"],
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"[ERROR] Could not parse pip-audit output:\n{proc.stdout[:500]}")
        sys.exit(1)


def _write_step_summary(unfixable: list[dict]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a") as f:
        f.write("## Accepted-Risk CVEs (no upstream fix available)\n\n")
        f.write("| Package | Version | CVE ID |\n")
        f.write("|---------|---------|--------|\n")
        for v in unfixable:
            f.write(f"| {v['package']} | {v['version']} | `{v['id']}` |\n")
        f.write(
            "\n> These vulnerabilities have no upstream fix version available. "
            "Risk is accepted per the SOUP register. "
            "This check will automatically fail when an upstream fix is released.\n"
        )


def main() -> None:
    data = _run_audit()
    fixable: list[dict] = []
    unfixable: list[dict] = []

    for dep in data.get("dependencies", []):
        for vuln in dep.get("vulns", []):
            entry = {
                "package": dep["name"],
                "version": dep["version"],
                "id": vuln["id"],
                "fix_versions": vuln.get("fix_versions", []),
            }
            (fixable if entry["fix_versions"] else unfixable).append(entry)

    if unfixable:
        print(f"[WARN] {len(unfixable)} CVE(s) with no available fix (accepted residual risk):")
        for v in unfixable:
            print(f"  {v['package']}=={v['version']}: {v['id']}")
        _write_step_summary(unfixable)

    if fixable:
        print(
            f"[FAIL] {len(fixable)} CVE(s) have an available fix version "
            "— remediate before merging:"
        )
        for v in fixable:
            print(f"  {v['package']}=={v['version']}: {v['id']} → fix: {v['fix_versions']}")
        sys.exit(1)

    print(
        f"[OK] pip-audit policy passed — "
        f"{len(unfixable)} accepted-risk CVE(s) (no upstream fix), 0 fixable CVEs."
    )


if __name__ == "__main__":
    main()
