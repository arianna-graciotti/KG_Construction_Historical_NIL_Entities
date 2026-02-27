#!/usr/bin/env python3
"""End-to-end pipeline test: Relink → Evaluation → LaTeX Tables.

Copies a small subset of real ZS data into /tmp/test_pipeline/, runs the
relink step (entity_linker.py), then runs evaluation (evaluate_all_properties.py),
and validates that all expected outputs are produced correctly.

Usage:
    python ObjectLinking/scripts/linking/test_pipeline.py
"""

import os
import sys
import shutil
import logging
import subprocess
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
_EVAL_DATA = _PROJECT_ROOT / "Evaluation" / "data"
_EVAL_SCRIPT = _PROJECT_ROOT / "Evaluation" / "scripts" / "evaluate_all_properties.py"

TEMP_ROOT = Path("/tmp/test_pipeline")

# Source files: one ZS file per (property, entity_type) combination.
# Each tuple is (property, entity_type, relative path under Evaluation/data/).
SOURCE_FILES = [
    ("CoC", "nil",
     "CoC/ZS/nil/QA_CoC_NIL_noctx_openrouter_google_gemma-2-27b-it_20250424_linked_evaluated.csv"),
    ("CoC", "qid",
     "CoC/ZS/qid/QA_CoC_QID_noctx_openrouter_google_gemma-2-27b-it_20250424_linked_evaluated.csv"),
    ("FamilyName", "nil",
     "FamilyName/ZS/nil/QA_FamilyName_NIL_noctx_openrouter_google_gemma-2-27b-it_20250507_linked_evaluated.csv"),
    ("FamilyName", "qid",
     "FamilyName/ZS/qid/QA_FamilyName_QID_noctx_openrouter_google_gemma-2-27b-it_20250507_linked_evaluated.csv"),
    ("GivenName", "nil",
     "GivenName/ZS/nil/QA_GivenName_nil_noctx_openrouter_google_gemma-2-27b-it_20250424_linked_evaluated.csv"),
    ("GivenName", "qid",
     "GivenName/ZS/qid/QA_GivenName_QID_noctx_openrouter_google_gemma-2-27b-it_20250424_linked_evaluated.csv"),
    ("sexGender", "nil",
     "sexGender/ZS/nil/QA_Gender_nil_noctx_openrouter_google_gemma-2-27b-it_20250424_linked_evaluated.csv"),
    ("sexGender", "qid",
     "sexGender/ZS/qid/QA_Gender_QID_noctx_openrouter_google_gemma-2-27b-it_20250424_linked_evaluated.csv"),
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_pipeline")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StepResult:
    """Track pass / fail / details for one test step."""

    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.details: list[str] = []

    def ok(self, msg: str = ""):
        self.passed = True
        if msg:
            self.details.append(msg)

    def fail(self, msg: str):
        self.passed = False
        self.details.append(f"FAIL: {msg}")


def print_summary(steps: list[StepResult]):
    """Print a coloured pass/fail summary."""
    print("\n" + "=" * 60)
    print("PIPELINE TEST SUMMARY")
    print("=" * 60)
    all_ok = True
    for s in steps:
        status = "PASS" if s.passed else "FAIL"
        marker = "\033[92m✓\033[0m" if s.passed else "\033[91m✗\033[0m"
        print(f"  {marker} [{status}] {s.name}")
        for d in s.details:
            print(f"         {d}")
        if not s.passed:
            all_ok = False
    print("=" * 60)
    if all_ok:
        print("\033[92mAll steps passed.\033[0m")
    else:
        print("\033[91mSome steps failed.\033[0m")
    print()
    return all_ok


# ===================================================================
# Step 1: Set up test data
# ===================================================================
def step_setup_data() -> StepResult:
    result = StepResult("Set up test data in /tmp/test_pipeline")

    # Clean previous run
    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT)

    copied = 0
    missing = []
    for prop, etype, relpath in SOURCE_FILES:
        src = _EVAL_DATA / relpath
        if not src.exists():
            missing.append(str(src))
            continue
        dst = TEMP_ROOT / "Evaluation" / "data" / relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    if missing:
        result.fail(f"Missing source files: {missing}")
        return result

    result.ok(f"Copied {copied} files")
    return result


# ===================================================================
# Step 2: Run relink
# ===================================================================
def step_run_relink() -> StepResult:
    result = StepResult("Run relink (entity_linker.py)")

    # We need entity_linker on sys.path
    sys.path.insert(0, str(_SCRIPT_DIR))
    try:
        from entity_linker import run_relink
    except ImportError as e:
        result.fail(f"Cannot import entity_linker: {e}")
        return result

    eval_data_dir = TEMP_ROOT / "Evaluation" / "data"
    rc = run_relink(eval_data_dir)
    if rc != 0:
        result.fail(f"run_relink() returned exit code {rc}")
        return result

    result.ok("run_relink() completed successfully")
    return result


# ===================================================================
# Step 3: Validate relink output
# ===================================================================
def step_validate_relink() -> StepResult:
    result = StepResult("Validate relink output")

    eval_data_dir = TEMP_ROOT / "Evaluation" / "data"
    csv_files = sorted(eval_data_dir.rglob("*_linked_evaluated.csv"))

    if not csv_files:
        result.fail("No CSV files found after relink")
        return result

    files_with_links = 0
    files_checked = 0

    for fpath in csv_files:
        df = pd.read_csv(fpath, low_memory=False)
        files_checked += 1

        if "linked_qid" not in df.columns:
            result.fail(f"No linked_qid column in {fpath.name}")
            return result

        non_empty = df["linked_qid"].dropna()
        non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
        if len(non_empty) > 0:
            files_with_links += 1

        # Also check: rows where llm_answer is empty should have linked_qid empty
        if "llm_answer" in df.columns:
            empty_answer = df["llm_answer"].isna() | (df["llm_answer"].astype(str).str.strip() == "")
            linked_for_empty = df.loc[empty_answer, "linked_qid"].dropna()
            linked_for_empty = linked_for_empty[linked_for_empty.astype(str).str.strip() != ""]
            if len(linked_for_empty) > 0:
                result.fail(
                    f"{fpath.name}: {len(linked_for_empty)} rows have linked_qid "
                    f"despite empty llm_answer"
                )
                return result

    if files_with_links == 0:
        result.fail("No files have any non-empty linked_qid values")
        return result

    result.ok(
        f"Checked {files_checked} files; "
        f"{files_with_links} have linked QIDs"
    )
    return result


# ===================================================================
# Step 4: Run evaluation
# ===================================================================
def step_run_evaluation() -> StepResult:
    result = StepResult("Run evaluation (evaluate_all_properties.py)")

    eval_data_dir = TEMP_ROOT / "Evaluation" / "data"
    output_dir = TEMP_ROOT / "evaluated"
    report_dir = TEMP_ROOT / "reports"
    tables_dir = TEMP_ROOT / "tables"

    cmd = [
        sys.executable,
        str(_EVAL_SCRIPT),
        "--folders", str(eval_data_dir),
        "--output", str(output_dir),
        "--report-output", str(report_dir),
        "--tables-output", str(tables_dir),
        "--verbose",
    ]

    log.info("Running: %s", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(_PROJECT_ROOT),
    )

    if proc.returncode != 0:
        result.fail(f"Exit code {proc.returncode}")
        if proc.stderr:
            # Show last 20 lines of stderr
            for line in proc.stderr.strip().splitlines()[-20:]:
                result.details.append(f"  stderr: {line}")
        return result

    result.ok("evaluate_all_properties.py completed successfully")
    return result


# ===================================================================
# Step 5: Validate evaluation output
# ===================================================================
def step_validate_evaluation() -> StepResult:
    result = StepResult("Validate evaluation output (reports + LaTeX)")

    report_dir = TEMP_ROOT / "reports"
    tables_dir = TEMP_ROOT / "tables"
    output_dir = TEMP_ROOT / "evaluated"

    # 5a: Check report CSVs
    expected_reports = []
    for etype in ("nil", "qid"):
        for metric in ("precision", "recall", "f1"):
            expected_reports.append(f"table_report_{etype}_{metric}.csv")

    missing_reports = []
    for rname in expected_reports:
        rpath = report_dir / rname
        if not rpath.exists():
            missing_reports.append(rname)

    if missing_reports:
        result.fail(f"Missing report files: {missing_reports}")
        return result

    # 5b: Check LaTeX tables
    expected_tex = ["evaluation_QID_rules.tex", "evaluation_NIL_rules.tex"]
    missing_tex = []
    for tname in expected_tex:
        tpath = tables_dir / tname
        if not tpath.exists():
            missing_tex.append(tname)

    if missing_tex:
        result.fail(f"Missing LaTeX files: {missing_tex}")
        return result

    # 5c: Check that at least some evaluated CSVs exist in the output dir
    evaluated_csvs = sorted(output_dir.rglob("*_evaluated.csv"))
    if not evaluated_csvs:
        result.fail("No evaluated CSVs found in output directory")
        return result

    # 5d: Spot-check one evaluated CSV for TP/FP/FN/TN columns
    sample = pd.read_csv(evaluated_csvs[0], low_memory=False)
    for col in ("TP", "FP", "FN", "TN"):
        if col not in sample.columns:
            result.fail(f"Column {col} missing from {evaluated_csvs[0].name}")
            return result

    result.ok(
        f"{len(expected_reports)} reports, "
        f"{len(expected_tex)} LaTeX files, "
        f"{len(evaluated_csvs)} evaluated CSVs"
    )
    return result


# ===================================================================
# Step 6: Validate true-negative handling
# ===================================================================
def step_validate_true_negatives() -> StepResult:
    result = StepResult("Validate true-negative handling")

    output_dir = TEMP_ROOT / "evaluated"
    evaluated_csvs = sorted(output_dir.rglob("*_evaluated.csv"))

    if not evaluated_csvs:
        result.fail("No evaluated CSVs to check")
        return result

    tn_found = 0
    files_checked = 0

    for fpath in evaluated_csvs:
        df = pd.read_csv(fpath, low_memory=False)
        files_checked += 1

        # Determine entity type from filename
        is_qid_file = "qid" in fpath.name.lower()

        if is_qid_file:
            # For QID files: rows where both linked_qid and qid_gold_true are
            # empty should have TN=1
            if "linked_qid" in df.columns and "qid_gold_true" in df.columns and "TN" in df.columns:
                linked_empty = df["linked_qid"].isna() | (df["linked_qid"].astype(str).str.strip() == "")
                gold_empty = df["qid_gold_true"].isna() | (df["qid_gold_true"].astype(str).str.strip() == "")
                both_empty = df[linked_empty & gold_empty]

                if len(both_empty) > 0:
                    tn_count = int(both_empty["TN"].sum())
                    tn_found += tn_count
                    # Check that all such rows actually got TN=1
                    non_tn = both_empty[both_empty["TN"] != 1]
                    if len(non_tn) > 0:
                        result.fail(
                            f"{fpath.name}: {len(non_tn)} rows with both "
                            f"linked_qid and qid_gold_true empty but TN != 1"
                        )
                        return result
        else:
            # For NIL files: rows where gold is "NIL" and linked_qid is
            # empty or "NIL" should be TN=1
            if "linked_qid" in df.columns and "qid_gold_true" in df.columns and "TN" in df.columns:
                gold_nil = df["qid_gold_true"].astype(str).str.strip().str.upper() == "NIL"
                linked_nil_or_empty = (
                    df["linked_qid"].isna()
                    | (df["linked_qid"].astype(str).str.strip() == "")
                    | (df["linked_qid"].astype(str).str.strip().str.upper() == "NIL")
                )
                both_nil = df[gold_nil & linked_nil_or_empty]

                if len(both_nil) > 0:
                    tn_count = int(both_nil["TN"].sum())
                    tn_found += tn_count
                    non_tn = both_nil[both_nil["TN"] != 1]
                    if len(non_tn) > 0:
                        result.fail(
                            f"{fpath.name}: {len(non_tn)} NIL rows expected "
                            f"TN=1 but got TN != 1"
                        )
                        return result

    result.ok(
        f"Checked {files_checked} files; "
        f"{tn_found} true-negative rows correctly marked"
    )
    return result


# ===================================================================
# Main
# ===================================================================
def main() -> int:
    log.info("Starting end-to-end pipeline test")
    log.info("Temp directory: %s", TEMP_ROOT)

    steps: list[StepResult] = []

    # Step 1
    r = step_setup_data()
    steps.append(r)
    if not r.passed:
        print_summary(steps)
        return 1

    # Step 2
    r = step_run_relink()
    steps.append(r)
    if not r.passed:
        print_summary(steps)
        return 1

    # Step 3
    r = step_validate_relink()
    steps.append(r)
    if not r.passed:
        print_summary(steps)
        return 1

    # Step 4
    r = step_run_evaluation()
    steps.append(r)
    if not r.passed:
        print_summary(steps)
        return 1

    # Step 5
    r = step_validate_evaluation()
    steps.append(r)
    if not r.passed:
        print_summary(steps)
        return 1

    # Step 6
    r = step_validate_true_negatives()
    steps.append(r)

    all_ok = print_summary(steps)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
