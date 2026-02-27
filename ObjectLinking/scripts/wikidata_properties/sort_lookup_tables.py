#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sort lookup table CSVs by ascending numeric QID.

Reads each CSV in ObjectLinking/lookup_tables/, sorts rows so that the
lowest QID (e.g. Q31) comes first, and writes back in place.  This ensures
deterministic first-match-wins behaviour when multiple QIDs share a label
or alias: the most canonical (lowest-numbered) QID wins.
"""

import csv
import re
import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

LOOKUP_DIR = Path(__file__).resolve().parent.parent.parent / "lookup_tables"

CSV_FILES = [
    "extracted_country_of_citizenship.csv",
    "extracted_family_names.csv",
    "extracted_given_names.csv",
    "extracted_occupations.csv",
    "extracted_gender.csv",
]

_QID_RE = re.compile(r"^Q(\d+)$")


def qid_sort_key(row):
    """Extract the numeric part of a QID for sorting."""
    m = _QID_RE.match(row["QID"])
    if m:
        return int(m.group(1))
    return float("inf")


def sort_csv(path: Path):
    """Sort a single CSV file by numeric QID in place."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        logger.warning(f"  {path.name}: empty file, skipping")
        return

    first_qid_before = rows[0]["QID"]
    last_qid_before = rows[-1]["QID"]

    rows.sort(key=qid_sort_key)

    first_qid_after = rows[0]["QID"]
    last_qid_after = rows[-1]["QID"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["QID", "Label", "Aliases"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        f"  {path.name}: {len(rows)} rows  "
        f"before=[{first_qid_before}..{last_qid_before}]  "
        f"after=[{first_qid_after}..{last_qid_after}]"
    )


def main():
    logger.info(f"Sorting lookup tables in {LOOKUP_DIR}")
    for name in CSV_FILES:
        path = LOOKUP_DIR / name
        if not path.exists():
            logger.warning(f"  {name}: file not found, skipping")
            continue
        sort_csv(path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
