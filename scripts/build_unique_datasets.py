"""Build de-duplicated JD and resume datasets from the evaluation parquet.

Each output has one row per unique document, keyed by the parquet's own `jd_id` / `resume_id`. Those IDs are
checked first: every ID must map to exactly one text and every text to exactly one ID, otherwise the script
stops rather than invent a mapping. Rows keep the order in which each document first appears.

Outputs (UTF-8 with BOM, so Excel reads the text correctly):
  Input/jd_dataset.csv      columns: jd_id, jd_text
  Input/resume_dataset.csv  columns: resume_id, resume_text
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import pyarrow.parquet as pq

DEFAULT_SOURCE = Path("Input/Candidate_JD_eval_159k.parquet")
OUT_DIR = Path("Input")


def unique_documents(table, id_col: str, text_col: str) -> list[tuple[str, str]]:
    ids = table[id_col].to_pylist()
    texts = table[text_col].to_pylist()
    id_to_text: dict[str, str] = {}
    text_to_id: dict[str, str] = {}
    for doc_id, text in zip(ids, texts):
        if doc_id is None or text is None or not text.strip():
            raise SystemExit(f"{id_col}/{text_col} has a null or empty value (id={doc_id!r}); fix the source first")
        if id_to_text.setdefault(doc_id, text) != text:
            raise SystemExit(f"{id_col} {doc_id} maps to more than one text; refusing to guess")
        if text_to_id.setdefault(text, doc_id) != doc_id:
            raise SystemExit(f"The same {text_col} appears under two IDs ({text_to_id[text]} and {doc_id}); refusing to guess")
    return list(id_to_text.items())


def write_csv(path: Path, header: list[str], rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    table = pq.read_table(args.source, columns=["jd_id", "jd_text", "resume_id", "resume_text"])
    jds = unique_documents(table, "jd_id", "jd_text")
    resumes = unique_documents(table, "resume_id", "resume_text")
    write_csv(args.out_dir / "jd_dataset.csv", ["jd_id", "jd_text"], jds)
    write_csv(args.out_dir / "resume_dataset.csv", ["resume_id", "resume_text"], resumes)
    print(f"source rows: {table.num_rows:,}")
    print(f"jd_dataset.csv: {len(jds):,} unique JDs")
    print(f"resume_dataset.csv: {len(resumes):,} unique resumes")


if __name__ == "__main__":
    main()
