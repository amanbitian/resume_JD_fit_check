from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import glob


OPTIONAL_COLUMNS = ("label", "source", "resume_domain", "jd_domain")


@dataclass(frozen=True, slots=True)
class PairRecord:
    pair_id: str
    source_file: str
    row_index: int
    resume: str
    job_description: str
    gold_label: str | None
    metadata: dict[str, Any]


def _pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised in a clean environment
        raise RuntimeError(
            "Reading parquet requires pyarrow. Install this project with: pip install -e ."
        ) from exc
    return pq


def resolve_parquet_files(value: str | Path) -> list[Path]:
    raw = str(value)
    candidate = Path(raw)
    if any(char in raw for char in "*?["):
        files = [Path(item) for item in glob.glob(raw)]
    elif candidate.is_dir():
        files = list(candidate.glob("*.parquet"))
    elif candidate.is_file():
        files = [candidate]
    else:
        raise FileNotFoundError(f"Dataset path does not exist: {value}")

    # macOS AppleDouble sidecars also end in .parquet but are not parquet files.
    files = sorted(path.resolve() for path in files if not path.name.startswith("._"))
    if not files:
        raise FileNotFoundError(f"No parquet files found at: {value}")
    return files


def iter_pairs(
    dataset: str | Path,
    *,
    resume_col: str = "resume",
    jd_col: str = "jd",
    label_col: str = "label",
    batch_size: int = 128,
    limit: int | None = None,
    offset: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
) -> Iterator[PairRecord]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if offset < 0 or (limit is not None and limit < 0):
        raise ValueError("offset and limit must be non-negative")

    pq = _pyarrow_parquet()
    emitted = 0
    global_index = 0
    for path in resolve_parquet_files(dataset):
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        missing = {resume_col, jd_col} - available
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
        columns = [resume_col, jd_col]
        columns.extend(col for col in (label_col, *OPTIONAL_COLUMNS[1:]) if col in available)
        columns = list(dict.fromkeys(columns))
        file_row = 0
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            data = batch.to_pydict()
            for local_index in range(batch.num_rows):
                selected = global_index >= offset and (global_index - offset) % num_shards == shard_index
                current_global_index = global_index
                global_index += 1
                current_file_row = file_row
                file_row += 1
                if not selected:
                    continue
                if limit is not None and emitted >= limit:
                    return
                resume = data[resume_col][local_index]
                jd = data[jd_col][local_index]
                if not isinstance(resume, str) or not isinstance(jd, str):
                    continue
                metadata = {
                    key: data[key][local_index]
                    for key in OPTIONAL_COLUMNS[1:]
                    if key in data
                }
                yield PairRecord(
                    pair_id=f"{path.name}:{current_file_row}",
                    source_file=str(path),
                    row_index=current_global_index,
                    resume=resume,
                    job_description=jd,
                    gold_label=data.get(label_col, [None] * batch.num_rows)[local_index],
                    metadata=metadata,
                )
                emitted += 1


def dataset_summary(dataset: str | Path) -> dict[str, Any]:
    pq = _pyarrow_parquet()
    files = resolve_parquet_files(dataset)
    rows = 0
    schemas: dict[str, list[str]] = {}
    for path in files:
        parquet = pq.ParquetFile(path)
        rows += parquet.metadata.num_rows
        schemas[path.name] = parquet.schema_arrow.names
    return {"files": [str(path) for path in files], "rows": rows, "schemas": schemas}

