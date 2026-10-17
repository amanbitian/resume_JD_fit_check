import pyarrow as pa
import pyarrow.parquet as pq

from candidate_fit.dataset import iter_pairs, resolve_parquet_files


def test_streaming_and_sharding(tmp_path):
    table = pa.table(
        {
            "resume": ["r0", "r1", "r2", "r3"],
            "jd": ["j0", "j1", "j2", "j3"],
            "label": ["No Fit", "Potential Fit", "Good Fit", "No Fit"],
            "source": ["test"] * 4,
        }
    )
    path = tmp_path / "train.parquet"
    pq.write_table(table, path)
    (tmp_path / "._train.parquet").write_bytes(b"not parquet")
    assert resolve_parquet_files(tmp_path) == [path.resolve()]
    records = list(iter_pairs(tmp_path, num_shards=2, shard_index=1, batch_size=2))
    assert [row.resume for row in records] == ["r1", "r3"]
    assert [row.pair_id for row in records] == ["train.parquet:1", "train.parquet:3"]

