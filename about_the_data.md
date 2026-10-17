# About the data

## Text length distribution (characters)

Each JD and resume is counted once, so these are unique texts: 18,530 JDs and 19,979 resumes.

| Statistic | JD | Resume |
|---|---:|---:|
| Min | 72 | 223 |
| 5th percentile | 438 | 276 |
| Median | 721 | 1,043 |
| 95th percentile | 785 | 6,129 |
| 99th percentile | 2,224 | 10,068 |
| Max | 8,171 | 29,084 |
| Mean | 680 | 1,543 |

**JD:** most JDs are short. 97.3% are under 1,000 characters, and only 0.3% are over 6,000.

**Resume:** the distribution is spread out.

- 27.0% are under 1,000 characters, which is short.
- 62.1% fall between 1,000 and 2,000 characters, which is the typical range.
- About 6.6% are over 4,000 characters. The top 1% are over 10,000.

## Resume-JD pairs (159K)

Length distribution (characters), 158,671 pairs.

| Statistic | JD | Resume | Pair (JD + Resume) |
|---|---:|---:|---:|
| Min | 72 | 223 | 659 |
| 5th percentile | 440 | 931 | 1,389 |
| Median | 1,153 | 4,379 | 6,263 |
| 95th percentile | 5,257 | 10,692 | 14,021 |
| 99th percentile | 7,368 | 15,341 | 18,772 |
| Max | 8,171 | 29,084 | 33,535 |
| Mean | 1,954 | 4,229 | 6,183 |

**Pair:** 2.5% are under 1,000 characters, 52.3% are over 6,000, and 17.3% are over 10,000.

These statistics are weighted by pair. Each JD and resume is counted once for every pair it appears in, so frequently reused texts count more. The 158,671 pairs are built from 18,530 unique JDs and 19,979 unique resumes.

## Data sources

The pair file `Input/Candidate_JD_eval_159k.parquet` combines three datasets. Each row records its origin in `source_dataset` and `source_file`.

| Dataset (`source_dataset`) | File (`source_file`) | Rows | Unique resumes | Unique JDs |
|---|---|---:|---:|---:|
| VaishnaviGudeats_resume_jd_matcher_dataset | `VaishnaviGudeats_resume_jd_matcher_dataset.csv` | 58,960 | 19,336 | 18,179 |
| resume-job-fit-merged-v1 | `train-00000-of-00002.parquet` | 40,009 | 642 | 280 |
| resume-job-fit-merged-v1 | `train-00001-of-00002.parquet` | 40,008 | 642 | 280 |
| resume-job-fit-merged-v1 | `test-00000-of-00001.parquet` | 12,712 | see below | see below |
| Hugging Face resume-jd-match | `train-00000-of-00001.parquet` | 6,227 | 642 | 280 |
| Hugging Face resume-jd-match | `test-00000-of-00001.parquet` | 755 | see below | see below |

The two `test` files share a filename, so the unique counts for them are combined: 477 resumes and 71 JDs across 13,467 rows.

The `source` column has a value only for the merged-v1 dataset: `generated_smart` (80,017 rows, the train shards) and `synthetic_test` (12,712 rows, the test file). The other two datasets have no `source` value. The `source` values suggest the merged-v1 data is generated or synthetic. Check that before drawing conclusions about real resumes.

Domains (`resume_domain`, `jd_domain`) are set only for the merged-v1 rows (92,729 rows).

- Resume domains: software 43,501; data 19,248; finance 18,364; engineering 4,398; management 2,781; design 1,413; sales 945; ai 766; healthcare 537; legal 341; hr 239; marketing 196.
- JD domains: software 44,349; finance 21,037; data 20,058; design 1,898; management 1,827; engineering 1,093; ai 823; hr 690; legal 434; sales 295; healthcare 180; marketing 45.

## Labels

The `label` column has two schemes. They don't map onto each other in the data, and the repo doesn't document the numeric one.

| Dataset | No Fit | Potential Fit | Good Fit | `0` | `1` |
|---|---:|---:|---:|---:|---:|
| merged-v1 (92,729) | 68,295 (73.6%) | 17,123 (18.5%) | 7,311 (7.9%) | | |
| Hugging Face (6,982) | 3,568 (51.1%) | 1,711 (24.5%) | 1,703 (24.4%) | | |
| Vaishnavi (58,960) | | | | 39,477 (67.0%) | 19,483 (33.0%) |

Percentages are shares of each dataset's own rows. The Vaishnavi rows use only `0` and `1`, and the other two datasets use only the named labels.

## Overlap between files

- The Vaishnavi file shares no resume or JD text with the merged-v1 or Hugging Face files.
- The two merged-v1 train shards and the Hugging Face train file all use the same 642 resumes and 280 JDs. Each of these 642 × 280 combinations is a candidate pair, and the shards sample from them.
- The test files are mostly built from the same resumes as the training data. 476 of the 477 test resumes also appear in the training files. None of the 71 test JDs appear in the training files.

The 159K evaluation file contains every one of these shards, including the training shards. Scoring this file therefore measures on training rows as well as test rows. The `README.md` warns against scoring the whole training directory because it includes the held-out test split, and this file is built from both. Confirm which rows a reported metric used.

## Notes

- The 99th percentile for the unique JDs is 2,222 when computed directly from `Input/jd_dataset.csv`, versus 2,224 in the table above. The difference comes from the percentile method.
- The JD line above says 0.3% of JDs are over 6,000 characters. Computed directly, that figure is 0.1% of unique JDs and 3.0% of pairs. Confirm which basis is intended before relying on it.
- The numeric `0` and `1` labels need mapping before evaluation.
