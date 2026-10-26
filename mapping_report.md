# MedalCare-XL Mapping Report
Generated: 2026-10-26 03:20

## Summary
- Total ECGs indexed: **16842 / 16900** (99.7%)
- Minimum expected group/parameter coverage: **95.7%** [PASS]
- Anatomical model split disjointness: **PASS**
- **Exp 1 verdict: PASS**
- Groups with no indexed records: **None**
- ECG rows without a model ID: **0**

## Indexed versus expected ECGs per group
| group   |   indexed |   expected |   pct |
|:--------|----------:|-----------:|------:|
| avblock |      1300 |       1300 | 100.0 |
| fam     |      1300 |       1300 | 100.0 |
| iab     |      1244 |       1300 |  95.7 |
| lae     |      1300 |       1300 | 100.0 |
| lbbb    |      1300 |       1300 | 100.0 |
| mi      |      7800 |       7800 | 100.0 |
| rbbb    |      1298 |       1300 |  99.8 |
| sinus   |      1300 |       1300 | 100.0 |

## Indexed versus expected ECGs per folder
| folder          |   indexed |   expected |   missing |
|:----------------|----------:|-----------:|----------:|
| sinus           |      1300 |       1300 |         0 |
| lbbb            |      1300 |       1300 |         0 |
| rbbb            |      1298 |       1300 |         2 |
| avblock         |      1300 |       1300 |         0 |
| fam             |      1300 |       1300 |         0 |
| iab             |      1244 |       1300 |        56 |
| lae             |      1300 |       1300 |         0 |
| mi/LAD_0.3      |      1300 |       1300 |         0 |
| mi/LAD_1.0      |      1300 |       1300 |         0 |
| mi/LCX_0.3_ant  |       700 |        700 |         0 |
| mi/LCX_0.3_post |       650 |        650 |         0 |
| mi/LCX_1.0_ant  |       600 |        600 |         0 |
| mi/LCX_1.0_post |       650 |        650 |         0 |
| mi/RCA_0.3      |      1300 |       1300 |         0 |
| mi/RCA_1.0      |      1300 |       1300 |         0 |

## ECGs per group and split
| group   |   train |   validation |   test |
|:--------|--------:|-------------:|-------:|
| avblock |     900 |          200 |    200 |
| fam     |    1040 |          130 |    130 |
| iab     |     994 |          124 |    126 |
| lae     |    1040 |          130 |    130 |
| lbbb    |     900 |          200 |    200 |
| mi      |    5350 |         1250 |   1200 |
| rbbb    |     898 |          200 |    200 |
| sinus   |     900 |          200 |    200 |

## Parameter coverage
| param             |   valid |   expected |     pct |
|:------------------|--------:|-----------:|--------:|
| v_APD.min         |   16842 |      16900 | 99.6568 |
| v_APD.max         |   16842 |      16900 | 99.6568 |
| a_cv_t.BulkTissue |   16842 |      16900 | 99.6568 |
| a_fibperc         |   16842 |      16900 | 99.6568 |

## Range check issues
None – all parameter ranges match paper values.

## Anatomical model IDs
- Model ID: `group::source_shape_id`, as selected for this analysis; source shape IDs are read from `info1` (LAE uses its `pwave_NNN` token).
- Unique group-scoped model IDs: **144**
- Model IDs represented in multiple groups (group count -> ID count): **{1: 144}**
- Model IDs appearing in multiple splits: **0**
- Raw `info1` shape tokens: **143 unique; group count -> token count: {1: 142, 2: 1}**
- Directory `run_id` values retained for sensitivity analysis: **13 unique; 2 appear in multiple splits**.
- Directory run IDs spanning splits: **{'run_S64': 2, 'run_S67': 2}**
All indexed model IDs appear in exactly one split.

## Rows not indexed against expected counts
| folder   |   missing |
|:---------|----------:|
| rbbb     |         2 |
| iab      |        56 |
