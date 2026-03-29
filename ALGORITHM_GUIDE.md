# Algorithm Quick Guide

Condensed reference for `algorithem_og.py`. For full documentation see the source code.

## Pipeline Stages

| Stage | What it does |
|-------|-------------|
| 1. Schema Adaptation | Maps column names (point output → algorithm expected) |
| 2. Mention Explosion | Splits multi-speaker fields (using `+` delimiter) into individual mentions |
| 3. Vectorization | TF-IDF character n-grams (3-5) + token vectors for similarity |
| 4. Graph Building | Per-phone similarity graphs. Edge if similarity ≥ 0.70 |
| 5. Clustering | Hierarchical Agglomerative Clustering per phone |
| 6. Resolution | Priority: CALL_VERIFIED → PHONEBOOK → INFERRED |
| 6.25. Cube2 Bridge | Merge clusters on same phone via phonebook contacts |
| 6.5. Cluster Merge | Merge clusters with similar canonical names |
| 7. Cross-Phone Link | Link entities across different phones (global entity IDs) |
| 8. Entity Fusion | Create unified entity representatives |

## Key Thresholds (in Config class)

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `SIMILARITY_THRESHOLD` | 0.70 | Minimum similarity for per-phone edge creation |
| `CUBE2_MATCH_THRESHOLD` | 0.75 | Minimum score for phonebook match |
| `CUBE2_MARGIN_THRESHOLD` | 0.15 | Required gap over second-best phonebook match |
| `GLOBAL_MATCH_THRESHOLD` | 0.90 | Cross-phone match score (with entity_id) |
| `GLOBAL_MATCH_NO_ENTITY_THRESHOLD` | 0.95 | Cross-phone match score (no entity_id) |
| `CROSS_PHONE_ANCHOR_SORT_THRESHOLD` | 85 | Cross-phone threshold for verified anchors |
| `CROSS_PHONE_UNVERIFIED_SORT_THRESHOLD` | 92 | Cross-phone threshold for unverified clusters |

## Resolution Priority

1. **CALL_VERIFIED**: Has `verified_entity_id` from cube3/cube4 data. Always wins.
2. **PHONEBOOK**: Matched a contact in cube2 (phonebook). Needs HIGH quality tier.
3. **INFERRED**: No external verification. Name from call data only.

## Constraint System

- **BLMZ Isolation**: "בלמ״ז" (unknown speaker) NEVER merges with real names.
- **Entity ID Hard Link**: Same `verified_entity_id` → forced merge.
- **Entity ID Hard Block**: Different `verified_entity_id` → blocked from merging.
- **Must-Not-Link**: Speakers from same call (side A vs side B) cannot merge.
- **Kunya Guard**: Common kunyas (אבו-אחמד etc.) alone cannot bridge across phones.

## Similarity Scoring

Three components (weights must sum to 1.0):
- **Character n-grams** (0.40): TF-IDF on 3-5 character sequences
- **Token set similarity** (0.35): Fuzzy matching of token sets
- **IDF Jaccard** (0.25): Token overlap weighted by rarity

## Common Issues and Where to Look

| Problem | Where to look in algorithem_og.py |
|---------|----------------------------------|
| Names not clustering on same phone | `SIMILARITY_THRESHOLD` (too high?) — search for `SIMILARITY_THRESHOLD` |
| Wrong phonebook match | `CUBE2_MATCH_THRESHOLD`, `CUBE2_MARGIN_THRESHOLD` — search for `_cube2_is_confident` |
| Entities not linking across phones | `CROSS_PHONE_*_THRESHOLD` — search for `variant_aware_cluster_score` |
| Kunya causing false merge | Kunya guards — search for `kunya` or `cohesion_gate_passes` |
| BLMZ leaking into entities | BLMZ isolation — search for `blmz` or `is_blmz` |
| Entity ID not propagating | Entity ID logic — search for `verified_entity_id` in resolution functions |

## Key Functions

| Function | What it does |
|----------|-------------|
| `main(cubes)` | Entry point. Validates input, runs pipeline. |
| `EntityResolutionPipeline.run()` | Executes all stages in order. |
| `SimilarityScorer.compute()` | Computes similarity between two mentions. |
| `HacClusterer.cluster_with_constraints()` | Per-phone hierarchical clustering. |
| `EntityResolver._resolve_as_verified/phonebook/inferred()` | Assigns resolution type to a cluster. |
| `Cube2Matcher.match()` | Matches a cluster against phonebook contacts. |
| `variant_aware_cluster_score()` | Scores two clusters for cross-phone linking. |
| `cohesion_gate_passes()` | Prevents weak bridges between clusters. |

## Editing Tips

- All thresholds are in the `Config` dataclass near the top of the file
- Search for the threshold name to find where it's used
- After editing, run `python run.py` — changes are picked up automatically
- Compare before/after by looking at the RESULTS table in terminal output
