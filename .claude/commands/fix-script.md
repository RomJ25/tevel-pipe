---
description: Fix or modify point.py or algorithem_og.py (PythonCube scripts)
---

Both `point.py` and `algorithem_og.py` are **PythonCube scripts**. When editing them, follow these rules:

## PythonCube contract (MUST follow)

1. Entry point: `def main(cubes):` — cubes is a dict of DataFrames
2. Always return a `pandas.DataFrame`, including on error
3. End file with: `if 'cubes' in globals(): result = main(cubes)`
4. Keep standalone — no cross-file imports, no filesystem access
5. Include `sanitize_dataframe()` and `create_error_df()` helpers
6. Use `cubes.get('cube1', pd.DataFrame())` to read input cubes

## How the scripts use cubes

### point.py
```python
cubes['cube1']  # Raw call data (required)
cubes['cube5']  # Config with main phone number (required)
cubes['cube3']  # Entity directory (optional)
cubes['cube4']  # Call-entity links (optional)
```
Returns: 24-column DataFrame with extracted names

### algorithem_og.py
```python
cubes['cube1']  # Call data from point.py output (required)
cubes['cube2']  # Phonebook contacts (optional)
```
Returns: 25-column DataFrame with resolved entities

## After editing

Run the test pipeline to verify your changes:
```bash
python run.py
```

Check results:
```bash
python run.py inspect
```

## Key locations in algorithem_og.py

- Config class (thresholds): search for `class Config`
- Similarity scoring: search for `class SimilarityScorer`
- Phonebook matching: search for `class Cube2Matcher`
- Cross-phone linking: search for `global_cluster_merge`
- Resolution cascade: search for `_resolve_cluster`
- Constraint system: search for `must_not_link` or `cohesion_gate`
