---
description: Debug why a specific entity/name is resolved incorrectly
---

Debug the entity resolution for the name "$ARGUMENTS":

Step 1: See current results for this name
```bash
cd $PROJECT_ROOT
python run.py inspect --name "$ARGUMENTS"
```

Step 2: Check what point.py extracted
Read `output/point_output.csv` and filter for rows where `name_cleaned_A` or `name_cleaned_B` contains "$ARGUMENTS".

Step 3: Check what the algorithm received
Read `output/algo_input.csv` and filter for rows where `clean_name_A` or `clean_name_B` contains "$ARGUMENTS".

Step 4: Check algorithm scoring and merge decisions
Read `output/algo_log.txt` and search for "$ARGUMENTS". Look for:
- SCORING lines (similarity scores between mentions)
- MERGE DIAGNOSTIC (which clusters merged and why)
- PHASE2 decisions (skip reasons: dsu, veto, cohesion)

Step 5: Identify the issue
Common problems:
- Names scoring too low → check similarity thresholds in `algorithem_og.py` Config class
- Wrong entities merging → check constraint system (entity_id conflicts, must_not_link)
- Kunya (אבו-X) causing false merges → check kunya guards in cohesion gate
- BLMZ leaking into real entities → check BLMZ isolation
- Cross-phone not linking → check cross-phone thresholds (CROSS_PHONE_ANCHOR_SORT_THRESHOLD)

Step 6: Make a fix in `algorithem_og.py`, then re-run:
```bash
python run.py
```
Compare the new results with step 1.
