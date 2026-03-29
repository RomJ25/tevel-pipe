---
description: Add a new test case to the input data
---

Add new test data based on: $ARGUMENTS

Step 1: Edit `input/point_cube1.csv` to add new call records.

The format for IntelliItemContent (the transcript) must be:
```
א - <metadata> / <name for side A>
ב - <metadata> / <name for side B>
```

Example row to add:
```csv
c009,050-1111111,050-7777777,"א - זוהה / <NAME_A>
ב - זוהה / <NAME_B>",2024-01-20 10:00:00
```

Step 2: If the new person should be verified, also add to:
- `input/point_cube3.csv` — the entity record (id, title, nicknames)
- `input/point_cube4.csv` — the call-to-entity link (call_id, phone, original_id)

Step 3: If the new person should be in a phonebook, add to:
- `input/algo_cube2.csv` — the contact (phone_number, name, entity_id)

Step 4: Run the pipeline:
```bash
cd $PROJECT_ROOT
python run.py
```

Step 5: Check results:
```bash
python run.py inspect --name "<the new name>"
```
