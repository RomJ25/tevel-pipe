# Pipeline Test System

## What this does

This pipeline processes **phone call records** to figure out **who spoke in each call**.

**Example**: Three calls mention "מחמד אלמסרי", "אבו-אחמד", and "אבו-אחמד מחמד" on different phone numbers. These are all the **same person** — "אבו-אחמד" is a kunya (Arabic nickname meaning "father of Ahmad") for מחמד אלמסרי. The pipeline figures this out and assigns them one entity ID.

**Stage 1** (`point.py`): Reads raw call transcripts and extracts the speaker names from each side of the call.
**Stage 2** (`algorithem_og.py`): Groups the extracted names into people (entities), linking them even across different phone numbers. Uses name similarity, verified data, and phonebook contacts.

### Key domain terms
- **Kunya** (אבו-X): Arabic nickname meaning "father of X". Very common — thousands of people share "אבו-אחמד". Weak identifier alone.
- **BLMZ** (בלמ״ז): Unidentified speaker. Always kept separate from real identities.
- **Side A / Side B**: The two parties in a phone call. Side A is usually the monitored number.

### PythonCube format
Both scripts are **PythonCube scripts**: `def main(cubes):` receives a dict of DataFrames, returns a DataFrame. The test runner (`run.py`) builds the `cubes` dict from the CSV files in `input/` and calls `main(cubes)`. Use `/fix-script` when editing them.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python run.py`

If anything is missing, `python run.py` will tell you exactly what's wrong.

## Quick start

```bash
python run.py
```

You will see output like this:

```
  STAGE 1: point.py (name extraction)
  Input:  8 calls
  Result: 8 rows (8 success, 0 failed)

  COLUMN MAPPING
  Renamed: original_call_id->call_id, num_A->pstn_A, ...

  STAGE 2: algorithem_og.py (entity resolution)
  Result: 9 CALL_VERIFIED, 4 INFERRED, 2 PHONEBOOK, 1 BLMZ

  RESULTS
  speaker_phone  global_entity_id  resolved_name      resolved_type   verified_entity_id  confidence
  0501111111     Eb39cb0a00653     מחמד אלמסרי         CALL_VERIFIED   E001                HIGH
  0502222222     Eb39cb0a00653     מחמד אלמסרי         CALL_VERIFIED   E001                HIGH
  0504444444     Ef26f3f62f9cc     אחמד כהן            PHONEBOOK       E004                HIGH
  0504444444     E_isolated_...    בלמ״ז               BLMZ                                LOW
```

## What the output means

- **CALL_VERIFIED**: Name confirmed by verified source data (cube3 + cube4 files). Most reliable.
- **PHONEBOOK**: Name matched from a phone's contact list (algo_cube2 file).
- **INFERRED**: Name seen in calls but not verified by any source. Least reliable.
- **BLMZ** (בלמ״ז): Unknown/unidentified speaker. Always isolated — never merged with real names.
- **global_entity_id**: Same ID = same person (even across different phones).
- **verified_entity_id**: ID from the verified data (cube3). Empty if not verified.

## Commands

| Command | What it does |
|---------|-------------|
| `python run.py` | Run full pipeline (both stages) |
| `python run.py point` | Run only name extraction |
| `python run.py algo` | Run only entity resolution (uses saved point output) |
| `python run.py inspect` | Show latest results table |
| `python run.py inspect --name "אבו"` | Filter results by name substring |
| `python run.py inspect --phone 5011` | Filter results by phone substring |

### Slash commands (type these in the Claude Code prompt for step-by-step guidance)

| Command | What it does |
|---------|-------------|
| `/run` | Run the full pipeline and show results |
| `/debug <name>` | Step-by-step guide to debug a specific entity |
| `/add-test <description>` | Step-by-step guide to add a new test case |
| `/fix-script` | PythonCube rules and code locations for editing the scripts |

## Files

| File/Folder | What it is |
|-------------|-----------|
| `run.py` | The runner script. **All logic is in this one file.** |
| `input/` | CSV input files go here |
| `output/` | Results appear here after running |
| `output/point_output.csv` | Extracted names (24 columns) |
| `output/algo_input.csv` | Intermediate file showing data sent to algorithm |
| `output/algo_output.csv` | Final resolved entities (25 columns) |
| `output/point_log.txt` | Debug log from point.py |
| `output/algo_log.txt` | Debug log from algorithm |

## Input CSV files (drop in input/)

How the input files connect:
- `point_cube1` — the raw call data (required)
- `point_cube5` — tells the pipeline which phone number is "ours" (required)
- `point_cube3` — a directory of known people with names, nicknames, IDs
- `point_cube4` — links specific calls to specific people in cube3
- `algo_cube2` — a separate phonebook (contacts saved on phones)

### point_cube1.csv (REQUIRED) — call transcript data

Columns: `call_id, pstn_A, pstn_B, IntelliItemContent, IntelliItemCreationTime`

- `pstn_A`: phone number of side A (usually the main/monitored number)
- `pstn_B`: phone number of side B (the other party)
- `IntelliItemContent`: the transcript text. Must contain lines with side markers:

```
א - זוהה / מחמד אלמסרי
ב - זוהה / אבו-אחמד
```

The `א -` marks side A's name, `ב -` marks side B's name. The `/` separates metadata from the name.

### point_cube5.csv (REQUIRED) — config

Columns: `SHLUHA`

One row with the main phone number (e.g., `050-1111111`). This tells the pipeline which phone is "side A".

### point_cube3.csv (optional) — verified entity data

Columns: `~apak_id, id, title, nicknames, status, id_number`

Known people with their verified names, nicknames (like kunya "אבו-X"), and entity IDs.

### point_cube4.csv (optional) — links calls to entities

Columns: `call_id, phone, original_id`

Maps call_id + phone to `~apak_id` in cube3, telling the pipeline which known person was in which call.

### algo_cube2.csv (optional) — phone contact list

Columns: `phone_number, ~parent_person_title, ~parent_person_nicknames, ~parent_person_id, status, id_number`

Contacts saved on phones. Used by the algorithm to match names against known contacts.

## Typical workflow

1. User asks: "Why is אבו-אחמד getting merged with the wrong entity?"
2. Run `python run.py inspect --name "אבו-אחמד"` to see the current result
3. Check `output/algo_input.csv` to see what data the algorithm received
4. Check `output/algo_log.txt` for scoring/merge details
5. Edit `algorithem_og.py` in this folder (see `ALGORITHM_GUIDE.md` for where to look)
6. Run `python run.py` again to see if the fix worked
7. Compare the new results with step 2

## Pipeline scripts (in this folder)

- **point.py** — Stage 1: name extraction (1963 lines). Edit directly.
- **algorithem_og.py** — Stage 2: entity resolution (8800+ lines). Edit directly.
- **ALGORITHM_GUIDE.md** — condensed reference: pipeline stages, thresholds, constraint system, key functions.

After editing, run `python run.py` to see the effect. Changes are picked up automatically.

**To understand the algorithm before editing**: read `ALGORITHM_GUIDE.md`. It explains what each stage does, which thresholds control what, and where to look in the code for common issues.

## Troubleshooting

| Error | Fix |
|-------|-----|
| "Cannot find input/point_cube1.csv" | Put your CSV files in the `input/` folder |
| "ERROR from point.py" | Check `output/point_log.txt` for details |
| "ERROR from algorithm" | Check `output/algo_log.txt` for details |
| "No algorithm input found" | Run `python run.py` (full) before `python run.py algo` |
| All extractions "failed" | Your IntelliItemContent needs `א -` and `ב -` side markers |
| Names look wrong | Check `output/point_output.csv` to see what point.py extracted |
| Entities not linking | Check `output/algo_input.csv` to see what algorithm received |
