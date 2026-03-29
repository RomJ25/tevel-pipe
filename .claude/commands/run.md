---
description: Run the full pipeline (point.py → algorithm) and show results
---

Run the full pipeline and show the results:

```bash
cd $PROJECT_ROOT
python run.py
```

After running, check the output:
- `output/point_output.csv` — names extracted by point.py (24 columns)
- `output/algo_input.csv` — data sent to algorithm after column mapping (16 columns)
- `output/algo_output.csv` — final entity resolution results (25 columns)
- `output/point_log.txt` — debug log from point.py
- `output/algo_log.txt` — debug log from algorithm

If something looks wrong, read the log files for details.
