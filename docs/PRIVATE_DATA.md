# Private Data Handling

Some content is specific to the real deployment site and must stay out of the public repository.
It is split into separate files that Git ignores, and restored locally whenever you need it.

## What is private

| Item | Where it lives | In Git? |
|---|---|---|
| Real BSSIDs, AP labels, SSID keyword, site description | `config/access_points.private.json` | No (git-ignored) |
| Raw Wi-Fi scan CSVs | `data/raw/` | No |
| Intermediate datasets (they contain BSSIDs) | `data/processed/` | No |
| Trained model and scaler | `models/*.pkl` | No |
| Notebook with saved outputs and original local paths | `notebooks/private/` | No |
| Public placeholder AP table | `config/access_points.example.json` | Yes |

## Restoring the private files

1. Copy `access_points.private.json` into `config/`.
2. (Only to retrain) copy the raw CSVs into `data/raw/device_1 ... device_4/`.
3. Run `python scripts/check_environment.py`; it reports whether the private config was found.

`config/access_points.private.json` has the same shape as the example file:

```json
{
  "ssid_keyword": "campus-wifi",
  "site_description": "Example Building, Floor 2",
  "access_points": { "00:00:5E:00:53:01": "Hallway A (2.4 GHz)" }
}
```

If the private file is missing, the app and notebook fall back to the example file automatically, so a fresh
clone always starts (it just will not match your trained model).

## How leaks are prevented

1. `.gitignore` excludes every private location above.
2. `scripts/check_sensitive.py` scans all committable files for MAC addresses, personal OS paths, notebook
   outputs, and any value stored in your private config. Matches are never printed.
3. `.githooks/pre-commit` runs that scan before each commit (`git config core.hooksPath .githooks`).
4. GitHub Actions runs the same scan on every push and pull request.

## Before every push

```bash
python scripts/check_sensitive.py
```

Clear notebook outputs before committing (Jupyter: *Kernel > Restart Kernel and Clear Outputs*); the scan
fails on saved outputs. If you ever commit something sensitive by mistake, rewriting history
(`git filter-repo`) and rotating anything secret is required; deleting the file in a new commit is not enough.
