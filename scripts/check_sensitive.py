"""Block sensitive content from being committed or pushed.

Scans every file that git could commit (tracked + untracked-but-not-ignored) for:
  * MAC addresses / BSSIDs (except the RFC 7042 documentation range used by the example config)
  * personal OS paths such as C:\\Users\\<name>\\... or /Users/<name>/...
  * notebook cells that still contain saved outputs (outputs may embed data or paths)
  * every value stored in config/access_points.private.json (SSID keyword, BSSIDs, labels, site description),
    when that private file exists on this machine

Usage (from repository root):
    python scripts/check_sensitive.py

Exit code 0 = clean, 1 = findings. Matches are never printed, only the file, line and category.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIVATE_CONFIG = ROOT / "config" / "access_points.private.json"

MAC_RE = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f])")
DOC_MAC_PREFIX = "00:00:5e:00:53:"  # RFC 7042 documentation range, safe to publish
OS_PATH_RE = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+|/Users/[^/\s]+/|/home/[^/\s]+/")
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".ruff_cache", ".ipynb_checkpoints"}


def candidate_files():
    """Files that could end up in a commit."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        return [ROOT / p for p in out if (ROOT / p).is_file()]
    except Exception:
        # Not a git repository yet: walk the tree but never look at known-private locations.
        files = []
        for p in ROOT.rglob("*"):
            rel = p.relative_to(ROOT).parts
            if not p.is_file() or SKIP_DIRS & set(rel):
                continue
            if rel[0] in {"data", "models"} and p.name != ".gitkeep":
                continue
            if "private" in p.name or "private" in rel[:-1]:
                continue
            files.append(p)
        return files


def private_terms():
    """Values from the private config that must never appear in committed files."""
    if not PRIVATE_CONFIG.exists():
        return []
    cfg = json.loads(PRIVATE_CONFIG.read_text(encoding="utf-8"))
    terms = {cfg.get("ssid_keyword", ""), cfg.get("site_description", "")}
    for bssid, label in cfg.get("access_points", {}).items():
        terms.update({bssid.lower(), bssid.lower().replace(":", "-"), label})
    return sorted(t.lower() for t in terms if t and len(t) >= 4)


def main() -> int:
    terms = private_terms()
    findings = []

    for path in candidate_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel == "scripts/check_sensitive.py":
            continue  # this file necessarily contains the detection patterns
        if rel == "config/access_points.private.json":
            findings.append((rel, 0, "private config is not ignored by git"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        if path.suffix == ".ipynb":
            try:
                nb = json.loads(text)
                for idx, cell in enumerate(nb.get("cells", [])):
                    if cell.get("cell_type") == "code" and cell.get("outputs"):
                        findings.append((rel, idx, "notebook cell has saved outputs (clear them)"))
            except json.JSONDecodeError:
                findings.append((rel, 0, "invalid notebook JSON"))

        for lineno, line in enumerate(text.splitlines(), start=1):
            for mac in MAC_RE.findall(line):
                if not mac.lower().replace("-", ":").startswith(DOC_MAC_PREFIX):
                    findings.append((rel, lineno, "MAC address / BSSID"))
            if OS_PATH_RE.search(line):
                findings.append((rel, lineno, "personal OS path"))
            low = line.lower()
            for term in terms:
                if term in low:
                    findings.append((rel, lineno, "value from private config"))
                    break

    if findings:
        print("Sensitive content found - do NOT commit or push:\n")
        for rel, lineno, kind in sorted(set(findings)):
            print(f"  {rel}:{lineno}  [{kind}]")
        print(f"\n{len(set(findings))} finding(s). Move the data into the git-ignored private files "
              "(see docs/PRIVATE_DATA.md).")
        return 1

    print(f"Sensitive-content check passed ({len(candidate_files())} files scanned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
