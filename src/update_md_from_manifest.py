"""
Sync image descriptions from image_manifest.json into manual.md.

Replaces every IMAGE_PLACEHOLDER block with the corresponding manifest entry.
Safe to re-run at any time — partial runs and re-runs both work correctly.

Usable descriptions: any non-null text that is not an error tag, not a model
refusal ("unable to analyze" etc.), and not a decorative-element placeholder.
Blocks without a usable description are left unchanged in the manual.

Importable:
    from update_md_from_manifest import load_manifest, update_markdown
    idx = load_manifest(output_dir="/path/to/output")
    update_markdown(idx, output_dir="/path/to/output")

Run:
    python update_md_from_manifest.py
"""

import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
OUTPUT_DIR  = PROJECT_DIR / "data"
MD_PATH     = OUTPUT_DIR / "manual.md"
MANIFEST    = OUTPUT_DIR / "image_manifest.json"

BLOCK_RE = re.compile(r'<!-- IMAGE_PLACEHOLDER.*?-->', re.DOTALL)

REFUSAL_PHRASES = (
    "unable to analyze", "cannot analyze", "i cannot", "i am unable",
    "cannot provide", "i'm unable", "not able to", "i don't have the ability",
)

# Tags that are intentional and should be written into the manual.
ALLOWED_TAGS = ("[DECORATIVE ELEMENT",)


def _is_usable(desc: str) -> bool:
    """Return True if desc should be written into manual.md."""
    if not desc:
        return False
    # Allow intentional placeholder tags.
    if any(desc.startswith(t) for t in ALLOWED_TAGS):
        return True
    # Reject error/missing tags.
    if desc.startswith("["):
        return False
    # Reject model refusals.
    d = desc.lower()
    if any(p in d for p in REFUSAL_PHRASES):
        return False
    return True


def load_manifest(output_dir=None):
    """
    Load image_manifest.json and return an index dict keyed by image index.

    Args:
        output_dir: directory containing image_manifest.json.
                    Defaults to OUTPUT_DIR (next to this script).
    """
    out     = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    entries = json.loads((out / "image_manifest.json").read_text())
    index   = {e["index"]: e for e in entries}
    usable  = sum(1 for e in entries if _is_usable(e.get("description") or ""))
    pending = sum(1 for e in entries if not e.get("description"))
    bad     = len(entries) - usable - pending
    print(f"Manifest: {len(entries)} total | {usable} writable | {pending} pending | {bad} bad/refusal (skipped)")
    return index


def rebuild_block(entry):
    desc  = entry.get("description") or "TO_BE_FILLED_BY_VISION_MODEL"
    model = entry.get("model") or "TO_BE_ASSIGNED"
    return (
        f"<!-- IMAGE_PLACEHOLDER\n"
        f"index: {entry['index']}\n"
        f"source: {entry['source']}\n"
        f"caption: {entry.get('caption', '')}\n"
        f"model: {model}\n"
        f"description: {desc}\n"
        f"-->"
    )


def update_markdown(index, output_dir=None):
    """
    Patch every IMAGE_PLACEHOLDER block in manual.md with the manifest description.

    Args:
        index:      dict returned by load_manifest().
        output_dir: directory containing manual.md.
                    Defaults to OUTPUT_DIR (next to this script).
    """
    out     = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    md_path = out / "manual.md"
    md = md_path.read_text(encoding="utf-8")
    replaced = skipped = no_entry = 0

    def patch(m):
        nonlocal replaced, skipped, no_entry
        block = m.group(0)

        idx_m = re.search(r'^index:\s*(\d+)', block, re.MULTILINE)
        if not idx_m:
            skipped += 1
            return block

        idx   = int(idx_m.group(1))
        entry = index.get(idx)
        if not entry:
            no_entry += 1
            return block

        desc = entry.get("description") or ""
        if not _is_usable(desc):
            # Pending (null) or bad description — reset to blank so no stale text lingers.
            if not desc:
                skipped += 1
                entry_blank = dict(entry, description="TO_BE_FILLED_BY_VISION_MODEL", model="TO_BE_ASSIGNED")
                return rebuild_block(entry_blank)
            skipped += 1
            return block

        replaced += 1
        return rebuild_block(entry)

    updated = BLOCK_RE.sub(patch, md)
    md_path.write_text(updated, encoding="utf-8")
    print(f"Updated: {replaced}  |  Skipped (pending/bad): {skipped}  |  No manifest entry: {no_entry}")


def main(output_dir=None):
    """
    Sync manifest descriptions into manual.md.

    Args:
        output_dir: directory containing image_manifest.json and manual.md.
                    Defaults to OUTPUT_DIR (next to this script).
    """
    index = load_manifest(output_dir=output_dir)
    update_markdown(index, output_dir=output_dir)
    print("Done — manual.md updated.")


if __name__ == "__main__":
    main()
