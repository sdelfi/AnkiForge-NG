#!/usr/bin/env python3
"""Build .ankiaddon file for publishing on AnkiWeb and GitHub Releases."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_PKG = ROOT / "src" / "ankiforge"
DIST = ROOT / "dist"

EXCLUDE_PATTERNS: set[str] = {
    "__pycache__",
    ".DS_Store",
    ".pyc",
    ".pyo",
    "meta.json",
    "cost_log.json",
    "py.typed",
}


def get_version() -> str:
    """Read __version__ from __init__.py."""
    init = SRC_PKG / "__init__.py"
    match = re.search(r'__version__\s*=\s*"(.+?)"', init.read_text())
    if not match:
        raise RuntimeError("Could not find __version__ in __init__.py")
    return match.group(1)


def should_exclude(path: Path) -> bool:
    """Check if a file should be excluded from the archive."""
    return any(part in EXCLUDE_PATTERNS for part in path.parts) or any(
        path.name.endswith(ext) for ext in (".pyc", ".pyo")
    )


def build() -> Path:
    """Build .ankiaddon archive and return the file path."""
    version = get_version()
    DIST.mkdir(exist_ok=True)

    output = DIST / f"AnkiForge_v{version}.ankiaddon"

    # Thin entry point that adds addon dir to sys.path so `from ankiforge.xxx` works
    entry_point = (
        "import sys, os\n"
        "sys.path.insert(0, os.path.dirname(__file__))\n"
        "from ankiforge import _register_addon\n"
        "_register_addon()\n"
    )

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        # Root __init__.py — thin entry point
        zf.writestr("__init__.py", entry_point)

        # manifest.json and config.json — to archive root
        manifest = SRC_PKG / "manifest.json"
        zf.write(manifest, "manifest.json")

        config = SRC_PKG / "config.json"
        if config.exists():
            zf.write(config, "config.json")

        # ankiforge/ package — all code inside subfolder
        for file in sorted(SRC_PKG.rglob("*")):
            if not file.is_file():
                continue
            if should_exclude(file):
                continue
            if file.name in ("manifest.json", "config.json"):
                continue

            arcname = "ankiforge/" + str(file.relative_to(SRC_PKG))
            zf.write(file, arcname)

        names = zf.namelist()

    print(f"Built: {output}")
    print(f"  Version: {version}")
    print(f"  Files: {len(names)}")
    print(f"  Size: {output.stat().st_size / 1024:.1f} KB")

    return output


if __name__ == "__main__":
    build()
