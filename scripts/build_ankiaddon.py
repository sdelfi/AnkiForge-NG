#!/usr/bin/env python3
"""Сборка .ankiaddon файла для публикации на AnkiWeb и GitHub Releases."""

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
    """Читает __version__ из __init__.py."""
    init = SRC_PKG / "__init__.py"
    match = re.search(r'__version__\s*=\s*"(.+?)"', init.read_text())
    if not match:
        raise RuntimeError("Не удалось найти __version__ в __init__.py")
    return match.group(1)


def should_exclude(path: Path) -> bool:
    """Проверяет, нужно ли исключить файл из архива."""
    return any(part in EXCLUDE_PATTERNS for part in path.parts) or any(
        path.name.endswith(ext) for ext in (".pyc", ".pyo")
    )


def build() -> Path:
    """Собирает .ankiaddon архив и возвращает путь к файлу."""
    version = get_version()
    DIST.mkdir(exist_ok=True)

    output = DIST / f"AnkiForgeX_v{version}.ankiaddon"

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        # manifest.json — в корень архива
        manifest = SRC_PKG / "manifest.json"
        zf.write(manifest, "manifest.json")

        # config.json — в корень архива (Anki ожидает рядом с manifest)
        config = SRC_PKG / "config.json"
        if config.exists():
            zf.write(config, "config.json")

        # Пакет ankiforge/ — весь код
        for file in sorted(SRC_PKG.rglob("*")):
            if not file.is_file():
                continue
            if should_exclude(file):
                continue
            # manifest.json и config.json уже добавлены в корень
            if file.name in ("manifest.json", "config.json"):
                continue

            arcname = str(file.relative_to(SRC_PKG.parent))
            zf.write(file, arcname)

        names = zf.namelist()

    print(f"Собрано: {output}")
    print(f"  Версия: {version}")
    print(f"  Файлов: {len(names)}")
    print(f"  Размер: {output.stat().st_size / 1024:.1f} KB")

    return output


if __name__ == "__main__":
    build()
