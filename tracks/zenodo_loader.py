"""
Loader for the OAEI 2025 BeyondEquivalence track (a.k.a. GeneralRelation Benchmark).

The track is NOT distributed via OAEI's TDRS service — Zenodo
(DOI 10.5281/zenodo.17091043) is the only official source per the OAEI 2025
trackseite (http://oaei.ontologymatching.org/2025/beyondequivalence/index.html).
That is why this loader exists alongside Evaluation._ensure_track_downloaded() instead
of being folded into it.

Default flow
------------
1. The ZIP is expected at <project-root>/benchmark.zip.
2. On first call, it is extracted once into
   ~/oaei_track_cache/zenodo/beyondequivalence_v1/extracted/benchmark/
   (mirrors the layout used by the existing TDRS loader).
3. Subsequent calls reuse the cache.

Both source and cache locations are configurable via constructor arguments or
the env vars ZENODO_BENCHMARK_ZIP and ZENODO_BENCHMARK_CACHE.

This loader does NOT download from Zenodo. The ZIP must be provided locally.
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# OAEI 2025 BeyondEquivalence — officially listed sub-datasets.
# Source: http://oaei.ontologymatching.org/2025/beyondequivalence/index.html
OFFICIAL_OAEI_2025_SUBSETS: frozenset[str] = frozenset({
    # STROMA / TaSeR test cases
    "g1-web", "g2-diseases", "g3-text", "g5-groceries", "g7-literature",
    # Product classification standards
    "etim-eclass", "eclass-gpc", "eclass-unspsc",
    "gpc-unspsc", "gpc-unspscplus",
})

# Files we expect inside each sub-dataset directory after extraction.
EXPECTED_FILES: frozenset[str] = frozenset({"source.rdf", "target.rdf", "reference.rdf"})

# macOS-junk path components silently stripped during extraction.
_MACOS_JUNK_PREFIX = "__MACOSX/"
_MACOS_JUNK_NAMES = {".DS_Store"}
_APPLEDOUBLE_PREFIX = "._"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_source_zip() -> Path:
    env = os.getenv("ZENODO_BENCHMARK_ZIP")
    if env:
        return Path(env).expanduser().resolve()
    return _project_root() / "benchmark.zip"


def _default_cache_root() -> Path:
    env = os.getenv("ZENODO_BENCHMARK_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / "oaei_track_cache" / "zenodo" / "beyondequivalence_v1"


def _is_macos_junk(name: str) -> bool:
    if name.startswith(_MACOS_JUNK_PREFIX):
        return True
    base = name.rsplit("/", 1)[-1]
    if base in _MACOS_JUNK_NAMES:
        return True
    if base.startswith(_APPLEDOUBLE_PREFIX):
        return True
    return False


def ensure_extracted(
    source_zip: Path | None = None,
    cache_root: Path | None = None,
    force: bool = False,
) -> Path:
    """Extract benchmark.zip into the cache once; return the path to <cache>/extracted/benchmark/.

    The returned directory contains one sub-directory per sub-dataset, each with
    source.rdf / target.rdf / reference.rdf.
    """
    source_zip = source_zip or _default_source_zip()
    cache_root = cache_root or _default_cache_root()
    extracted_root = cache_root / "extracted"
    benchmark_dir = extracted_root / "benchmark"

    if force and extracted_root.exists():
        logger.info("force=True — removing existing cache at %s", extracted_root)
        shutil.rmtree(extracted_root)

    if benchmark_dir.is_dir() and any(benchmark_dir.iterdir()):
        logger.debug("Cache already populated at %s — skipping extraction", benchmark_dir)
        return benchmark_dir

    if not source_zip.is_file():
        raise FileNotFoundError(
            f"benchmark.zip not found at {source_zip}. "
            "Place it in the project root, or set ZENODO_BENCHMARK_ZIP / pass source_zip=. "
            "Official source: https://zenodo.org/records/17091043 (DOI 10.5281/zenodo.17091043)."
        )

    extracted_root.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s -> %s", source_zip, extracted_root)

    n_extracted = 0
    n_skipped = 0
    with zipfile.ZipFile(source_zip) as zf:
        for info in zf.infolist():
            if _is_macos_junk(info.filename):
                n_skipped += 1
                continue
            zf.extract(info, extracted_root)
            n_extracted += 1

    logger.info(
        "Extracted %d entr(ies); filtered %d macOS-junk entr(ies)",
        n_extracted, n_skipped,
    )

    _validate_extracted_layout(benchmark_dir)
    return benchmark_dir


def _validate_extracted_layout(benchmark_dir: Path) -> None:
    """Warn about anything in the extracted tree that isn't an EXPECTED_FILES file in a sub-dir."""
    if not benchmark_dir.is_dir():
        raise RuntimeError(
            f"After extraction, expected directory {benchmark_dir} does not exist. "
            "ZIP layout may have changed."
        )

    unexpected: list[str] = []
    missing: list[str] = []

    for top_child in sorted(benchmark_dir.iterdir()):
        if top_child.is_file():
            unexpected.append(str(top_child.relative_to(benchmark_dir)))
            continue
        present = set()
        for sub_child in sorted(top_child.iterdir()):
            if sub_child.is_dir():
                unexpected.append(str(sub_child.relative_to(benchmark_dir)) + "/")
                continue
            if sub_child.name not in EXPECTED_FILES:
                unexpected.append(str(sub_child.relative_to(benchmark_dir)))
            else:
                present.add(sub_child.name)
        for needed in EXPECTED_FILES - present:
            missing.append(f"{top_child.name}/{needed}")

    if unexpected:
        logger.warning(
            "Unexpected entr(ies) in extracted benchmark — ZIP layout may have changed: %s",
            unexpected,
        )
    if missing:
        logger.warning(
            "Missing expected file(s) in extracted benchmark — ZIP layout may have changed: %s",
            missing,
        )


def list_subdatasets(
    cache_root: Path | None = None,
    source_zip: Path | None = None,
) -> list[str]:
    """Return the names of all sub-datasets present in the extracted cache (sorted)."""
    benchmark_dir = ensure_extracted(source_zip=source_zip, cache_root=cache_root)
    return sorted(p.name for p in benchmark_dir.iterdir() if p.is_dir())


def load_subdataset(
    name: str,
    cache_root: Path | None = None,
    source_zip: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Return (source_path, target_path, reference_path) for the given sub-dataset.

    Logs a warning when `name` is not on the OAEI 2025 official whitelist.
    Raises ValueError if the sub-dataset directory does not exist, or
    FileNotFoundError if any of the three expected files is missing.
    """
    benchmark_dir = ensure_extracted(source_zip=source_zip, cache_root=cache_root)

    if name not in OFFICIAL_OAEI_2025_SUBSETS:
        logger.warning(
            "Sub-dataset '%s' is in the ZIP but not officially listed in OAEI 2025 "
            "BeyondEquivalence — using anyway.", name,
        )

    sub_dir = benchmark_dir / name
    if not sub_dir.is_dir():
        available = sorted(p.name for p in benchmark_dir.iterdir() if p.is_dir())
        raise ValueError(
            f"Sub-dataset '{name}' not found under {benchmark_dir}. "
            f"Available: {available}"
        )

    paths = (sub_dir / "source.rdf", sub_dir / "target.rdf", sub_dir / "reference.rdf")
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Sub-dataset '{name}' is missing expected file(s): {missing} (at {sub_dir})"
        )
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    available = list_subdatasets()
    print(f"Available sub-datasets ({len(available)}):")
    for n in available:
        marker = "OFFICIAL" if n in OFFICIAL_OAEI_2025_SUBSETS else "extra   "
        print(f"  [{marker}] {n}")
    src, tgt, ref = load_subdataset("g7-literature")
    print(
        "\ng7-literature paths:\n"
        f"  source:    {src}\n"
        f"  target:    {tgt}\n"
        f"  reference: {ref}"
    )
