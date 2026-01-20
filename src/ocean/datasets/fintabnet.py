from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FinTabNetEntry:
    """A single FinTabNet record (one PDF page)."""

    pdf_filename: str
    png_path: Path
    table_bbox: Optional[List[float]]
    cells: List[Dict[str, Any]]
    structure_tokens: List[str]


def parse_fintabnet_jsonl(
    jsonl_path: str | Path,
    png_root: str | Path,
    *,
    assume_png_ext: str = ".png",
    skip_missing_png: bool = True,
    limit: int | None = None,
) -> Dict[str, FinTabNetEntry]:
    """Parse FinTabNet JSONL and map PDF filenames -> parsed entries.

    In the original notebooks, image filenames were created by replacing `.pdf` with `.png`.
    This function keeps that default but can be adapted.

    Returns a dict keyed by `pdf_filename` (the JSONL `filename`).
    """
    jsonl_path = Path(jsonl_path)
    png_root = Path(png_root)

    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    out: Dict[str, FinTabNetEntry] = {}
    missing = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break

            sample = json.loads(line)
            pdf_filename = sample["filename"]
            png_filename = Path(pdf_filename).with_suffix(assume_png_ext).name
            png_path = png_root / png_filename

            if skip_missing_png and not png_path.exists():
                missing += 1
                if missing <= 5 or missing % 1000 == 0:
                    logger.warning("PNG missing (showing first 5 then every 1000): %s", png_path)
                continue

            html = sample.get("html", {})
            structure_tokens = (html.get("structure", {}) or {}).get("tokens", []) or []
            cells = html.get("cells", []) or []

            out[pdf_filename] = FinTabNetEntry(
                pdf_filename=pdf_filename,
                png_path=png_path,
                table_bbox=sample.get("bbox"),
                cells=cells,
                structure_tokens=structure_tokens,
            )

    logger.info(
        "Parsed %s entries from %s (missing PNGs: %s)", len(out), jsonl_path.name, missing
    )
    return out
