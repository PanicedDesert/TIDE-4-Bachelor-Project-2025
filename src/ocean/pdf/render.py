from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def render_pdf_to_png(
    pdf_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    dpi: int = 300,
    overwrite: bool = False,
) -> List[Path]:
    """Render each page of a PDF to PNG using PyMuPDF.

    Notes vs the original notebook:
    - The notebook saved only one PNG per PDF (overwriting for multi-page PDFs).
      Here we suffix with page index to be safe.
    - FinTabNet filenames are usually single-page PDFs; keeping page suffix does
      not hurt.
    """
    pdf_path = Path(pdf_path)
    if out_dir is None:
        out_dir = pdf_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    out_paths: List[Path] = []

    for page_idx, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        base = pdf_path.stem
        if doc.page_count == 1:
            filename = f"{base}.png"
        else:
            filename = f"{base}_p{page_idx:03d}.png"
        out_path = out_dir / filename

        if out_path.exists() and not overwrite:
            out_paths.append(out_path)
            continue

        pix.save(str(out_path))
        out_paths.append(out_path)

    return out_paths


def batch_render_from_jsonl(
    jsonl_path: str | Path,
    pdf_root: str | Path,
    *,
    dpi: int = 300,
    overwrite: bool = False,
    limit: int | None = None,
) -> int:
    """Render PDFs referenced in a FinTabNet-style JSONL to PNG.

    The JSONL entries must contain `filename` (relative PDF path/name).

    Returns the number of PDFs processed.
    """
    import json

    jsonl_path = Path(jsonl_path)
    pdf_root = Path(pdf_root)

    processed = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if limit is not None and processed >= limit:
                break

            sample = json.loads(line)
            rel_pdf = sample.get("filename")
            if not rel_pdf:
                continue

            pdf_path = pdf_root / rel_pdf
            if not pdf_path.exists():
                logger.warning("Missing PDF: %s", pdf_path)
                continue

            try:
                out_paths = render_pdf_to_png(pdf_path, dpi=dpi, overwrite=overwrite)
                processed += 1
                if processed % 500 == 0:
                    logger.info("Rendered %s PDFs (latest: %s)", processed, out_paths[0])
            except Exception as e:
                logger.exception("Failed rendering %s: %s", pdf_path, e)

    logger.info("Done. Rendered %s PDFs referenced by %s", processed, jsonl_path)
    return processed
