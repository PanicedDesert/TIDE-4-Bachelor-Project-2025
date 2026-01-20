from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ocean.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def _cmd_papers(args: argparse.Namespace) -> int:
    """Paper retrieval subcommand."""
    from ocean.papers.core_api import (
        download_pdf_from_work,
        iter_papers,
        preprocess_metadata,
        save_jsonl,
    )

    if args.action == "search":
        items = list(iter_papers(args.query, pulls=args.pulls, runs=args.runs))
        if args.preprocess:
            items = preprocess_metadata(items)
        if args.out:
            save_jsonl(items, args.out)
            print(f"Saved {len(items)} works to {args.out}")
        else:
            print(f"Fetched {len(items)} works")
        return 0

    if args.action == "download":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        items = list(iter_papers(args.query, pulls=args.pulls, runs=args.runs))
        n_ok = 0
        for work in items:
            path = download_pdf_from_work(work, out_dir)
            if path is not None:
                n_ok += 1
        print(f"Downloaded {n_ok} PDFs to {out_dir}")
        return 0

    raise ValueError(f"Unknown papers action: {args.action}")


def _cmd_pdf(args: argparse.Namespace) -> int:
    from ocean.pdf.render import batch_render_from_jsonl, render_pdf_to_png

    if args.action == "render":
        out = render_pdf_to_png(args.pdf, out_dir=args.out_dir, dpi=args.dpi, overwrite=args.overwrite)
        print(f"Rendered {len(out)} PNG(s)")
        return 0

    if args.action == "render-from-jsonl":
        n = batch_render_from_jsonl(
            args.jsonl,
            args.pdf_root,
            dpi=args.dpi,
            overwrite=args.overwrite,
            limit=args.limit,
        )
        print(f"Rendered {n} PDFs")
        return 0

    raise ValueError(f"Unknown pdf action: {args.action}")


def _cmd_openai(args: argparse.Namespace) -> int:
    from ocean.models.openai_csv_extractor import OpenAIVisionConfig, batch_extract_csv

    # Build image map from a folder (png/jpg)
    folder = Path(args.images_dir)
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    paths = sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])
    images = {p.stem: p for p in paths}

    cfg = OpenAIVisionConfig(model=args.model, max_tokens=args.max_tokens)
    batch_extract_csv(images, cfg=cfg, out_jsonl=args.out_jsonl, limit=args.limit)
    print(f"Done. Wrote results to {args.out_jsonl}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ocean", description="Ocean pipeline CLI")
    p.add_argument("--log-level", default=None, help="INFO, DEBUG, WARNING...")

    sub = p.add_subparsers(dest="cmd", required=True)

    # papers
    papers = sub.add_parser("papers", help="Search & download papers (CORE API)")
    papers.add_argument("action", choices=["search", "download"])
    papers.add_argument("--query", required=True)
    papers.add_argument("--pulls", type=int, default=100)
    papers.add_argument("--runs", type=int, default=10)
    papers.add_argument("--out", default=None, help="Save search results as JSONL")
    papers.add_argument("--out-dir", default="retrieved_papers", help="Directory for PDFs")
    papers.add_argument("--preprocess", action="store_true", help="Normalize title/abstract text")
    papers.set_defaults(func=_cmd_papers)

    # pdf
    pdf = sub.add_parser("pdf", help="Render PDFs to images")
    pdf.add_argument("action", choices=["render", "render-from-jsonl"])
    pdf.add_argument("--pdf", default=None, help="Single PDF path (for render)")
    pdf.add_argument("--out-dir", default=None)
    pdf.add_argument("--jsonl", default=None, help="FinTabNet-style JSONL (for render-from-jsonl)")
    pdf.add_argument("--pdf-root", default=".")
    pdf.add_argument("--dpi", type=int, default=300)
    pdf.add_argument("--overwrite", action="store_true")
    pdf.add_argument("--limit", type=int, default=None)
    pdf.set_defaults(func=_cmd_pdf)

    # openai
    oa = sub.add_parser("openai", help="Run OpenAI vision CSV extraction")
    oa.add_argument("--images-dir", required=True)
    oa.add_argument("--out-jsonl", required=True)
    oa.add_argument("--model", default="gpt-4o")
    oa.add_argument("--max-tokens", type=int, default=4000)
    oa.add_argument("--limit", type=int, default=None)
    oa.set_defaults(func=_cmd_openai)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    logger.debug("Args: %s", args)

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
