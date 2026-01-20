from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Extract the table from this image and output only plain CSV text — "
    "no Markdown formatting, no triple backticks, no language tags, no explanations, no extra text.\n"
    "Convert checkmarks (✓) and crosses (✗) into '1' and '0'.\n"
    "Flatten multi-level headers into a single header row by combining levels, using a space or underscore if needed.\n"
    "If there are grouped sections or repeated row headers, repeat the values to ensure each row is complete.\n"
    "The result must be pure CSV text, machine-readable, and directly usable for SQL ingestion.\n"
    "Return only the CSV content itself, nothing else."
)


@dataclass
class OpenAIVisionConfig:
    model: str = "gpt-4o"
    max_tokens: int = 4000
    prompt: str = DEFAULT_PROMPT


def _encode_image_b64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def extract_csv_from_image(
    image_path: str | Path,
    *,
    cfg: OpenAIVisionConfig | None = None,
) -> str:
    """Call OpenAI vision model and return raw CSV text."""
    cfg = cfg or OpenAIVisionConfig()
    image_path = Path(image_path)

    try:
        from openai import OpenAI
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OpenAI dependency not installed. Install with: pip install -e '.[openai]'"
        ) from e

    client = OpenAI()

    b64 = _encode_image_b64(image_path)

    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": cfg.prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=cfg.max_tokens,
    )

    return (resp.choices[0].message.content or "").strip()


def batch_extract_csv(
    images: Dict[str, Path],
    *,
    cfg: OpenAIVisionConfig | None = None,
    out_jsonl: Optional[str | Path] = None,
    limit: Optional[int] = None,
) -> Dict[str, str]:
    """Run extraction over a dict of {id: path}.

    If out_jsonl is provided, writes one JSON object per line:
      {"id": ..., "image_path": ..., "csv": ...}
    """
    import json

    cfg = cfg or OpenAIVisionConfig()
    results: Dict[str, str] = {}

    out_f = None
    if out_jsonl is not None:
        out_path = Path(out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")

    try:
        for idx, (k, path) in enumerate(images.items()):
            if limit is not None and idx >= limit:
                break
            logger.info("Extracting CSV (%s/%s): %s", idx + 1, (limit or len(images)), k)
            try:
                csv = extract_csv_from_image(path, cfg=cfg)
                results[k] = csv
                if out_f:
                    out_f.write(
                        json.dumps({"id": k, "image_path": str(path), "csv": csv}, ensure_ascii=False)
                        + "\n"
                    )
            except Exception:
                logger.exception("Failed extraction for %s", k)
    finally:
        if out_f:
            out_f.close()

    return results
