from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoreConfig:
    """Configuration for the CORE API.

    CORE's API has multiple versions; this client keeps things simple and
    expects you to provide the base url and your API key.

    Env vars supported:
      - CORE_BASE_URL (default: https://api.core.ac.uk)
      - CORE_API_KEY
    """

    base_url: str = os.getenv("CORE_BASE_URL", "https://api.core.ac.uk")
    api_key: str | None = os.getenv("CORE_API_KEY")


def robust_get(
    session: requests.Session,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    max_retries: int = 5,
    timeout_s: int = 60,
    backoff_s: int = 5,
) -> Optional[requests.Response]:
    """GET with simple retry/backoff.

    This mirrors the notebook logic but is deterministic and reusable.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, headers=headers, stream=True, timeout=timeout_s)
            return resp
        except (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ) as e:
            logger.warning("GET failed (%s/%s): %s", attempt, max_retries, e)
            time.sleep(backoff_s * attempt)

    logger.error("GET failed after %s attempts: %s", max_retries, url)
    return None


def _auth_headers(cfg: CoreConfig) -> Dict[str, str]:
    if not cfg.api_key:
        raise RuntimeError(
            "CORE_API_KEY is not set. Export it as an env var before running."
        )
    return {"Authorization": f"Bearer {cfg.api_key}"}


def search_papers(
    query: str,
    *,
    cfg: CoreConfig | None = None,
    limit: int = 100,
    offset: int = 0,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """Search papers via CORE.

    Returns the raw JSON response. The exact endpoint can vary by CORE API
    version; adjust `endpoint` if needed.
    """
    cfg = cfg or CoreConfig()
    session = session or requests.Session()

    # NOTE: CORE's API has changed over time. This endpoint name might need
    # to be adjusted for your account/plan. The goal here is to keep the code
    # organized; you can swap the URL in one place.
    endpoint = f"{cfg.base_url}/v3/search/works?q={requests.utils.quote(query)}&limit={limit}&offset={offset}"

    resp = robust_get(session, endpoint, headers=_auth_headers(cfg))
    if resp is None:
        raise RuntimeError("CORE request failed (no response)")

    if resp.status_code != 200:
        raise RuntimeError(f"CORE request failed: HTTP {resp.status_code}: {resp.text[:500]}")

    return resp.json()


def iter_papers(
    query: str,
    *,
    cfg: CoreConfig | None = None,
    pulls: int = 100,
    runs: int = 10,
    session: Optional[requests.Session] = None,
) -> Iterable[Dict[str, Any]]:
    """Yield papers across multiple paginated pulls.

    This replaces the missing `getPaperAPI.collect_data` referenced in the notebook.
    """
    cfg = cfg or CoreConfig()
    session = session or requests.Session()

    for i in range(runs):
        offset = i * pulls
        data = search_papers(query, cfg=cfg, limit=pulls, offset=offset, session=session)

        # Try common response shapes
        items = (
            data.get("results")
            or data.get("data")
            or data.get("hits")
            or data.get("works")
            or []
        )

        if not isinstance(items, list):
            logger.warning("Unexpected search response shape. keys=%s", list(data.keys()))
            break

        for item in items:
            yield item

        if len(items) < pulls:
            break


def download_pdf_from_work(
    work: Dict[str, Any],
    out_dir: str | Path,
    *,
    cfg: CoreConfig | None = None,
    session: Optional[requests.Session] = None,
    filename: Optional[str] = None,
) -> Optional[Path]:
    """Download a PDF for a given work dict if a downloadable link exists.

    The notebook assumed `work['links'][0]` is a download. Here we look for a
    best-effort download link.
    """
    cfg = cfg or CoreConfig()
    session = session or requests.Session()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    links = work.get("links") or []
    pdf_url = None
    for link in links:
        if isinstance(link, dict) and link.get("type") in {"download", "pdf", "fullText"}:
            pdf_url = link.get("url")
            break
    if not pdf_url:
        return None

    resp = robust_get(session, pdf_url, headers=_auth_headers(cfg))
    if resp is None or resp.status_code != 200:
        logger.warning("PDF download failed: %s (status=%s)", pdf_url, getattr(resp, "status_code", None))
        return None

    work_id = str(work.get("id") or work.get("_id") or "unknown").replace("/", "_")
    out_name = filename or f"{work_id}.pdf"
    out_path = out_dir / out_name

    with out_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)

    return out_path


def preprocess_text(text: Optional[str]) -> str:
    """Minimal text normalization from the notebook."""
    import re
    import string

    text = text or ""
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    return text.strip()


def preprocess_metadata(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for item in items:
        item = dict(item)
        item["title"] = preprocess_text(item.get("title"))
        item["abstract"] = preprocess_text(item.get("abstract"))
        out.append(item)
    return out


def save_jsonl(items: Iterable[Dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path
