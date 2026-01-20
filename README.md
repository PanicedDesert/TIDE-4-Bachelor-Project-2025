# Ocean (refactored)

This repository started as a set of Jupyter notebooks for a pipeline around **scientific PDFs** and **table extraction** (FinTabNet/TableBank + GPT‑4o + Detectron2).

It has been refactored into a **normal Python package** with:
- `src/` layout (importable modules)
- small files with single responsibilities
- a CLI (`ocean ...`) for reproducible runs
- configuration via environment variables / arguments (no hardcoded API keys)

The original notebooks are preserved in `original/` for reference.

---

## Pipeline overview (methodology)

The project is organized as a set of stages you can run independently or end‑to‑end:

1) **Paper retrieval (optional)**
   - Query CORE for papers (title/abstract/links)
   - Optionally download available PDFs
   - Optionally normalize metadata for later text processing

2) **PDF → image rendering (dataset preparation)**
   - Render each PDF page to PNG using PyMuPDF (fast and consistent)
   - In FinTabNet, each “PDF” is typically a single page; we still handle multi‑page PDFs safely

3) **Dataset parsing (FinTabNet)**
   - Parse the JSONL annotations (table bbox, cell bboxes, structure tokens)
   - Map each annotation row to the corresponding rendered PNG

4) **Table extraction options**

   **Option A: OpenAI Vision → CSV**
   - Send each PNG to an OpenAI vision model with a strict “CSV only” prompt
   - Store model outputs (JSONL) so you can evaluate later without re‑calling the API

   **Option B: Train a joint detector (Detectron2)**
   - Train one detector with 2 classes: `table` and `cell`
   - Add a *cell containment* auxiliary loss (GTE‑inspired): predicted cells should fall inside a predicted table

5) **Evaluation (not fully implemented yet)**
   - The repo stores the *inputs + outputs* needed for evaluation:
     - OpenAI CSV outputs
     - FinTabNet ground truth cells/structure/table bbox
   - Implementing full cell‑matching or structure accuracy metrics is project‑specific and can be added cleanly now that the code is modular.

---

## Project structure

```
.
├─ src/ocean/
│  ├─ cli/                 # command-line entrypoints
│  │  └─ main.py
│  ├─ papers/              # paper retrieval + metadata preprocessing
│  │  └─ core_api.py
│  ├─ pdf/                 # pdf utilities
│  │  └─ render.py
│  ├─ datasets/            # dataset readers/parsers
│  │  └─ fintabnet.py
│  ├─ models/
│  │  ├─ openai_csv_extractor.py   # OpenAI vision -> CSV
│  │  └─ gte_detectron2.py         # Detectron2 containment-loss helper
│  └─ utils/
│     └─ logging.py
│
├─ original/               # your original notebooks (kept for reference)
├─ configs/                # reserved for future YAML/Detectron2 configs
├─ scripts/                # reserved for one-off runnable scripts
├─ tests/                  # reserved for unit tests
└─ README.md
```

Why this layout:
- **Imports work the same everywhere** (`pip install -e .` then `from ocean...`).
- Code is **testable** (small modules; no global state).
- You can run everything **from the CLI** with explicit inputs and outputs.

---

## Installation

Create a fresh environment (recommended) and install editable:

```bash
pip install -e .
```

Optional extras:

```bash
# OpenAI vision extraction
pip install -e ".[openai]"

# Core ML dependencies (torch, opencv, matplotlib)
pip install -e ".[ml]"
```

Detectron2 installation is system/CUDA specific and should be installed separately.

---

## Configuration

### CORE API (paper retrieval)
Set:
- `CORE_API_KEY` (required)
- `CORE_BASE_URL` (optional, default: `https://api.core.ac.uk`)

Example:

```bash
export CORE_API_KEY="..."
```

### OpenAI
Set standard OpenAI env vars (recommended):
- `OPENAI_API_KEY` (required)
- optionally `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID` depending on your setup

Example:

```bash
export OPENAI_API_KEY="..."
```

---

## Usage (CLI)

After installing (`pip install -e .`), you get an `ocean` command.

### 1) Search papers on CORE

```bash
ocean papers search --query "oceanographic" --pulls 100 --runs 3 --out outputs/core_results.jsonl --preprocess
```

### 2) Download PDFs (when available)

```bash
ocean papers download --query "oceanographic" --pulls 50 --runs 2 --out-dir retrieved_papers
```

### 3) Render a single PDF to PNG

```bash
ocean pdf render --pdf retrieved_papers/some.pdf --out-dir outputs/images --dpi 300
```

### 4) Render PDFs referenced in a FinTabNet JSONL

Assuming:
- PDFs are in `fintabnet/pdf/`
- JSONL points to those PDF filenames

```bash
ocean pdf render-from-jsonl --jsonl fintabnet/FinTabNet_1.0.0_table_train.jsonl --pdf-root fintabnet/pdf --dpi 300
```

### 5) OpenAI vision extraction (PNG folder → JSONL results)

```bash
ocean openai --images-dir fintabnet/pdf --out-jsonl outputs/gpt_tables.jsonl --model gpt-4o --limit 50
```

---

## Notes on changes from the notebooks

- No hardcoded keys. Everything uses environment variables.
- PDF rendering is now safe for multi-page PDFs (page suffixing).
- `getPaperAPI.collect_data` from the notebook didn’t exist in the repo; it is replaced by `iter_papers()` in `ocean/papers/core_api.py`.
- Detectron2 customization is separated into `gte_detectron2.py` (clean place to evolve the joint model and containment loss).

---

## Next recommended improvements

If you want this repo to be “production-ready”, these are the best next steps:

1) Add a **proper experiment config** (YAML) and an explicit `train` CLI subcommand for Detectron2.
2) Add **evaluation scripts**:
   - CSV → table grid parsing
   - cell matching metrics (precision/recall/F1)
   - structure metrics (TEDS-style if desired)
3) Add unit tests for:
   - JSONL parsing
   - PDF rendering
   - metadata preprocessing

---

## License

MIT (set in `pyproject.toml`).
