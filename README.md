# Media Annotator & Smart Renamer

This project provides a cross-platform CLI and PySide6 GUI for scanning media libraries, extracting metadata, recognizing faces, generating LLM descriptions, and planning/applying safe renames or copies.

## Installation

### Recommended: create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install the package

```bash
pip install .
```

### Optional extras

Some extras (especially `faces` and `llm_local`) depend on packages that may not yet publish wheels for every
Python version. If you see missing or skipped extras, try Python 3.11 or 3.12 in your virtual environment.

> Tip (zsh users): quote the extra to avoid glob expansion, e.g. `pip install ".[faces]"`.

```bash
pip install ".[faces]"
pip install ".[llm_ollama]"
pip install ".[llm_lmstudio]"
pip install ".[llm_local]"
pip install ".[gui]"
pip install ".[faiss]"
```

### External binaries

Install these on your system so metadata extraction and video probing work:

- `exiftool`
- `ffmpeg` / `ffprobe`

## CLI Usage

```bash
media-annotator scan /path/to/media
media-annotator faces preprocess /path/to/media
media-annotator faces review-unknowns
media-annotator describe /path/to/media --backend ollama --model llava
media-annotator plan-renames /path/to/media --output-file rename_plan.json
media-annotator apply rename_plan.json --apply
media-annotator doctor
```

The CLI will store its SQLite database and cache under `~/.media_annotator` by default. Use the `doctor` command
to confirm dependencies and GPU/LLM backends are available before running a full pipeline.

## GUI

```bash
media-annotator gui
```

The GUI offers a pipeline tab, unknown people management, and rename preview.
