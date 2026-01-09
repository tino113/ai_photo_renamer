# Media Annotator & Smart Renamer

This project provides a cross-platform CLI and PySide6 GUI for scanning media libraries, extracting metadata, recognizing faces, generating LLM descriptions, and planning/applying safe renames or copies.

## Installation

```bash
pip install .
```

Optional extras:

```bash
pip install .[faces]
pip install .[llm_ollama]
pip install .[llm_lmstudio]
pip install .[llm_local]
pip install .[gui]
pip install .[faiss]
```

External binaries required:

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

## GUI

```bash
media-annotator gui
```

The GUI offers a pipeline tab, unknown people management, and rename preview.
