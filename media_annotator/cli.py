from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich import print
from rich.table import Table

from media_annotator.config import AppConfig
from media_annotator.db import dao
from media_annotator.db.migrations import run_migrations
from media_annotator.db.models import FaceEmbedding, MediaItem, MediaFace, Person
from media_annotator.db.session import create_session
from media_annotator.logging import setup_logging
from media_annotator.pipeline.apply_changes import apply_plan
from media_annotator.pipeline.describe_media import describe_media
from media_annotator.pipeline.rename_plan import generate_plan
from media_annotator.pipeline.runner import run_describe, run_faces, scan_media

app = typer.Typer(help="Media Annotator & Smart Renamer")
faces_app = typer.Typer(help="Face recognition commands")
app.add_typer(faces_app, name="faces")


def _pip_install(extras: str) -> None:
    if extras == "base":
        cmd = ["python", "-m", "pip", "install", "."]
    else:
        cmd = ["python", "-m", "pip", "install", f"media-annotator[{extras}]"]
    subprocess.run(cmd, check=False)


def _check_import(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _check_binary(name: str, args: list[str]) -> bool:
    result = subprocess.run([name] + args, capture_output=True, text=True)
    return result.returncode == 0


def _suggest_binary_install(name: str) -> None:
    print(f"Missing binary: {name}")
    if shutil.which("brew"):
        print(f"Try: brew install {name}")
    if shutil.which("winget"):
        print(f"Try: winget install {name}")
    print("Linux: use your package manager (apt/dnf/pacman) to install.")


@app.callback()
def init_app() -> None:
    config = AppConfig()
    config.ensure_dirs()
    setup_logging(config.log_dir)
    base_modules = ["pydantic", "typer", "sqlalchemy", "numpy", "loguru", "xxhash", "httpx", "cv2", "PIL"]
    for module in base_modules:
        if not _check_import(module):
            print(f"Missing required module: {module}")
            if typer.confirm("Install base dependencies now?"):
                _pip_install("base")
                break
    if not _check_binary("exiftool", ["-ver"]):
        _suggest_binary_install("exiftool")
    if not _check_binary("ffprobe", ["-version"]):
        _suggest_binary_install("ffprobe")


@app.command()
def scan(input_dir: Path) -> None:
    config = AppConfig()
    scan_media(config, input_dir)


@faces_app.command("preprocess")
def faces_preprocess(
    input_dir: Path,
    video_sample_rate: float = typer.Option(0.5),
    video_max_frames: int = typer.Option(300),
    threshold_known: float = typer.Option(0.65),
    threshold_unknown: float = typer.Option(0.7),
    use_faiss: bool = typer.Option(True),
    json_progress: bool = typer.Option(False, "--json-progress"),
) -> None:
    config = AppConfig()
    config.ensure_dirs()
    config.faces.video_sample_rate = video_sample_rate
    config.faces.video_max_frames = video_max_frames
    config.faces.known_match_threshold = threshold_known
    config.faces.unknown_match_threshold = threshold_unknown
    config.faces.use_faiss = use_faiss
    def _progress(path: str, status: str) -> None:
        if json_progress:
            print(json.dumps({"path": path, "status": status}))

    run_faces(config, input_dir, progress_callback=_progress)


@faces_app.command("review-unknowns")
def review_unknowns() -> None:
    config = AppConfig()
    session_factory = create_session(str(config.db_path))
    with session_factory() as session:
        run_migrations(session)
        unknowns = dao.get_unknown_people(session)
        table = Table("ID", "Name", "Occurrences")
        for person in unknowns:
            count = (
                session.query(MediaFace)
                .filter(MediaFace.person_id == person.person_id)
                .with_entities(MediaFace.count)
                .all()
            )
            total = sum(c[0] for c in count)
            table.add_row(str(person.person_id), person.display_name or "(unknown)", str(total))
        print(table)
        for person in unknowns:
            examples = (
                session.query(FaceEmbedding)
                .filter(FaceEmbedding.person_id == person.person_id)
                .limit(3)
                .all()
            )
            print(f"Examples for {person.display_name}: {[ex.media_path for ex in examples]}")
            name = typer.prompt(f"Enter name for {person.display_name} (leave blank to skip)", default="")
            if name:
                person.display_name = name
                person.is_known = True
                session.commit()
                logger.info("Updated person {} to {}", person.person_id, name)


@app.command()
def describe(
    input_dir: Path,
    backend: str = typer.Option("ollama"),
    model: str = typer.Option("llava"),
    base_url: Optional[str] = typer.Option(None),
    json_progress: bool = typer.Option(False, "--json-progress"),
) -> None:
    config = AppConfig()
    config.llm.backend = backend
    config.llm.model = model
    config.llm.base_url = base_url
    def _progress(path: str, status: str) -> None:
        if json_progress:
            print(json.dumps({"path": path, "status": status}))

    run_describe(config, input_dir, progress_callback=_progress)


@app.command("plan-renames")
def plan_renames(
    input_dir: Path,
    output_file: Path = Path("rename_plan.json"),
    output_dir: Optional[Path] = None,
    mirror_structure: bool = typer.Option(False, "--mirror-structure"),
) -> None:
    config = AppConfig()
    config.pipeline.copy_mirror_structure = mirror_structure
    config.pipeline.copy_mode = output_dir is not None
    session_factory = create_session(str(config.db_path))
    with session_factory() as session:
        run_migrations(session)
        items = [
            {
                "path": item.path,
                "meta_json": item.meta_json,
                "hash": item.hash,
            }
            for item in session.query(MediaItem).filter(MediaItem.path.like(f"{input_dir}%")).all()
        ]
    plan = generate_plan(config, items, output_dir, input_root=input_dir)
    output_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Plan written to {output_file}")


@app.command()
def apply(
    plan_file: Path,
    mode: str = "rename",
    output_dir: Optional[Path] = None,
    apply_changes: bool = typer.Option(False, "--apply"),
    undo_file: Optional[Path] = typer.Option(None),
) -> None:
    config = AppConfig()
    config.pipeline.dry_run = not apply_changes
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if mode == "copy" and output_dir is None:
        output_root = plan.get("output_root")
        if output_root:
            output_dir = Path(output_root)
    session_factory = create_session(str(config.db_path))
    with session_factory() as session:
        run_migrations(session)
        apply_plan(session, plan, mode, output_dir, dry_run=config.pipeline.dry_run, undo_file=undo_file)


@app.command()
def doctor() -> None:
    print("Checking Python dependencies...")
    modules = {
        "pydantic": "base",
        "typer": "base",
        "sqlalchemy": "base",
        "cv2": "base",
        "PIL": "base",
        "numpy": "base",
        "loguru": "base",
        "xxhash": "base",
        "httpx": "base",
        "insightface": "faces",
        "onnxruntime": "faces",
        "openai": "llm_lmstudio",
        "transformers": "llm_local",
        "PySide6": "gui",
        "faiss": "faiss",
    }
    missing_extras = set()
    for module, extra in modules.items():
        if not _check_import(module):
            print(f"Missing: {module} (extra: {extra})")
            missing_extras.add(extra)
    if missing_extras:
        for extra in missing_extras:
            if extra == "base":
                continue
            if typer.confirm(f"Install Python deps for [{extra}] now?"):
                _pip_install(extra)

    print("Checking external binaries...")
    if not _check_binary("exiftool", ["-ver"]):
        _suggest_binary_install("exiftool")
    if not _check_binary("ffprobe", ["-version"]):
        _suggest_binary_install("ffprobe")


@app.command()
def gui() -> None:
    from media_annotator.ui.gui_app import run_gui

    run_gui()
