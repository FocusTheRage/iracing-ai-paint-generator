"""
Use iRacing's built-in 3D car model viewer for accurate paint preview.

iRacing does not ship 3D meshes for external apps — the sim's own UI viewer is the
only way to spin the real car with your TGA applied (same approach as Trading Paints
Sim Preview).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import winreg
from pathlib import Path

from cars_config import (
    IRacingCar,
    get_output_filenames,
    get_paint_install_path,
    get_paint_install_paths,
)
from paint_processor import get_session_output_dir

# Common iRacing UI install locations (Windows).
_UI_CANDIDATES: tuple[Path, ...] = (
    Path(r"D:\iRacing\ui\iRacingUI.exe"),
    Path(r"C:\Program Files (x86)\iRacing\ui\iRacingUI.exe"),
    Path(r"C:\Program Files\iRacing\ui\iRacingUI.exe"),
    Path(r"E:\iRacing\ui\iRacingUI.exe"),
)

def resolve_file_path(file_value: object | None) -> str | None:
    """Normalize Gradio File / FileData / dict / str to a local path string."""
    if file_value is None:
        return None
    if isinstance(file_value, str):
        stripped = file_value.strip()
        return stripped or None
    if isinstance(file_value, dict):
        path = file_value.get("path")
        return str(path) if path else None
    path = getattr(file_value, "path", None)
    if path:
        return str(path)
    root = getattr(file_value, "root", None)
    if root and len(root) > 0:
        first = root[0]
        if isinstance(first, dict):
            return first.get("path")
        return getattr(first, "path", None)
    return None


def find_iracing_ui_executable() -> Path | None:
    """Return iRacingUI.exe if installed."""
    for candidate in _UI_CANDIDATES:
        if candidate.exists():
            return candidate

    for root in (os.environ.get("ProgramFiles(x86)", ""), os.environ.get("ProgramFiles", "")):
        if not root:
            continue
        guess = Path(root) / "iRacing" / "ui" / "iRacingUI.exe"
        if guess.exists():
            return guess

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\iRacing") as key:
            install, _ = winreg.QueryValueEx(key, "LocalInstallDir")
            ui = Path(install) / "ui" / "iRacingUI.exe"
            if ui.exists():
                return ui
    except OSError:
        pass

    return None


def _clear_mip_cache(install_dir: Path, customer_id: str) -> int:
    """Remove cached .mip files so iRacing reloads fresh TGAs."""
    removed = 0
    paint_name, spec_name = get_output_filenames(customer_id)
    for name in (
        f"car_{customer_id}.mip",
        f"car_spec_{customer_id}.mip",
        paint_name.replace(".tga", ".mip"),
        spec_name.replace(".tga", ".mip"),
    ):
        target = install_dir / name
        if target.exists():
            target.unlink()
            removed += 1
    return removed


def resolve_paint_sources(
    car: IRacingCar,
    customer_id: str,
    paint_file: str | Path | object | None = None,
    spec_file: str | Path | object | None = None,
) -> tuple[Path, Path | None]:
    """
    Locate paint/spec TGAs for install or preview.

    Prefers explicit paths (Gradio File output) but falls back to the app's
    session output folder so copy/install still works when gr.State is empty
    or points at a stale Gradio temp file.
    """
    paint_name, spec_name = get_output_filenames(customer_id)
    session_dir = get_session_output_dir(car, customer_id)

    paint_src: Path | None = None
    if isinstance(paint_file, Path):
        paint_src = paint_file
    else:
        resolved = resolve_file_path(paint_file)
        if resolved:
            paint_src = Path(resolved)

    if paint_src is None or not paint_src.exists():
        fallback = session_dir / paint_name
        if fallback.exists():
            paint_src = fallback
        elif paint_src is not None:
            raise FileNotFoundError(f"Paint file not found: {paint_src}")
        else:
            raise FileNotFoundError(
                "No generated paint found — click **Generate Paint** first.\n"
                f"Expected: `{fallback}`"
            )

    spec_src: Path | None = None
    if isinstance(spec_file, Path):
        spec_src = spec_file
    else:
        resolved_spec = resolve_file_path(spec_file)
        if resolved_spec:
            candidate = Path(resolved_spec)
            if candidate.exists():
                spec_src = candidate

    if spec_src is None:
        fallback_spec = session_dir / spec_name
        if fallback_spec.exists():
            spec_src = fallback_spec

    return paint_src, spec_src


def install_paint_for_preview(
    car: IRacingCar,
    customer_id: str,
    paint_file: str | Path | object | None = None,
    spec_file: str | Path | object | None = None,
) -> list[Path]:
    """Copy TGAs into every applicable iRacing paint folder and clear mip cache."""
    dest_dirs = get_paint_install_paths(car, customer_id)

    paint_src, spec_src = resolve_paint_sources(
        car, customer_id, paint_file, spec_file
    )

    paint_name, spec_name = get_output_filenames(customer_id)
    copied_to: list[Path] = []

    for dest_dir in dest_dirs:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_paint = dest_dir / paint_name
        dest_spec = dest_dir / spec_name
        shutil.copy2(paint_src, dest_paint)

        if spec_src is not None:
            shutil.copy2(spec_src, dest_spec)

        _clear_mip_cache(dest_dir, customer_id)

        if not dest_paint.exists():
            raise OSError(f"Copy failed — file not written to {dest_paint}")

        copied_to.append(dest_dir)

    return copied_to


def launch_iracing_ui() -> tuple[bool, str]:
    """Start the iRacing UI (contains the 3D car model viewer)."""
    exe = find_iracing_ui_executable()
    if exe is None:
        return (
            False,
            "Could not find iRacingUI.exe. Open the iRacing app manually from Steam or the Start menu.",
        )
    try:
        subprocess.Popen(
            [str(exe)],
            cwd=str(exe.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True, f"Launched iRacing UI (`{exe}`)."
    except OSError as exc:
        return False, f"Failed to launch iRacing UI: {exc}"


def open_iracing_3d_preview(
    car: IRacingCar,
    customer_id: str,
    paint_file: object | None = None,
    spec_file: object | None = None,
) -> str:
    """Install paint files and launch iRacing UI for 3D preview."""
    try:
        dest_dirs = install_paint_for_preview(car, customer_id, paint_file, spec_file)
    except FileNotFoundError as exc:
        return f"**Error:** {exc}"
    except OSError as exc:
        return f"**Error:** {exc}"

    launched, launch_msg = launch_iracing_ui()
    folder_lines = "\n".join(f"- `{d}`" for d in dest_dirs)
    lines = [
        "**Paint installed for iRacing 3D preview**",
        folder_lines,
        f"- {launch_msg}",
    ]
    if launched:
        lines.extend(
            [
                "",
                f"1. In iRacing UI, open **My Content → Cars → {car.display_name}**",
                "2. Open **Car Model** or **Paint Shop**",
                f"3. Select your **Custom Paint** (Customer ID **{customer_id}**)",
                "4. Drag to rotate — this is the real in-sim 3D viewer",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Open iRacing manually, then open My Content → Cars → your vehicle → Paint Shop.",
            ]
        )

    return "\n".join(lines)