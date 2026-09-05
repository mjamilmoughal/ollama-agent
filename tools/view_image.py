"""A tool that lets a vision-capable model actually see a local image file."""

from __future__ import annotations

import base64
import io
import os

from langchain_core.tools import tool

from ._project import PROJECT_ROOT

MAX_FILE_BYTES = 20_000_000
MAX_DIMENSION = 1024
JPEG_QUALITY = 85
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}


def _resolve(path: str) -> str | None:
    path = (path or "").strip().strip('"')
    if not path:
        return None
    if not os.path.isabs(path):
        path = str((PROJECT_ROOT / path).resolve())
    return path


@tool
def view_image(path: str) -> str | list:
    """Load a local image file so you can actually see its content.

    Pass a path -- absolute, or relative to the current project (e.g. one
    returned by find_project_file). Only offered when the currently selected
    model supports vision; if you don't have this tool, the active model
    can't see images, so tell the user to switch to a vision-capable model.
    Large images are automatically downscaled before being sent.
    """
    resolved = _resolve(path)
    if not resolved:
        return "Missing an image path to view."
    if not os.path.isfile(resolved):
        return f"'{path}' is not a file. Use find_project_file or find_file first."

    ext = os.path.splitext(resolved)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return f"'{path}' does not look like a supported image file ({', '.join(sorted(SUPPORTED_EXTENSIONS))})."

    try:
        size = os.path.getsize(resolved)
    except OSError as exc:
        return f"Could not stat '{path}': {exc}"
    if size > MAX_FILE_BYTES:
        return f"'{path}' is too large to view safely (limit: {MAX_FILE_BYTES} bytes)."

    try:
        from PIL import Image
    except ImportError:
        return "Image support requires the 'Pillow' package (pip install Pillow)."

    try:
        with Image.open(resolved) as img:
            img.load()
            original_size = img.size
            has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
            if max(img.size) > MAX_DIMENSION:
                scale = MAX_DIMENSION / max(img.size)
                new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
                img = img.resize(new_size, Image.LANCZOS)

            buffer = io.BytesIO()
            if has_alpha:
                img.convert("RGBA").save(buffer, format="PNG", optimize=True)
                mime = "image/png"
            else:
                img.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
                mime = "image/jpeg"
    except Exception as exc:
        return f"Could not open '{path}' as an image: {exc}"

    dimension_note = f"{img.width}x{img.height}"
    if img.size != original_size:
        dimension_note += f", downscaled from {original_size[0]}x{original_size[1]}"

    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return [
        {"type": "text", "text": f"Image loaded from '{path}' ({dimension_note})."},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]
