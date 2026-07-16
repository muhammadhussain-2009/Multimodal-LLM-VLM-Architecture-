"""
backend/image_preprocessor.py
==============================
Robust image normalization layer for user-uploaded STEM diagrams.

Handles:
- JPEG, PNG, WebP, GIF (first frame), BMP, TIFF
- HEIC/HEIF (converted via Pillow plugin if available)
- Corrupt EXIF headers (auto-stripped)
- Truncated / partial uploads (graceful PIL recovery mode)
- Preserves aspect ratio via letterbox zero-padding
- Outputs canonical 448×448 RGB PNG bytes ready for VLM ingestion

Design notes:
- No external network calls — purely CPU-bound transforms.
- Works headlessly (opencv-python-headless, no display server needed).
- Falls back to Pillow-only path if OpenCV is unavailable.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger("ImagePreprocessor")

# Target resolution — matches LLaVA 1.5 / InstructBLIP optimal input size
TARGET_SIZE = (448, 448)

# ---------------------------------------------------------------------------
# Try to import OpenCV (optional — falls back to pure-Pillow path)
# ---------------------------------------------------------------------------
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.info("OpenCV not available — using pure-Pillow preprocessing path.")

try:
    from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
    ImageFile.LOAD_TRUNCATED_IMAGES = True   # tolerate truncated uploads
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    raise RuntimeError("Pillow is required. Install with: pip install Pillow")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_image(
    raw_bytes: bytes,
    filename: str = "upload",
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> bytes:
    """
    Normalize an uploaded image to a canonical RGB PNG suitable for VLM input.

    Args:
        raw_bytes:   Raw file bytes from the upload endpoint.
        filename:    Original filename (used only for extension hints).
        target_size: Output (width, height). Default 448×448.

    Returns:
        PNG-encoded bytes of the normalized image.

    Raises:
        ValueError: If the bytes cannot be interpreted as any image format.
    """
    pil_img = _load_image_safely(raw_bytes, filename)
    pil_img = _to_rgb(pil_img)
    pil_img = _strip_exif(pil_img)
    pil_img = _letterbox_resize(pil_img, target_size)

    out_buf = io.BytesIO()
    pil_img.save(out_buf, format="PNG", optimize=False)
    return out_buf.getvalue()


def validate_image(raw_bytes: bytes, filename: str = "upload") -> Tuple[bool, str]:
    """
    Validate that the uploaded bytes represent a parseable image.

    Returns:
        (True, "") on success.
        (False, <reason>) on failure.
    """
    if len(raw_bytes) == 0:
        return False, "File is empty (0 bytes)."

    if len(raw_bytes) < 10:
        return False, "File too small to be a valid image."

    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif",
                          ".bmp", ".tiff", ".tif", ".heic", ".heif"}
    ext = os.path.splitext(filename.lower())[1]
    if ext and ext not in allowed_extensions:
        return False, (
            f"File extension '{ext}' is not supported. "
            f"Please upload one of: {', '.join(sorted(allowed_extensions))}"
        )

    try:
        img = _load_image_safely(raw_bytes, filename)
        w, h = img.size
        if w < 16 or h < 16:
            return False, f"Image too small: {w}×{h}px. Minimum is 16×16."
        if w > 8192 or h > 8192:
            return False, f"Image too large: {w}×{h}px. Maximum is 8192×8192."
        return True, ""
    except ValueError as e:
        return False, str(e)


def get_image_metadata(raw_bytes: bytes, filename: str = "upload") -> dict:
    """Return basic metadata about the image without full preprocessing."""
    try:
        buf = io.BytesIO(raw_bytes)
        img = Image.open(buf)
        return {
            "width":  img.width,
            "height": img.height,
            "mode":   img.mode,
            "format": img.format or "UNKNOWN",
            "size_bytes": len(raw_bytes),
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_image_safely(raw_bytes: bytes, filename: str) -> "Image.Image":
    """Attempt to open raw bytes as a PIL Image, trying multiple strategies."""
    buf = io.BytesIO(raw_bytes)

    # Strategy 1: Direct PIL open
    try:
        img = Image.open(buf)
        img.load()        # force full decode to catch truncation errors
        return img
    except Exception:
        pass

    # Strategy 2: HEIC via pillow-heif plugin (installed separately)
    ext = os.path.splitext(filename.lower())[1]
    if ext in (".heic", ".heif"):
        try:
            from pillow_heif import register_heif_opener   # type: ignore
            register_heif_opener()
            buf.seek(0)
            img = Image.open(buf)
            img.load()
            return img
        except ImportError:
            raise ValueError(
                "HEIC files require the pillow-heif package. "
                "Install with: pip install pillow-heif"
            )
        except Exception as exc:
            raise ValueError(f"Failed to decode HEIC file: {exc}")

    # Strategy 3: OpenCV decode (handles some corrupt headers PIL can't)
    if HAS_CV2:
        try:
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if cv_img is not None:
                # Convert BGR→RGB and wrap in PIL
                rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb, mode="RGB")
        except Exception:
            pass

    raise ValueError(
        "Unable to decode the uploaded file as an image. "
        "Please upload a valid JPG, PNG, WebP, or similar image."
    )


def _to_rgb(img: "Image.Image") -> "Image.Image":
    """Ensure image is in RGB mode (handles RGBA, L, P, CMYK, etc.)."""
    if img.mode == "RGBA":
        # Composite onto white background to remove alpha channel
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _strip_exif(img: "Image.Image") -> "Image.Image":
    """Remove EXIF metadata (fixes rotation issues and strips sensitive data)."""
    try:
        # Auto-rotate based on EXIF orientation, then strip all EXIF
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    # Create a fresh image without any metadata
    clean = Image.new("RGB", img.size)
    clean.paste(img)
    return clean


def _letterbox_resize(
    img: "Image.Image",
    target_size: Tuple[int, int],
) -> "Image.Image":
    """
    Resize image to fit within target_size while preserving aspect ratio.
    Pads with grey (128, 128, 128) to fill the remaining space (letterbox).
    """
    target_w, target_h = target_size
    orig_w, orig_h = img.size

    # Compute scale factor
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # High-quality downsampling
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Create grey canvas and paste centered
    canvas = Image.new("RGB", (target_w, target_h), (128, 128, 128))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return canvas
