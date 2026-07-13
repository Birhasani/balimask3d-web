"""Image preparation and Zero123++ multiview grid handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


class ImagePreprocessingError(ValueError):
    """Raised when an input or generated multiview image is invalid."""


def load_input_image(path: Path) -> Image.Image:
    """Load an input image eagerly so file errors occur before GPU inference."""

    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise ImagePreprocessingError(f"input image does not exist or is not a file: {image_path}")
    try:
        with Image.open(image_path) as image:
            return image.convert("RGBA")
    except (UnidentifiedImageError, OSError) as error:
        raise ImagePreprocessingError(f"could not decode input image '{image_path}': {error}") from error


def preprocess_image(
    input_path: Path,
    output_path: Path,
    *,
    remove_background: bool,
    foreground_ratio: float = 0.85,
) -> Image.Image:
    """Apply the official background path or preserve notebook-ready RGBA input."""

    image = load_input_image(input_path)
    if remove_background:
        try:
            import rembg

            from src.utils.infer_util import remove_background as remove_bg
            from src.utils.infer_util import resize_foreground
        except ImportError as error:
            raise ImagePreprocessingError(
                "background removal requires the repository's rembg dependencies"
            ) from error
        image = remove_bg(image, rembg.new_session())
        image = resize_foreground(image, foreground_ratio)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise ImagePreprocessingError(f"failed to save processed image: {destination}")
    return image


def split_multiview_grid(
    grid_image: Image.Image,
    output_paths: tuple[Path, ...],
) -> tuple[Image.Image, ...]:
    """Split Zero123++'s 3-row by 2-column grid in row-major model view order."""

    if len(output_paths) != 6:
        raise ImagePreprocessingError("exactly six multiview output paths are required")
    array = np.asarray(grid_image.convert("RGB"), dtype=np.uint8)
    height, width = array.shape[:2]
    if height % 3 or width % 2:
        raise ImagePreprocessingError(
            f"multiview grid must be divisible into 3x2 tiles, received {width}x{height}"
        )

    tile_height, tile_width = height // 3, width // 2
    images: list[Image.Image] = []
    index = 0
    for row in range(3):
        for column in range(2):
            tile = Image.fromarray(
                array[
                    row * tile_height : (row + 1) * tile_height,
                    column * tile_width : (column + 1) * tile_width,
                ]
            )
            path = output_paths[index]
            path.parent.mkdir(parents=True, exist_ok=True)
            tile.save(path, format="PNG")
            images.append(tile)
            index += 1
    return tuple(images)
