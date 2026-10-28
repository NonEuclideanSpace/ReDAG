from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image, ImageOps

from .types import BoundingBox, FrozenModel


class Frame(FrozenModel):
    frame_id: str
    source_hw: tuple[int, int]
    canvas_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    padding_xy: tuple[int, int]
    scale_xy: tuple[float, float]
    rgb_resample: str = "lanczos"
    coordinate_convention: str = "pixel_edges_integer_centers_half_integer"

    @property
    def matrix(self) -> np.ndarray:
        sx, sy = self.scale_xy
        px, py = self.padding_xy
        return np.array(((sx, 0, px), (0, sy, py), (0, 0, 1)), dtype=np.float64)

    @property
    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.matrix)


def normalize_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("RGB input must be uint8 HWC")
        image = Image.fromarray(image)
    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands() or image.info.get("transparency") is not None:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
        image = Image.alpha_composite(background, rgba)
    return image.convert("RGB")


def make_frame(source_hw: tuple[int, int], canvas_hw: tuple[int, int], frame_id: str) -> Frame:
    source_h, source_w = source_hw
    canvas_h, canvas_w = canvas_hw
    if min(source_h, source_w, canvas_h, canvas_w) <= 0:
        raise ValueError("Image dimensions must be positive")
    scale = min(canvas_w / source_w, canvas_h / source_h)
    resized_w = max(1, min(canvas_w, round(scale * source_w)))
    resized_h = max(1, min(canvas_h, round(scale * source_h)))
    return Frame(
        frame_id=frame_id,
        source_hw=source_hw,
        canvas_hw=canvas_hw,
        resized_hw=(resized_h, resized_w),
        padding_xy=((canvas_w - resized_w) // 2, (canvas_h - resized_h) // 2),
        scale_xy=(resized_w / source_w, resized_h / source_h),
    )


def letterbox(image: Image.Image | np.ndarray, frame: Frame) -> Image.Image:
    image = normalize_image(image)
    if (image.height, image.width) != frame.source_hw:
        raise ValueError("Image dimensions do not match the registered frame")
    height, width = frame.canvas_hw
    resized_h, resized_w = frame.resized_hw
    result = Image.new("RGB", (width, height), (128, 128, 128))
    result.paste(image.resize((resized_w, resized_h), Image.Resampling.LANCZOS), frame.padding_xy)
    return result


def transform_mask(
    mask: np.ndarray, frame: Frame, kind: Literal["binary", "soft"] = "binary"
) -> np.ndarray:
    mask = validate_mask(mask, allow_empty=True)
    if mask.shape != frame.source_hw:
        raise ValueError("Mask dimensions do not match the registered source frame")
    if kind == "binary" and not np.all((mask == 0) | (mask == 1)):
        raise ValueError("Binary masks may only contain zero and one")
    resized_h, resized_w = frame.resized_hw
    mode = Image.Resampling.NEAREST if kind == "binary" else Image.Resampling.BILINEAR
    resized = np.asarray(
        Image.fromarray(mask).resize((resized_w, resized_h), mode), dtype=np.float32
    )
    output = np.zeros(frame.canvas_hw, dtype=np.float32)
    px, py = frame.padding_xy
    output[py : py + resized_h, px : px + resized_w] = resized
    return output


def validate_mask(mask: np.ndarray, allow_empty: bool = False) -> np.ndarray:
    result = np.asarray(mask, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Masks must be finite two-dimensional float values in [0, 1]")
    if not allow_empty and not np.any(result > 0):
        raise ValueError("An explicitly declared region cannot be empty")
    return result


def transform_points(
    points: np.ndarray, frame: Frame, inverse: bool = False, centers_at_zero: bool = False
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Points must be a finite [N, 2] array")
    if centers_at_zero and not inverse:
        points = points + 0.5
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    result = homogeneous @ (frame.inverse if inverse else frame.matrix).T
    result = result[:, :2] / result[:, 2:]
    if centers_at_zero and inverse:
        result -= 0.5
    return result


def transform_box(box: BoundingBox, frame: Frame, inverse: bool = False) -> BoundingBox:
    source_h, source_w = frame.canvas_hw if inverse else frame.source_hw
    target_h, target_w = frame.source_hw if inverse else frame.canvas_hw
    xy = np.array(((box.x0 * source_w, box.y0 * source_h), (box.x1 * source_w, box.y1 * source_h)))
    xy = transform_points(xy, frame, inverse=inverse)
    xy /= np.array((target_w, target_h))
    xy = np.clip(xy, 0, 1)
    return BoundingBox(
        x0=float(xy[0, 0]), y0=float(xy[0, 1]), x1=float(xy[1, 0]), y1=float(xy[1, 1])
    )


def box_mask(box: BoundingBox, canvas_hw: tuple[int, int]) -> np.ndarray:
    height, width = canvas_hw
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    x_coverage = np.maximum(0, np.minimum(x + 1, box.x1 * width) - np.maximum(x, box.x0 * width))
    y_coverage = np.maximum(0, np.minimum(y + 1, box.y1 * height) - np.maximum(y, box.y0 * height))
    return y_coverage[:, None] * x_coverage[None, :]


def pool_membership(
    mask: np.ndarray, patch_size: int = 16, threshold: float = 0.25
) -> tuple[np.ndarray, bool]:
    mask = validate_mask(mask)
    height, width = mask.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Mask dimensions must be divisible by the latent patch size")
    coverage = (
        mask.reshape(height // patch_size, patch_size, width // patch_size, patch_size)
        .mean(axis=(1, 3))
        .reshape(-1)
    )
    membership = coverage >= threshold
    fallback = not bool(membership.any())
    if fallback:
        membership[int(coverage.argmax())] = True
    return membership, fallback


def expand_mask(mask: np.ndarray, pixels: int = 32) -> np.ndarray:
    mask = validate_mask(mask)
    if pixels < 0:
        raise ValueError("Expansion must be nonnegative")
    if pixels == 0:
        return mask.copy()
    padded = np.pad(mask, ((pixels, pixels), (0, 0)))
    vertical = np.lib.stride_tricks.sliding_window_view(padded, 2 * pixels + 1, axis=0).max(axis=-1)
    padded = np.pad(vertical, ((0, 0), (pixels, pixels)))
    return (
        np.lib.stride_tricks.sliding_window_view(padded, 2 * pixels + 1, axis=1)
        .max(axis=-1)
        .astype(np.float32)
    )


def box_iou(first: BoundingBox, second: BoundingBox) -> float:
    intersection = max(0, min(first.x1, second.x1) - max(first.x0, second.x0)) * max(
        0, min(first.y1, second.y1) - max(first.y0, second.y0)
    )
    first_area = (first.x1 - first.x0) * (first.y1 - first.y0)
    second_area = (second.x1 - second.x0) * (second.y1 - second.y0)
    return intersection / max(first_area + second_area - intersection, 1e-12)


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = validate_mask(first)
    second = validate_mask(second)
    if first.shape != second.shape:
        raise ValueError("Masks must be in the same frame")
    return float(np.minimum(first, second).sum() / max(np.maximum(first, second).sum(), 1e-12))
