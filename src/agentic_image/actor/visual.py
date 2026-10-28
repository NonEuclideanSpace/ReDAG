from pathlib import Path
from urllib.parse import quote

import torch

from agentic_image.learning.transaction import atomic_torch_save

from .precision import VISUAL_PRECISION_PROFILE


def image_features(model, tensors, artifact_ids, config, device):
    grid = tensors["image_grid_thw"]
    pixels = tensors["pixel_values"]
    model.visual.eval()
    if not config.image_cache_directory or len(artifact_ids) != grid.shape[0]:
        result = model.get_image_features(pixels.to(device), grid, return_dict=True)
        return torch.cat(result.pooler_output), ()
    directory = (
        Path(config.image_cache_directory)
        / f"{config.revision}_{config.max_image_pixels}_{VISUAL_PRECISION_PROFILE}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    offset = 0
    features, paths = [], []
    for artifact_id, image_grid in zip(artifact_ids, grid):
        patch_count = int(image_grid.prod())
        grid_values = image_grid.cpu().tolist()
        signature = "_".join(str(value) for value in grid_values)
        path = directory / f"{quote(artifact_id, safe='')}_{signature}.pt"
        identity = {
            "artifact_id": artifact_id,
            "model_revision": config.revision,
            "max_image_pixels": config.max_image_pixels,
            "precision_profile": VISUAL_PRECISION_PROFILE,
            "grid_thw": grid_values,
        }
        if path.is_file():
            stored = torch.load(path, map_location="cpu", weights_only=True)
            if stored["identity"] != identity:
                raise ValueError("A frozen image cache entry belongs to different processor inputs")
            feature = stored["image_features"]
        else:
            image_pixels = pixels[offset : offset + patch_count]
            feature = (
                model.get_image_features(
                    image_pixels.to(device), image_grid[None], return_dict=True
                )
                .pooler_output[0]
                .detach()
                .cpu()
            )
            atomic_torch_save(
                path,
                {
                    "identity": identity,
                    "grid_thw": image_grid.cpu().clone(),
                    "pixel_values": image_pixels.detach().cpu().clone(),
                    "image_features": feature,
                },
            )
        features.append(feature.to(device))
        paths.append(str(path.resolve()))
        offset += patch_count
    if offset != pixels.shape[0]:
        raise ValueError("Image patch offsets do not cover the actual processor pixels")
    return torch.cat(features), tuple(paths)
