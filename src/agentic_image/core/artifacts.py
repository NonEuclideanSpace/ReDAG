from __future__ import annotations

import hashlib
import io
import json
import os
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from .geometry import Frame, normalize_image
from .types import FrozenModel


def canonical_json(value: Any) -> bytes:
    def convert(item):
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    return unicodedata.normalize(
        "NFC",
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=convert,
        ),
    ).encode("utf-8")


def content_id(kind: str, payload: bytes) -> str:
    return f"{kind}:{hashlib.sha256(payload).hexdigest()}"


def atomic_write(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ArtifactRef(FrozenModel):
    artifact_id: str
    value_id: str
    kind: Literal["rgb", "array", "json", "text"]
    relative_path: str
    dtype: str | None = None
    shape: tuple[int, ...] = ()
    frame: Frame | None = None
    producer_id: str | None = None
    input_slots: tuple[tuple[str, int, str], ...] = ()
    stamps: dict[str, Any] = {}
    metadata: dict[str, Any] = {}


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        (self.root / "values").mkdir(parents=True, exist_ok=True)
        (self.root / "refs").mkdir(parents=True, exist_ok=True)

    def _store(
        self,
        kind: str,
        payload: bytes,
        raw_identity: bytes,
        extension: str,
        dtype: str | None = None,
        shape: tuple[int, ...] = (),
        frame: Frame | None = None,
        producer_id: str | None = None,
        input_slots: tuple[tuple[str, int, str], ...] = (),
        stamps: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        header = canonical_json({"kind": kind, "dtype": dtype, "shape": shape, "frame": frame})
        value_id = content_id("value", header + b"\0" + raw_identity)
        relative_path = f"values/{value_id.split(':')[1]}.{extension}"
        path = self.root / relative_path
        if not path.exists():
            atomic_write(path, payload)
        identity = {
            "value_id": value_id,
            "producer": producer_id,
            "slots": input_slots,
            "stamps": stamps or {},
            "metadata": metadata or {},
        }
        artifact_id = content_id("artifact", canonical_json(identity))
        ref = ArtifactRef(
            artifact_id=artifact_id,
            value_id=value_id,
            kind=kind,
            relative_path=relative_path,
            dtype=dtype,
            shape=shape,
            frame=frame,
            producer_id=producer_id,
            input_slots=input_slots,
            stamps=stamps or {},
            metadata=metadata or {},
        )
        ref_path = self.root / "refs" / f"{artifact_id.split(':')[1]}.json"
        if not ref_path.exists():
            atomic_write(ref_path, canonical_json(ref))
        return ref

    def put_image(self, image: Image.Image | np.ndarray | str | Path, **kwargs) -> ArtifactRef:
        if isinstance(image, (str, Path)):
            with Image.open(image) as loaded:
                image = normalize_image(loaded)
        else:
            image = normalize_image(image)
        array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        output = io.BytesIO()
        image.save(output, format="PNG", compress_level=1)
        return self._store(
            "rgb",
            output.getvalue(),
            array.tobytes(),
            "png",
            dtype="uint8",
            shape=array.shape,
            **kwargs,
        )

    def put_array(self, array: np.ndarray, **kwargs) -> ArtifactRef:
        array = np.asarray(array)
        if array.dtype.kind not in "biufc" or array.dtype.hasobject:
            raise ValueError("Only numeric arrays may be stored")
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite tensors cannot be stored")
        array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
        output = io.BytesIO()
        np.save(output, array, allow_pickle=False)
        return self._store(
            "array",
            output.getvalue(),
            array.tobytes(),
            "npy",
            dtype=array.dtype.str,
            shape=array.shape,
            **kwargs,
        )

    def put_json(self, value: Any, **kwargs) -> ArtifactRef:
        payload = canonical_json(value)
        return self._store("json", payload, payload, "json", **kwargs)

    def put_text(self, value: str, **kwargs) -> ArtifactRef:
        payload = unicodedata.normalize("NFC", value).encode("utf-8")
        return self._store("text", payload, payload, "txt", **kwargs)

    def get_ref(self, artifact_id: str) -> ArtifactRef:
        prefix, separator, identity = artifact_id.partition(":")
        if (
            prefix != "artifact"
            or not separator
            or len(identity) != 64
            or any(char not in "0123456789abcdef" for char in identity)
        ):
            raise KeyError(f"Invalid artifact ID: {artifact_id}")
        path = self.root / "refs" / f"{identity}.json"
        if not path.is_file():
            raise KeyError(f"Unknown artifact: {artifact_id}")
        return ArtifactRef.model_validate_json(path.read_bytes())

    def path(self, artifact_id: str) -> Path:
        ref = self.get_ref(artifact_id)
        path = (self.root / ref.relative_path).resolve()
        if not path.is_relative_to(self.root / "values"):
            raise ValueError("Artifact payload path escapes its store")
        return path

    def get_image(self, artifact_id: str) -> Image.Image:
        if self.get_ref(artifact_id).kind != "rgb":
            raise TypeError("Artifact is not an RGB image")
        with Image.open(self.path(artifact_id)) as image:
            return image.convert("RGB")

    def get_array(self, artifact_id: str) -> np.ndarray:
        if self.get_ref(artifact_id).kind != "array":
            raise TypeError("Artifact is not an array")
        return np.load(self.path(artifact_id), allow_pickle=False)

    def get_json(self, artifact_id: str) -> Any:
        if self.get_ref(artifact_id).kind != "json":
            raise TypeError("Artifact is not JSON")
        return json.loads(self.path(artifact_id).read_bytes())

    def get_text(self, artifact_id: str) -> str:
        if self.get_ref(artifact_id).kind != "text":
            raise TypeError("Artifact is not text")
        return self.path(artifact_id).read_text()
