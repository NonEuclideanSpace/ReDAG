from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AtomScore:
    atom_id: str
    value: float
    known: bool
    threshold: float = 0.8
    exact_pass: bool | None = None
    evidence: tuple[str, ...] = ()

    def __post_init__(self):
        if not np.isfinite(self.value) or not 0 <= self.value <= 1:
            raise ValueError("Atom scores must be finite probabilities")

    @property
    def passed(self):
        return self.known and (
            self.value >= self.threshold if self.exact_pass is None else self.exact_pass
        )


@dataclass(frozen=True)
class RewardResult:
    reward: float
    success: bool
    dense: float
    regression: float
    preservation: float
    cost: float
    atoms: tuple[AtomScore, ...]
    terminal_reason: str

    def to_dict(self):
        return asdict(self)


def dense_score(atoms):
    if not atoms:
        raise ValueError("Evaluation requires at least one fixed atom")
    values = np.asarray([atom.value if atom.known else 0.0 for atom in atoms])
    return float(0.7 * values.mean() + 0.3 * values.min())


def preservation_score(
    source, output, mask, source_features=None, output_features=None, patch_coverage=None
):
    source_raw = np.asarray(source)
    output_raw = np.asarray(output)
    source = source_raw.astype(np.float32)
    output = output_raw.astype(np.float32)
    if source.shape != output.shape or source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError("Preservation needs registered RGB canvases")
    if source_raw.dtype == np.uint8 or source.max() > 1:
        source = source / 255
    if output_raw.dtype == np.uint8 or output.max() > 1:
        output = output / 255
    if (
        not np.isfinite(source).all()
        or not np.isfinite(output).all()
        or np.any((source < 0) | (source > 1))
        or np.any((output < 0) | (output > 1))
    ):
        raise ValueError("Preservation images must contain finite RGB values in [0, 1]")
    if mask is None:
        return 1.0
    mask = np.asarray(mask, dtype=np.float32)
    if (
        mask.shape != source.shape[:2]
        or not np.isfinite(mask).all()
        or np.any((mask < 0) | (mask > 1))
    ):
        raise ValueError("Protection mask does not match the source canvas")
    area = float(mask.sum())
    if area == 0:
        return 1.0
    pixel_distance = float((np.abs(output - source) * mask[..., None]).sum() / (3 * area))
    pixel_score = float(np.exp(-pixel_distance / 0.05))
    if patch_coverage is None or not np.any(np.asarray(patch_coverage) >= 0.5):
        return pixel_score
    if source_features is None or output_features is None:
        raise ValueError("Protected DINO patches require source and output features")
    left = np.asarray(source_features, dtype=np.float32)
    right = np.asarray(output_features, dtype=np.float32)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Invalid preservation features")
    norms = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    if np.any(norms <= 0):
        raise ValueError("Zero-norm preservation features")
    cosine = np.clip((left * right).sum(-1) / norms, -1, 1)
    selected = np.asarray(patch_coverage).reshape(-1) >= 0.5
    feature_score = float(((1 + cosine.reshape(-1)[selected]) / 2).mean())
    return 0.5 * pixel_score + 0.5 * feature_score


def terminal_reward(
    atoms,
    *,
    renders,
    inspects,
    actions,
    invalids,
    preservation=1.0,
    source_atoms=(),
    protected_atom_ids=(),
    image_present=True,
    terminal_reason="finished",
    budgets=(3, 2, 10),
):
    atoms = tuple(atoms)
    if invalids >= 2 or not image_present or terminal_reason in {"abandoned", "protocol_failure"}:
        return RewardResult(-1.0, False, 0.0, 0.0, 1.0, 0.0, atoms, terminal_reason)
    if not 0 <= preservation <= 1:
        raise ValueError("Preservation must lie in [0, 1]")
    source = {atom.atom_id: atom for atom in source_atoms}
    final = {atom.atom_id: atom for atom in atoms}
    protected = [key for key in protected_atom_ids if key in source and source[key].passed]
    if any(key not in final for key in protected):
        raise ValueError("A preserved requirement disappeared from terminal evaluation")
    regression = (
        sum(not final[key].passed for key in protected) / len(protected) if protected else 0.0
    )
    denominator = budgets[0] + 0.1 * budgets[1] + 0.01 * budgets[2]
    cost = (renders + 0.1 * inspects + 0.01 * actions) / denominator
    if not 0 <= cost <= 1 + 1e-9:
        raise ValueError("Logical execution cost exceeded the episode budget")
    dense = dense_score(atoms)
    success = all(atom.passed for atom in atoms)
    reward = float(success) + 0.3 * dense - 0.15 * regression
    reward -= 0.05 * (1 - preservation) + 0.1 * cost + 0.02 * invalids
    return RewardResult(
        reward, success, dense, regression, preservation, cost, atoms, terminal_reason
    )
