from collections import Counter, defaultdict
from hashlib import blake2b
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

from .io import read_jsonl, write_jsonl, write_report


class UnionFind:
    def __init__(self, count):
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, item):
        while item != self.parent[item]:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


class HammingIndex:
    def __init__(self):
        self.root = None

    def add(self, value, payload):
        if self.root is None:
            self.root = [value, [payload], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(payload)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [payload], {}]
                return
            node = node[2][distance]

    def query(self, value, radius=6):
        pending = [self.root] if self.root is not None else []
        matches = []
        while pending:
            node = pending.pop()
            distance = (value ^ node[0]).bit_count()
            if distance <= radius:
                matches.extend(node[1])
            pending.extend(
                child
                for edge, child in node[2].items()
                if distance - radius <= edge <= distance + radius
            )
        return matches


def perceptual_hash(image):
    array = np.asarray(
        image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64
    )
    coefficients = dctn(array, type=2, norm="ortho")[:8, :8].flatten()
    bits = coefficients > np.median(coefficients[1:])
    bits[0] = False
    return sum(int(value) << index for index, value in enumerate(bits))


def partition_families(
    input_path,
    output_path,
    artifacts,
    minimums=None,
    feature_fn=None,
    seed=20261027,
    reviewed_pairs_path=None,
):
    minimums = (
        {"development": 200, "calibration": 600, "diagnostic-test": 500}
        if minimums is None
        else dict(minimums)
    )
    if any(count < 0 for count in minimums.values()):
        raise ValueError("Family split quotas must be nonnegative")
    rows = list(read_jsonl(input_path))
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Candidate sample IDs must be unique")
    union = UnionFind(len(rows))
    session_owner, value_owner, image_owner = {}, {}, {}
    image_ids = []
    for index, row in enumerate(rows):
        for group in (
            row.get("session_id"),
            row.get("template_instance_id"),
            *row.get("family_links", []),
        ):
            if group:
                if group in session_owner:
                    union.union(index, session_owner[group])
                session_owner[group] = index
        for image_id in [
            row.get("source_image_id"),
            row["target_image_id"],
            *row.get("appearance_image_ids", []),
        ]:
            if image_id is None:
                continue
            value = artifacts.get_ref(image_id).value_id
            if value in value_owner:
                union.union(index, value_owner[value])
            else:
                value_owner[value] = index
                image_ids.append(image_id)
                image_owner[image_id] = index
    reviews = {}
    if reviewed_pairs_path:
        for row in read_jsonl(reviewed_pairs_path):
            reviews[tuple(sorted((row["left_image_id"], row["right_image_id"])))] = bool(
                row["same_family"]
            )
    index = HammingIndex()
    features = {}
    uncertain = []
    near_matches = 0
    for image_id in image_ids:
        image = artifacts.get_image(image_id)
        fingerprint = perceptual_hash(image)
        for previous in index.query(fingerprint):
            if union.find(image_owner[previous]) == union.find(image_owner[image_id]):
                continue
            pair = tuple(sorted((previous, image_id)))
            same_family = reviews.get(pair)
            if same_family is None and feature_fn is not None:
                for identifier in pair:
                    if identifier not in features:
                        feature = np.asarray(
                            feature_fn(artifacts.get_image(identifier)), dtype=np.float32
                        ).reshape(-1)
                        norm = np.linalg.norm(feature)
                        if not np.isfinite(feature).all() or norm == 0:
                            raise ValueError(
                                "Invalid frozen visual feature during duplicate isolation"
                            )
                        features[identifier] = feature / norm
                similarity = float(np.dot(features[pair[0]], features[pair[1]]))
                same_family = similarity >= 0.98
            if same_family is True:
                union.union(image_owner[previous], image_owner[image_id])
                near_matches += 1
            elif same_family is None:
                uncertain.append(
                    {
                        "left_image_id": previous,
                        "right_image_id": image_id,
                        "reason": "perceptual_candidate_requires_review",
                    }
                )
        index.add(fingerprint, image_id)
    groups = defaultdict(list)
    for row_index in range(len(rows)):
        groups[union.find(row_index)].append(row_index)
    unresolved_roots = {
        union.find(image_owner[pair[side]])
        for pair in uncertain
        for side in ("left_image_id", "right_image_id")
    }
    family_ids = {}
    eligible = []
    official_collision = set()
    for root, members in groups.items():
        identity = "\n".join(sorted(rows[item]["sample_id"] for item in members)).encode()
        family_id = "family:" + blake2b(identity, digest_size=16).hexdigest()
        family_ids[root] = family_id
        official_splits = {rows[item].get("official_split", "train") for item in members}
        if "train" in official_splits and official_splits & {"dev", "test"}:
            official_collision.add(root)
        if official_splits == {"train"} and root not in unresolved_roots:
            eligible.append(root)
    eligible.sort(
        key=lambda root: blake2b(f"{seed}:{family_ids[root]}".encode(), digest_size=16).digest()
    )
    required = sum(minimums.values())
    allocations = {}
    if len(eligible) < required:
        report = {
            "status": "insufficient_independent_families",
            "eligible_families": len(eligible),
            "required_holdout_families": required,
            "unresolved_near_duplicate_pairs": len(uncertain),
            "official_collision_families": len(official_collision),
        }
        write_jsonl(Path(output_path).with_suffix(".duplicate_review.jsonl"), uncertain)
        write_report(Path(output_path).with_suffix(".report.json"), report)
        raise ValueError(
            f"Only {len(eligible)} independent eligible families are available; {required} holdout families are required"
        )
    offset = 0
    for split, count in minimums.items():
        allocations.update({root: split for root in eligible[offset : offset + count]})
        offset += count
    allocations.update({root: "train" for root in eligible[offset:]})
    for row_index, row in enumerate(rows):
        root = union.find(row_index)
        official = row.get("official_split", "train")
        if official in {"dev", "test"}:
            split = f"official-{official}"
        elif root in official_collision:
            split = "quarantine-official-overlap"
        elif root in unresolved_roots:
            split = "quarantine-near-duplicate"
        else:
            split = allocations[root]
        row["family_id"] = family_ids[root]
        row["split_id"] = split
    report = {
        "status": "partitioned",
        "rows": len(rows),
        "families": len(groups),
        "confirmed_near_duplicate_merges": near_matches,
        "unresolved_near_duplicate_pairs": len(uncertain),
        "official_collision_families": len(official_collision),
        "rows_by_split": dict(Counter(row["split_id"] for row in rows)),
    }
    report["families_by_split"] = dict(
        Counter(split for split, family in set((row["split_id"], row["family_id"]) for row in rows))
    )
    write_jsonl(output_path, rows)
    write_jsonl(Path(output_path).with_suffix(".duplicate_review.jsonl"), uncertain)
    write_report(Path(output_path).with_suffix(".report.json"), report)
    return report
