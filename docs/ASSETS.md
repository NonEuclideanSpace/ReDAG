# Research assets and restoration

**Bulk asset publication is deferred.** Documents, explanations, and code are being published first in separate pushes. The original backup remains the source of truth for the experimental assets. No uploaded release pack or restoration from a real GitHub release asset has been verified yet. Do not interpret a partially present catalog as a completed upload.

The planned release is `research-assets-20261028` in `NonEuclideanSpace/ReDAG`. Large pretrained base weights are excluded. Project adapter checkpoints, optimizer state, experiment metadata, datasets, generated images, logs, and reusable environment archives are retained where selected by the asset manifests. A checkpoint containing adapter and optimizer state is not treated as an excluded pretrained base model.

Original asset bytes take priority over editorial cleanup. Chinese text, comments, historical paths, and original document wording inside preserved research assets remain unchanged so that hashes and experimental evidence retain their meaning. English explanations and translations are provided separately.

## Catalog layout

The completed publication uses these files under `provenance/assets/`:

| Catalog | Purpose |
| --- | --- |
| `project-latest.jsonl.gz` | The latest selected project path, file hash, size, source archive, and disposition. |
| `project-versions.jsonl.gz` | Historical project versions, retained for provenance. The default restore selects the latest catalog. |
| `direct-files.jsonl.gz` | Selected original backup and original-workspace files, including opaque environment archives. |
| `objects.jsonl.gz` | Maps each content hash to its byte count, release pack, and `objects/<sha256>` member. |
| `packs.json` | Repository, release tag, pack names, sizes, hashes, and GitHub asset records. |

Each release pack is a gzip-compressed tar archive containing objects named by SHA-256. Identical bytes share an object. An opaque file larger than the pack target may be stored as an ordered list of byte chunks. Restoration joins those chunks and verifies the full original file hash; it does not unpack or execute the archive being restored.

Catalogs and packs may be under construction until publication finishes. A missing object or pack entry is an incomplete publication, not permission to substitute a different file. The restore helper stops when a selected object lacks a complete pack record.

## Restore into a separate directory

`scripts/restore_release_assets.py` uses only the Python standard library. It never executes project code or loads pickle/checkpoint objects. Its default is a dry run with no downloads or filesystem writes. A destination is required, and restoring over the repository itself or one of its parent directories is rejected.

Inspect a small project selection first:

```sh
python scripts/restore_release_assets.py \
  --repository-dir /path/to/ReDAG \
  --destination /path/to/restored-project \
  --cache-dir /path/to/redag-asset-cache \
  --scope project \
  --prefix data/calibration
```

After the release catalogs and packs are available, apply that selection:

```sh
python scripts/restore_release_assets.py \
  --repository-dir /path/to/ReDAG \
  --destination /path/to/restored-project \
  --cache-dir /path/to/redag-asset-cache \
  --scope project \
  --prefix data/calibration \
  --download \
  --apply
```

`--download` uses an already authenticated GitHub CLI (`gh`) to request the exact pack name from the catalog's release. Access to this repository is required. Without `--download`, needed packs must already be in `<cache-dir>/packs/`; previously verified object bytes can also be reused from `<cache-dir>/objects/`.

Omit `--prefix` to select the entire scope. `--scope backup` restores selected paths relative to the original external backup. `--scope original-workspace` restores selected paths relative to the original workspace. Each scope should normally use its own destination. The `project` scope restores experimental project paths and can include historical project source; keep it separate from the maintained repository code.

The report lists selected file bytes, unique object bytes, and referenced pack bytes. The cache retains both packs and extracted objects, so a full restoration needs room for the cache as well as the output. Prefix selections may still need a whole pack when an object shares that pack with unrelated files.

## Integrity and overwrite behavior

Downloaded packs are checked against their catalog SHA-256 and byte count. Extracted objects are checked independently. Each output is assembled into a temporary file, checked against the complete file hash and length, and atomically moved into place. The original source files are never changed.

An existing byte-identical output is reused. A different output causes an applied restore to stop unless `--overwrite` is explicit. A dry run reports those conflicts. Directories are not recursively deleted or replaced with files.

The helper rejects traversal, unsafe tar members, symbolic-link ancestors, external link targets, cycles, and incompatible case or Unicode-normalization aliases. Legacy absolute links under `/workspace/agentic-image-pilot-small/` are converted to relative links that stay inside the chosen destination. A narrow prefix may leave an internal link dangling until its target is restored. Ambiguous relative targets that traverse upward after a named component are rejected instead of silently changing their meaning.

Only the `release_object`, `directory`, and `restore_link` dispositions are materialized. Exclusions remain listed in the catalogs. Model downloads and other excluded dependencies must be supplied separately.

Restoration preserves file contents; it does not rewrite machine-specific paths inside code or metadata, recreate the original operating system, or establish that training can resume on new hardware. The implementation and GPU-validation boundaries are documented in `IMPLEMENTATIONS.md` and `VALIDATION.md`.
