import sys
from pathlib import Path


root = Path(sys.argv[1])
for path in (root / "groundingdino" / "models" / "GroundingDINO" / "csrc").rglob("*"):
    if path.suffix not in {".cpp", ".cu", ".h", ".cuh"}:
        continue
    source = path.read_text()
    updated = source.replace(".type().is_cuda()", ".is_cuda()")
    updated = updated.replace(".scalar_type().is_cuda()", ".is_cuda()")
    updated = updated.replace("value.type()", "value.scalar_type()")
    updated = updated.replace(".data<", ".data_ptr<")
    if updated != source:
        path.write_text(updated)
