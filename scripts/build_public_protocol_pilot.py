import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_image.config import load_config
from agentic_image.data.public_protocol import build_public_contract_records
from agentic_image.runtime.components import Components


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--tasks", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--families", type=int, default=32)
parser.add_argument("--resume", action="store_true")
arguments = parser.parse_args()
components = Components(load_config(arguments.config))
path = build_public_contract_records(
    components, arguments.tasks, arguments.output, arguments.families, arguments.resume
)
print(path, flush=True)
