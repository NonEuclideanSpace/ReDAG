import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_image.checks.tools import serve_tools
from agentic_image.core.artifacts import ArtifactStore


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/tools.yaml")
parser.add_argument("--device")
parser.add_argument("--port", type=int)
arguments = parser.parse_args()
config = yaml.safe_load(Path(arguments.config).read_text())
if arguments.device is not None:
    config["device"] = arguments.device
if arguments.port is not None:
    config["port"] = arguments.port
serve_tools(ArtifactStore(config["artifacts"]), config, port=config.get("port", 8091))
