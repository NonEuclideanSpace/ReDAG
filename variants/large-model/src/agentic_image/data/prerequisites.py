import json
from pathlib import Path


def require_qualified_renderer(components, renderer_version, sampler_id):
    policy = components.config["models"].get("renderer_policy")
    if not policy or not Path(policy).exists():
        raise ValueError("Repair data requires a saved renderer SFT policy checkpoint")
    selection_path = components.directory / "sampler_selection.json"
    if not selection_path.is_file():
        raise ValueError("Run the sampler comparison and selection before collecting repair labels")
    selection = json.loads(selection_path.read_text())
    if selection.get("status") != "accepted" and not selection.get("accepted", False):
        raise ValueError("The selected sampler has not passed its recorded quality comparison")
    valid_ids = {
        value for value in (selection.get("sampler_id"), selection.get("candidate")) if value
    }
    if valid_ids and sampler_id not in valid_ids:
        raise ValueError("Requested repair sampler does not match the accepted sampler selection")
    if not renderer_version:
        raise ValueError("A fixed renderer version is required")
    configured = components.config["renderer"]
    for field in ("steps", "eta", "cfg_scale"):
        if field in selection and selection[field] != configured[field]:
            raise ValueError(f"Selected sampler {field} differs from the runtime configuration")
    return selection
