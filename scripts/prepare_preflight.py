import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_image.core.artifacts import ArtifactStore
from agentic_image.core.types import (
    CheckSpec,
    PublicCheckAtom,
    PublicTask,
    RequestSpan,
    UserRequest,
)
from agentic_image.data.io import write_jsonl


artifacts = ArtifactStore("data/artifacts")
source = artifacts.put_image(
    "third_party/GroundingDINO/.asset/cat_dog.jpeg",
    metadata={"purpose": "deployment_probe", "role": "public_source"},
)
image = artifacts.get_image(source.artifact_id)
checks = json.loads(Path("deployment/visual-tools-cpu.json").read_text())
box = next(row["box"] for row in checks["detection"]["detections"] if row["label"] == "cat")
reference = artifacts.put_image(
    image.crop(tuple(value * size for value, size in zip(box, [image.width, image.height] * 2))),
    producer_id=source.artifact_id,
    metadata={"purpose": "deployment_probe", "role": "public_appearance_reference"},
)
text = (
    "Keep a cat on the left and a dog on the right. "
    "Preserve the cat's appearance shown in the reference image."
)
task = PublicTask(
    task_id="deployment-probe:cat-dog",
    family_id="deployment-probe:groundingdino-example",
    split_id="deployment-probe",
    source_dataset="deployment-probe",
    license_record={"source": "GroundingDINO/.asset/cat_dog.jpeg", "purpose": "deployment_only"},
    task_mode="edit",
    user_request=UserRequest(
        text=text,
        source_image_id=source.artifact_id,
        appearance_image_ids=(reference.artifact_id,),
        spans=(RequestSpan(span_id="span:request", start=0, end=len(text), text=text),),
    ),
    check_spec=CheckSpec(
        extractor_version="deployment-probe-v1",
        atoms=(
            PublicCheckAtom(
                check_id="check:request",
                raw_span_ids=("span:request",),
                predicate="The image satisfies the complete original request.",
                expected_value=text,
                public_reference_ids=(reference.artifact_id,),
            ),
        ),
    ),
)
write_jsonl("data/preflight/public_tasks.jsonl", [task.model_dump(mode="json")])
print("PREFLIGHT_INPUT_READY", task.task_id)
