from pathlib import Path

from huggingface_hub import snapshot_download


bert = snapshot_download(
    "google-bert/bert-base-uncased",
    revision="86b5e0934494bd15c9632b12f734a8a67f723594",
    local_dir="models/tools/bert-base-uncased",
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ],
)
source = Path(
    "third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
).read_text()
source = source.replace('"bert-base-uncased"', repr(str(Path(bert).resolve())))
Path("models/tools/grounding_config.py").write_text(source)
