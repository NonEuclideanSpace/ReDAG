import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys


def file_record(value):
    path = Path(value).resolve()
    stat = path.stat()
    return {"path": str(path), "device": stat.st_dev, "inode": stat.st_ino, "bytes": stat.st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=("py312", "tools"), required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--marker", required=True)
    args = parser.parse_args()
    if Path("/.restore-check-marker").read_text().strip() != args.marker:
        raise RuntimeError("Restored filesystem marker is absent or different")
    result = {"name": args.name, "executable": sys.executable, "prefix": sys.prefix,
        "base_prefix": sys.base_prefix, "sys_path": sys.path, "python": sys.version,
        "executable_file": file_record(sys.executable), "modules": {}, "gpu_tested": False}
    names = (["torch", "torchvision", "transformers", "diffusers", "peft", "bitsandbytes",
              "causal_conv1d", "fla", "agentic_image"] if args.name == "py312"
             else ["torch", "torchvision", "transformers", "groundingdino", "sam2"])
    for name in names:
        try:
            module = importlib.import_module(name)
            result["modules"][name] = {"status": "imported", "version": str(getattr(module, "__version__", "unknown")),
                "file": file_record(module.__file__)}
        except Exception as error:
            result["modules"][name] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    try:
        import torch
        torch.set_num_threads(1)
        value = torch.arange(9, dtype=torch.float32, device="cpu").reshape(3, 3).requires_grad_()
        loss = (value @ value).sum()
        loss.backward()
        result["cpu_tensor"] = {"status": "passed" if torch.isfinite(value.grad).all().item() else "failed",
            "loss": loss.item(), "grad_sum": value.grad.sum().item(), "device": str(value.device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_device_count": torch.cuda.device_count()}
    except Exception as error:
        result["cpu_tensor"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    if args.name == "tools":
        try:
            from PIL import Image, ImageDraw, ImageFont
            text = "RESTORE CHECK 2026"
            image = Image.new("RGB", (900, 130), "white")
            ImageDraw.Draw(image).text((20, 30), text, font=ImageFont.load_default(size=48), fill="black")
            image.save("/tmp/restore-ocr.png")
            ocr = Path(args.project) / ".runtime/ocr/bin/tesseract"
            data = Path(args.project) / ".runtime/ocr/share/tessdata"
            process = subprocess.run([str(ocr), "/tmp/restore-ocr.png", "stdout", "--tessdata-dir", str(data),
                "-l", "eng", "--psm", "7"], capture_output=True, text=True, timeout=45)
            actual = " ".join(process.stdout.upper().split())
            result["ocr"] = {"status": "passed" if process.returncode == 0 and actual == text else "failed",
                "returncode": process.returncode, "expected": text, "actual": actual,
                "stderr": process.stderr[-4000:], "binary": file_record(ocr),
                "language_data": file_record(data / "eng.traineddata"), "input": "/tmp/restore-ocr.png"}
        except Exception as error:
            result["ocr"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    print("RESTORE_RESULT=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
