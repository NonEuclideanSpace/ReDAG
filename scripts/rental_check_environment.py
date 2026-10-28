import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import bitsandbytes as bnb


started = time.monotonic()
report = {'python': sys.version, 'executable': sys.executable, 'torch_cuda': torch.version.cuda, 'packages': {}, 'checks': {}, 'training': False}
for name in ['torch', 'torchvision', 'transformers', 'diffusers', 'peft', 'bitsandbytes', 'causal-conv1d', 'flash-linear-attention']:
    try:
        report['packages'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        report['packages'][name] = None
report['nvidia_smi'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free,driver_version', '--format=csv'], text=True)
report['disks'] = subprocess.check_output(['df', '-hT', '/workspace', '/model'], text=True)
report['disk_persistence'] = 'Not established from instance mount inventory'
for device in ['cuda:0', 'cuda:1']:
    torch.cuda.reset_peak_memory_stats(device)
    value = torch.linspace(-1, 1, 512 * 512, device=device, dtype=torch.bfloat16).reshape(512, 512)
    linear = bnb.nn.Linear4bit(512, 512, bias=False, compute_dtype=torch.bfloat16, compress_statistics=True, quant_type='nf4')
    linear.weight.data.copy_(torch.eye(512))
    linear = linear.to(device)
    x = value[:4].clone().requires_grad_(True)
    output = linear(x)
    output.float().square().mean().backward()
    report['checks'][device] = {'nf4_forward_finite': bool(output.isfinite().all()), 'nf4_backward_finite': bool(x.grad.isfinite().all()), 'input_grad_norm': float(x.grad.float().norm()), 'peak_bytes': torch.cuda.max_memory_allocated(device), 'capability': torch.cuda.get_device_capability(device)}
    del value, linear, x, output
    torch.cuda.empty_cache()
report['elapsed_seconds'] = time.monotonic() - started
path = Path('/workspace/agentic-image-pilot-small/deployment/rental-environment.json')
path.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
