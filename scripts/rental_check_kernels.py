import json
from pathlib import Path
import time

import torch
from causal_conv1d import causal_conv1d_fn
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers.models.qwen3_5 import modeling_qwen3_5


torch.cuda.set_device(1)
torch.manual_seed(11)
started = time.monotonic()
report = {'device': 'cuda:1', 'training': False, 'checks': {}, 'fast_path_available': modeling_qwen3_5.is_fast_path_available}
x = torch.randn(1, 256, 32, device='cuda:1', dtype=torch.bfloat16, requires_grad=True)
weight = torch.randn(256, 4, device='cuda:1', dtype=torch.bfloat16, requires_grad=True)
y = causal_conv1d_fn(x, weight, activation='silu')
y.float().square().mean().backward()
torch.cuda.synchronize()
report['checks']['causal_conv1d'] = {'output_finite': bool(y.isfinite().all()), 'input_grad_finite': bool(x.grad.isfinite().all()), 'weight_grad_finite': bool(weight.grad.isfinite().all()), 'weight_grad_norm': float(weight.grad.float().norm())}
print('CAUSAL_DONE', report['checks']['causal_conv1d'], flush=True)
q, k, v = [torch.randn(1, 32, 2, 128, device='cuda:1', dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
g = torch.full((1, 32, 2), -0.1, device='cuda:1', dtype=torch.float32, requires_grad=True)
beta = torch.full((1, 32, 2), 0.5, device='cuda:1', dtype=torch.bfloat16, requires_grad=True)
output, state = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
output.float().square().mean().backward()
torch.cuda.synchronize()
report['checks']['gated_delta_rule'] = {'output_finite': bool(output.isfinite().all()), 'state_finite': bool(state.isfinite().all()), 'gradients_finite': all(bool(value.grad.isfinite().all()) for value in [q, k, v, g, beta]), 'gradient_norms': {name: float(value.grad.float().norm()) for name, value in zip(['q', 'k', 'v', 'g', 'beta'], [q, k, v, g, beta])}}
report['elapsed_seconds'] = time.monotonic() - started
report['peak_bytes'] = torch.cuda.max_memory_allocated(1)
Path('/workspace/agentic-image-pilot-small/deployment/rental-kernel-check.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
