from contextlib import contextmanager
import threading

import torch
import torch.nn.functional as F


VISUAL_PRECISION_PROFILE = "qwen-vision-fp32-layernorm-v2"
_MATMUL_PRECISION_LOCK = threading.RLock()


@contextmanager
def actor_matmul_precision():
    with _MATMUL_PRECISION_LOCK:
        previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            yield
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous


class FP32LayerNorm(torch.nn.LayerNorm):
    def forward(self, inputs):
        output = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.to(inputs.dtype)


def configure_visual_layer_norms(vision):
    for name, module in tuple(vision.named_modules()):
        if not name or not isinstance(module, torch.nn.LayerNorm):
            continue
        replacement = FP32LayerNorm(
            module.normalized_shape,
            eps=module.eps,
            elementwise_affine=module.elementwise_affine,
            bias=module.bias is not None,
            device="meta",
            dtype=torch.float32,
        )
        replacement.weight = module.weight
        replacement.bias = module.bias
        replacement.train(module.training)
        parent, _, child = name.rpartition(".")
        setattr(vision.get_submodule(parent) if parent else vision, child, replacement)
