import torch


def make_fp32_cache(config):
    from transformers.cache_utils import DynamicCache, LinearAttentionLayer

    class FP32LinearAttentionLayer(LinearAttentionLayer):
        def lazy_initialization(self, conv_states=None, recurrent_states=None):
            if conv_states is not None:
                super().lazy_initialization(conv_states=conv_states)
            if recurrent_states is not None:
                if recurrent_states.dtype != torch.float32:
                    raise TypeError("The recurrent GDN cache must receive FP32 states")
                self.recurrent_states = torch.zeros_like(recurrent_states)
                self.is_recurrent_states_initialized = True

    cache = DynamicCache(config=config)
    expected = config.get_text_config().layer_types
    for index, layer_type in enumerate(expected):
        if layer_type == "linear_attention":
            cache.layers[index] = FP32LinearAttentionLayer()
    return cache


def validate_cache_precision(cache):
    for layer in cache.layers:
        if getattr(layer, "is_conv_states_initialized", False):
            if layer.conv_states.dtype != torch.bfloat16:
                raise TypeError("GDN convolution states must use BF16")
        if getattr(layer, "is_recurrent_states_initialized", False):
            if layer.recurrent_states.dtype != torch.float32:
                raise TypeError("GDN recurrent states must use FP32")
        keys = getattr(layer, "keys", None)
        if keys is not None and keys.numel() and keys.dtype != torch.bfloat16:
            raise TypeError("Full-attention cache must use BF16")
