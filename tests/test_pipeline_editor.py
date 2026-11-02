import torch
from PIL import Image
from redag.graph.schema import ControllerAction
from redag.models.editor import PipelineEditor


class Output:
    def __init__(self, image):
        self.images = [image]


class QwenLike:
    def __init__(self):
        self.calls = []

    def set_progress_bar_config(self, **kwargs):
        pass

    def __call__(self, prompt=None, image=None, negative_prompt=None, true_cfg_scale=1.0, num_inference_steps=40, generator=None):
        self.calls.append(dict(prompt=prompt, image=image, negative_prompt=negative_prompt, true_cfg_scale=true_cfg_scale, steps=num_inference_steps))
        return Output(Image.new('RGBA', (96, 80), (255, 0, 0, 255)))


class FluxLike:
    def __init__(self):
        self.calls = []

    def set_progress_bar_config(self, **kwargs):
        pass

    def __call__(self, prompt=None, image=None, guidance_scale=3.5, num_inference_steps=28, generator=None):
        self.calls.append(dict(prompt=prompt, image=image, guidance_scale=guidance_scale, steps=num_inference_steps))
        return Output(Image.new('RGB', (40, 40), (0, 255, 0)))


def action():
    return ControllerAction('modify_attr', (0.25, 0.25, 0.75, 0.75), 'Make the cup red.')


def test_guidance_free_pipeline_gets_no_negative_prompt():
    pipeline = QwenLike()
    editor = PipelineEditor(pipeline, num_steps=12, cfg_scale=1.0)
    out = editor.render(torch.rand(3, 100, 120), action(), seed=3)
    call = pipeline.calls[-1]
    assert call['true_cfg_scale'] == 1.0 and call['negative_prompt'] is None and call['steps'] == 12
    assert isinstance(call['image'], list) and len(call['image']) == 1
    assert out.shape == (3, 50, 60) and torch.allclose(out[0], torch.ones_like(out[0])) and out[1].abs().max() < 1e-6


def test_guided_pipelines_receive_their_own_guidance_argument():
    qwen = QwenLike()
    PipelineEditor(qwen, cfg_scale=4.0).render(torch.rand(3, 64, 64), action(), seed=1)
    assert qwen.calls[-1]['true_cfg_scale'] == 4.0 and qwen.calls[-1]['negative_prompt'] == ' '
    flux = FluxLike()
    out = PipelineEditor(flux, cfg_scale=2.5).render(torch.rand(3, 64, 64), action(), seed=1)
    assert flux.calls[-1]['guidance_scale'] == 2.5 and not isinstance(flux.calls[-1]['image'], list)
    assert out.shape == (3, 32, 32) and out[1].min() > 0.99
