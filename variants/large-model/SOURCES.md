# Sources and implementation provenance

The project implements `plan/Agentic_Image_Generation_v3.pdf`. The four-card profile changes deployment and accumulation only. The requested local stage excludes remote training and omits the document's file-hash auditing workflow.

Model weights and datasets are downloaded from their original providers and are not bundled. Their separate model, dataset, and image-source terms continue to apply.

## Runtime dependencies

| Component | Fixed upstream |
| --- | --- |
| Qwen3.8-27B controller | [Model snapshot](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) |
| FLUX.2-klein-base-9B and its codecs | [Model snapshot](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/tree/32773329fbe7e81a90ef971740e8ba4b0364ecf3) |
| Qwen3-VL-32B-Instruct judge | [Model snapshot](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/tree/1e6810acf5d8ebd189b4e75590877b64358e454c) |
| Transformers 5.12.1 | [ddb849a](https://github.com/huggingface/transformers/tree/ddb849abe009d1089e6c691bfc897f27211c663c) |
| PEFT 0.20.0 | [a5526d2](https://github.com/huggingface/peft/tree/a5526d27a9d47d1e8264d5e1b1f96c0fdc79464e) |
| Diffusers 0.38.0 | [275869d](https://github.com/huggingface/diffusers/tree/275869dcae4ebcfee6a80253fdabc56033335020) |
| bitsandbytes 0.50.2 | [08a9956](https://github.com/bitsandbytes-foundation/bitsandbytes/tree/08a9956b939857ae2c4fbd56753960cc370fa7cf) |
| Flash Linear Attention 0.6.0 | [413ad2c](https://github.com/fla-org/flash-linear-attention/tree/413ad2c655ece57b94bc9ddc51fdbd43cf09d984) |
| causal-conv1d 1.7.0 | [cd81f04](https://github.com/Dao-AILab/causal-conv1d/tree/cd81f0413cad2fc1e6f17e785ac39f59aae690cd) |

`render/encoders.py` and `render/attention.py` adapt the frozen codec, text-template/layer assembly, token packing, and attention projection ordering from Hugging Face Diffusers. Relevant upstream files are `pipeline_flux2_klein.py` and `transformer_flux2.py` at the revision above. The adaptations add explicit region/reference layouts, strict text lengths, standalone frozen services, bounded caches, query chunking, and FP32 attention operations. Diffusers is Apache-2.0; its license is retained in `THIRD_PARTY_LICENSES/Apache-2.0.txt`.

`actor/cache.py` specializes the fixed Transformers hybrid cache so recurrent state remains FP32. The controller uses the official Qwen3.5-family model class and native visual feature interfaces. The actor and renderer both use the original PEFT adapter APIs; neither merges adapters into quantized bases. Transformers and PEFT are Apache-2.0.

The stochastic transition, stratified score replay, head VJP, hierarchy, transaction, and typed runtime are implemented locally from the supplied plan. The plan cites [flow_grpo](https://github.com/yifan123/flow_grpo/tree/879042cf5707f8b90daa98d147d7deac2317c5da), [Tencent UniRL](https://github.com/Tencent-Hunyuan/UniRL/tree/8cfbac17765004afb68fef08a4e011a4c8daf555), [G-U-N UniRL](https://github.com/G-U-N/UniRL/tree/53c28c9ae209583c358e2afc67d0a97f3a636afe), and [RePlan](https://github.com/JIA-Lab-research/RePlan/tree/6c9b12f0c619cc8ca6f50b536b24aff377525f8c) as algorithmic and implementation references. This package does not import their training loops or substitute their objectives for the specified two-score objective.

## Visual tools and data

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO/tree/856dde20aee659246248e20734ef9ba5214f5e44): FP32 detector in the separate CUDA 12.4 environment; the setup script applies the documented PyTorch scalar-type compatibility patch.
- [SAM2](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4): SAM2.1-small box-prompted masks.
- [DINOv2](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8): frozen ViT-B/14 identity and preservation features.
- [Tesseract 5.5.0](https://github.com/tesseract-ocr/tesseract/tree/5.5.0) and [tessdata_fast 4.1.0](https://github.com/tesseract-ocr/tessdata_fast/tree/4.1.0): explicitly routed English and numeric OCR.
- [MagicBrush](https://huggingface.co/datasets/osunlp/MagicBrush/tree/6c254560879df7851233ea0fd2ae2a62764d0ef2): official splits, image sessions, and independent/iterative edit evaluation.
- [OmniEdit-Filtered-1.2M](https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M/tree/b6eec40a8a6994aa380497e33b1964742d7e028b): bounded streaming import with the specified score filter.

External repositories are installed without removing their original notices. Independent paired annotations, calibration decisions, and blinded quality judgments must be supplied as actual reviewed records.

## Official evaluation

- [GenEval](https://github.com/djghosh13/geneval/tree/af4902f24d3ca90ebbb446dd9891a59e0f82725f): original detector evaluation and category macro-average.
- [GenEval2](https://github.com/facebookresearch/GenEval2/tree/a6e82d2289e8d418f27f0adee77908b07060eea3): original `soft_tifa_gm` with [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b), separate from the internal judge.
- [MagicBrush image evaluation](https://github.com/OSU-NLP-Group/MagicBrush/blob/d2d7f84d71ff23154e3f16f0c8f19392782bd201/evaluation/image_eval.py): official DINOv1 evaluation, separate from internal DINOv2 preservation.
