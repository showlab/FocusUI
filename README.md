<div align="center">
<img src="assets/figures/banner.png?raw=true" width="90%" style="margin-bottom: 10px;">
<hr>
</div>

**TL;DR:** Find a smart way to watch screenshot for VLM.

<p align="center">
  <a href="https://arxiv.org/abs/">
    <img src="https://img.shields.io/badge/arXiv-Paper-red.svg" alt="arXiv">
  </a>
  <a href="https://huggingface.co/collections/">
    <img src="https://img.shields.io/badge/HuggingFace-Models-yellow.svg" alt="HuggingFace">
  </a>
  <a href="https://huggingface.co/datasets/">
    <img src="https://img.shields.io/badge/HuggingFace-Dataset-blue.svg" alt="Dataset">
  </a>
  <a href="https://focusui.github.io">
    <img src="https://img.shields.io/badge/Project-Page-green.svg" alt="Project Page">
  </a>
</p>

<p align="center">
  <b>Mingyu Ouyang</b><sup>1</sup>, <b>Kevin Qinghong Lin</b><sup>2</sup>, <b>Mike Zheng Shou</b><sup>1†</sup>, <b>Hwee Tou Ng</b><sup>1†</sup>
  <br>
  <sup>1</sup>National University of Singapore &nbsp;&nbsp; <sup>2</sup>University of Oxford
  <br>
  <sup>†</sup>Corresponding authors
</p>

<p align="center">
  <img src="assets/figures/1a-teaser@2x.png" alt="FocusUI Teaser" width="80%">
</p>

## Overview ✨

Vision-Language Models (VLMs) have shown remarkable performance in UI grounding tasks, but high-resolution screenshots are tokenized into thousands of visual tokens (e.g., ~4700 for 2K resolution), causing significant computational overhead. In contrast, **humans naturally focus on regions of interest** when interacting with UI. **FocusUI** is an efficient UI grounding framework that selects patches most relevant to the instruction while preserving positional continuity for precise grounding.

### Key Innovations
1. **Query-Guided Visual Token Selection**: Constructs patch-level supervision by fusing instruction-conditioned scores with rule-based UI-graph scores that down-weight large homogeneous regions.
2. **POSPAD (Position-Preserving Padding)**: A novel strategy that compresses each contiguous sequence of dropped visual tokens into a single special marker placed at the sequence's last index, preserving positional continuity crucial for UI grounding.
<p align="center">
  <img src="assets/figures/2-focusui@2x.png" alt="FocusUI Architecture" width="95%">
</p>

## Updates 📣

- [2025/12/29] Project page and code base released.

## Quick Start 🚀

### Installation

```bash
# Clone the repository
git clone https://github.com/showlab/FocusUI.git
cd FocusUI

# Install dependencies
pip install -r requirements.txt
```

### Inference

```python
from focusui.modeling_focusui_qwen25vl import FocusUI_Qwen2_5_VLForConditionalGenerationWithPointer
from transformers import AutoProcessor
import torch

# Load model and processor
model_path = "path/to/focusui-7b"
model = FocusUI_Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
).eval()
processor = AutoProcessor.from_pretrained(model_path)

# Prepare conversation
conversation = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a GUI agent..."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "screenshot.png"},
            {"type": "text", "text": "Click on the search button"}
        ]
    }
]

# Configure visual token selection
model.apply_visual_token_select = True
model.visual_reduct_ratio = 0.5  # Keep 50% of visual tokens

# Run inference
from focusui.inference import inference_focusui_token_select
result = inference_focusui_token_select(
    conversation=conversation,
    model=model,
    tokenizer=processor.tokenizer,
    data_processor=processor,
    topk=3,
)

# Get predicted coordinates
print(f"Top-k points: {result['topk_points']}")
```

## Training 🧠

FocusUI uses a two-stage training process:

### Stage 1: Train Patch Scorer Only

```bash
bash scripts/train/stage_1_ft_focusui_scorer.sh
```

This stage trains only the PatchScorer module while freezing the base VLM.

### Stage 2: Full Fine-tuning

```bash
bash scripts/train/stage_2_ft_focusui.sh
```

This stage fine-tunes the entire model with the trained PatchScorer.

## Evaluation 📊

Run evaluation on grounding benchmarks:

```bash
# ScreenSpot-Pro
python -m evaluation.ss_pro_eval \
    --model_name_or_path path/to/focusui-7b \
    --data_path ./dataset/ScreenSpot-Pro \
    --save_path ./results/ss_pro \
    --visual_reduct_ratio 0.5

# ScreenSpot-V2
python -m evaluation.ss_v2_eval \
    --model_name_or_path path/to/focusui-7b \
    --data_path ./dataset/ScreenSpot-v2_HF \
    --save_path ./results/ss_v2

# UI-Vision
python -m evaluation.ui_vision_eval \
    --model_name_or_path path/to/focusui-7b \
    --data_path ./dataset/ui_benchmarks/ui-vision \
    --save_path ./results/ui_vision

# OSWorld-G
python -m evaluation.os_world_g_eval \
    --model_name_or_path path/to/focusui-7b \
    --data_path ./dataset/OSWorld-G_HF \
    --save_path ./results/osworld_g
```

**Key Evaluation Options**

| Argument | Description | Default |
|----------|-------------|---------|
| `--apply_visual_token_select` | Enable visual token selection | True |
| `--visual_reduct_ratio` | Token retention ratio (1.0 = keep all) | 0.5 |

## Model Zoo 🧩

| Model | Backbone | Parameters | HuggingFace |
|-------|----------|------------|-------------|
| FocusUI-3B | Qwen2.5-VL-3B | 3B | [Coming Soon] |
| FocusUI-7B | Qwen2.5-VL-7B | 7B | [Coming Soon] |
| FocusUI-2B | Qwen3-VL-2B | 2B | [Coming Soon] |



## Citation 📝

If you find FocusUI useful for your research, please cite:

```bibtex
@article{ouyang2025focusui,
  title   = {FocusUI: Efficient UI Grounding via Position-Preserving Visual Token Selection},
  author  = {Ouyang, Mingyu and Lin, Kevin Qinghong and Shou, Mike Zheng and Ng, Hwee Tou},
  year    = {2025},
  journal = {arXiv preprint},
}
```

## Acknowledgements 🙏

FocusUI builds upon [Qwen2.5/3-VL](https://github.com/QwenLM/Qwen3-VL) and [GUI-Actor](https://github.com/microsoft/GUI-Actor) as backbone models. We thank the open-source community for their valuable contributions.
