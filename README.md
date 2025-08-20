# Vision Language Driving Perception

**Fine-tuning Vision-Language Models for Autonomous Driving Decision Planning**

---

## Overview

Vision Language Driving Perception is an open-source project focused on fine-tuning Vision-Language Models (VLMs) for decision planning in autonomous driving scenarios. By leveraging the expressive power of pre-trained VLMs, this project adapts them to downstream driving tasks such as behavior prediction, maneuver classification, and goal-directed planning.

This repository provides tools, datasets, and training pipelines to adapt InterVL2-1B for real-world autonomous driving decision modules.

---

## Features

- **VLM Fine-Tuning Pipeline**: Modular pipeline to fine-tune VLM on driving-specific tasks.
- **Dataset Integration**: Supports structured scene data (e.g., nuPlan, Waymo, or custom vectorized environments).
- **Prompt Engineering for Driving Tasks**: Custom vision-language prompts for planning-relevant tasks.
- **Evaluation Tools**: Custom metrics for VLM output quality and scenario performance.

---

## Use Cases

- Planning-aware scene understanding  
- Maneuver prediction with vision-language reasoning  
- Goal-directed trajectory selection  
- Safety-critical decision refinement using natural language context

---

## Project Structure

```bash
Vision Language Driving Perception/
│
├── README.md
├── setup.py
├── command
│   ├── InternVL2-1B.sh
│   └── eval.sh
├── config
│   └── zero_stage1_config.json
├── data
│   ├── sample.jsonl
│   └── vlm_samples
├── docs
│   ├── env_install.md
│   ├── vlm_finetune.md
│   └── vlm_trt.md
├── internvl
│   ├── __init__.py
│   ├── conversation.py
│   ├── dist_utils.py
│   ├── model
│   ├── patch
│   └── train
├── scripts
│   ├── internvl_eval.py
│   ├── pytorch_internvl_infer.py
│   └── trt_internvl_infer.py
├── tools
│   ├── __init__.py
│   ├── arrow2jsonl.py
│   ├── bart_score.py
│   ├── convert_parquet.py
│   ├── convert_to_int8.py
│   ├── extract_mlp.py
│   ├── extract_video_frames.py
│   ├── extract_vit.py
│   ├── json2jsonl.py
│   ├── jsonl2jsonl.py
│   ├── merge_lora.py
│   ├── replace_llm.py
│   └── resize_pos_embed.py
└── vlmtrt
    ├── build_vit_engine.py
    ├── conversation.py
    └── convert_qwen2_ckpt.py
```

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/thillai-c/Vision-Language-Driving-Perception.git
cd Vision-Language-Driving-Perception
```

### 2. Install dependencies
```bash
conda create -n vision-language-driving-perception python=3.10
conda activate vision-language-driving-perception
pip install -r requirements.txt
```

### 3. Prepare dataset
You can use scene data from nuPlan, Waymo, or a custom driving dataset. Please follow [docs/data_prepare.md](docs/data_prepare.md) for instructions.

### 4. Run training
Please follow [docs/vlm_finetune.md](docs/vlm_finetune.md) to fine-tune VLMs.

## Evaluation

Please follow [docs/vlm_finetune.md](docs/vlm_finetune.md) to evaluate your fine-tuned model on benchmark scenarios.

## VLM Converter Module

The VLM Converter is a performance-boosting module designed to convert and quantize large Vision-Language Models (VLMs) using TensorRT-LLM, significantly improving inference speed while maintaining accuracy.

Please follow [docs/vlm_trt.md](docs/vlm_trt.md) to convert and quantize VLMs.

## Acknowledgments

This project builds upon the work of:

- [InterVL](https://github.com/OpenGVLab/InternVL)
- [nuPlan](https://github.com/motional/nuplan-devkit)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [BLIP](https://github.com/salesforce/BLIP)
- [LLaVA](https://github.com/haotian-liu/LLaVA)

And the broader open-source autonomous driving and VLM communities.