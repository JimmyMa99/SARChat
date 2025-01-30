# SARChat

<p align="center">
  <img src="./assets/logo.png" alt="SARChat Logo" width="800"/>
</p>

<p align="center">
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg"/>
  </a>
  <a href="https://huggingface.co/datasets/YourOrg/SARChat">
    <img alt="HF Dataset" src="https://img.shields.io/badge/🤗-Dataset-yellow.svg"/>
  </a>
  <a href="https://huggingface.co/YourOrg/SARChat">
    <img alt="HF Model" src="https://img.shields.io/badge/🤗-Models-blue.svg"/>
  </a>
  <a href="https://modelscope.cn/datasets/YourOrg/SARChat">
    <img alt="ModelScope Dataset" src="https://img.shields.io/badge/ModelScope-Dataset-orange.svg"/>
  </a>
  <a href="https://modelscope.cn/models/YourOrg/SARChat">
    <img alt="ModelScope Model" src="https://img.shields.io/badge/ModelScope-Models-green.svg"/>
  </a>
  <a href="https://arxiv.org/abs/xxxx.xxxxx">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg"/>
  </a>
</p>

## Introduction

SARChat-Bench-2M is the first large-scale multimodal dialogue dataset focusing on Synthetic Aperture Radar (SAR) imagery. It contains approximately 2 million high-quality SAR image-text pairs, supporting multiple tasks including scene classification, image captioning, visual question answering, and object localization. We conducted comprehensive evaluations on 11 state-of-the-art vision-language models (including Qwen2VL, InternVL2.5, and LLaVA), establishing the first multi-task benchmark in the SAR domain.

📑 Read more about SARChat in our [paper](https://arxiv.org/abs/xxxx.xxxxx).

## Key Features

- 🌟 **2M+** high-quality SAR image-text pairs
- 🔍 Covers diverse scenes including marine, terrestrial and urban areas
- 📊 **6 task-specific benchmarks** with fine-grained annotations
- 🤖 Evaluated on **11 SOTA vision-language models**
- 🛠️ Ready-to-use format with shape, count, location labels

## Dataset Statistics

### Tasks Statistics

| Task | Train Set | Test Set |
|------|-----------|-----------|
| Classification | 81,788 | 10,024 |
| Fine-Grained Description | 46,141 | 6,032 |
| Instance Counting | 95,493 | 11,704 |
| Spatial Grounding | 94,456 | 11,608 |
| Cross-Modal Identification | 1,423,548 | 175,565 |
| Referring | 95,486 | 11,703 |

### Words Statistics

| Metric                 | Value       |
| ---------------------- | ----------- |
| Total Words            | 43,978,559  |
| Total Sentences        | 4,222,143   |
| Average Caption Length | 10.66  |

## Quick Start

🤗 Visit our [Hugging Face dataset page](https://huggingface.co/datasets/YourOrg/SARChat) for more details and examples.

## SARChat Models

We have trained several models using the SARChat dataset. These models are available on Hugging Face:

- [SARChat-7B](https://huggingface.co/YourOrg/SARChat-7B)
- [SARChat-13B](https://huggingface.co/YourOrg/SARChat-13B)
- [SARChat-34B](https://huggingface.co/YourOrg/SARChat-34B)

## Citation

If you use this dataset or our models in your research, please cite our paper.

## Contact

For any questions or feedback, please contact:

- 📧 Email: mazhiming312@outlook.com
- 💬 GitHub Issues: Feel free to open an issue in this repository


## Acknowledgments

We would like to thank:
- [Organization/Individual] for their support in data collection
- [Organization/Individual] for their valuable feedback
- [Computing Infrastructure] for providing computing resources
- All contributors who helped create and improve this dataset

---
<p align="center">
  <i>If you find SARChat useful, please consider giving it a star ⭐</i>
</p>