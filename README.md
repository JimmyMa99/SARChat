# SARChat-Bench-2M

## Introduction

SARChat-Bench-2M is the first large-scale multimodal dialogue dataset focusing on Synthetic Aperture Radar (SAR) imagery. It contains approximately 2 million high-quality SAR image-text pairs, supporting multiple tasks including scene classification, image captioning, visual question answering, and object localization. We conducted comprehensive evaluations on 16 state-of-the-art vision-language models, establishing the first multi-task benchmark in the SAR domain.

## Key Features

- 🌟 **2M+** high-quality SAR image-text pairs
- 🔍 Covers diverse scenes including marine, terrestrial and urban areas  
- 📊 **6 task-specific benchmarks** with fine-grained annotations
- 🤖 Evaluated on **16 SOTA vision-language models**
- 🛠️ Ready-to-use format with shape, count, location labels

## Dataset Statistics

### Tasks Statistics

| Task | Train Set | Test Set |
|:---:|:---:|:---:|
| Classification | 81,788 | 10,024 |
| Fine-Grained Description | 46,141 | 6,032 |
| Instance Counting | 95,493 | 11,704 |
| Spatial Grounding | 94,456 | 11,608 |
| Cross-Modal Identification | 1,423,548 | 175,565 |
| Referring | 95,486 | 11,703 |

### Words Statistics

| Metric | Value |
|:---:|:---:|
| Total Words | 43,978,559 |
| Total Sentences | 4,222,143 |
| Average Caption Length | 10.66 |

## Model Overview

Our benchmark includes evaluation of multiple vision-language models with varying parameter sizes:
- Small models (1B-2B parameters)
- Medium models (4B-5B parameters)  
- Large models (7B-8B parameters)

The models were evaluated across all six tasks in our benchmark, demonstrating varying capabilities in SAR image interpretation, spatial reasoning, and cross-modal understanding.

