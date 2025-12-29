# MQA-SFT
Medical Question Answering using SFT and PEFT Approach
# MQA-SFT  
### Medical Question Answering using Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT)


## Overview
**MQA-SFT** is a research-focused repository accompanying a scientific paper on **Medical Question Answering (MQA)** using **Large Language Models (LLMs)**.  
The project investigates the effectiveness of **Supervised Fine-Tuning (SFT)** combined with **Parameter-Efficient Fine-Tuning (PEFT)** techniques to build accurate, efficient, and scalable medical QA systems.

The approach aims to reduce computational cost while maintaining strong performance, making it suitable for deployment on **resource-constrained environments** such as Google Colab.

## Research Objectives
- Adapt a pre-trained LLM to the **medical QA domain** using **SFT**
- Apply **PEFT techniques (e.g., LoRA)** to reduce memory and training overhead
- Evaluate performance using **Exact Match (EM)** and **F1-score**
- Enable reproducibility for academic research

## Model Architecture
- **Base Model:** Qwen2.5-3B  
- **Architecture:** Decoder-only Transformer  
- **Key Components:**
  - Grouped-Query Attention (GQA)
  - Rotary Positional Embeddings (RoPE)
  - RMSNorm normalization
  - SwiGLU activation functions
- **Context Length:** Up to 32K tokens


## Repository Structure
```text
MQA-SFT/
│
├── data/                # Medical QA datasets and splits
├── notebooks/           # Colab-ready notebooks
├── training/            # SFT and PEFT training scripts
├── evaluation/          # Evaluation scripts (EM, F1)
├── models/              # Saved checkpoints / LoRA adapters
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

