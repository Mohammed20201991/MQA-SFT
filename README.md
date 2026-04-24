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
```

## Dataset

The system is trained on medical question–answer datasets (MedQuAD-like resources), formatted as:
## Installation
```
git clone https://github.com/your-username/MQA-SFT.git
cd MQA-SFT
pip install -r requirements.txt
```
- Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0
- transformers
- datasets
- peft
- accelerate

## Training
Supervised Fine-Tuning (SFT)
`python training/train_sft.py`

Parameter-Efficient Fine-Tuning (PEFT – LoRA)
`python training/train_peft.py`

## Supported features:
Mixed precision (FP16)
GPU / Google Colab
Automatic checkpoint saving

## Evaluation
The model is evaluated using standard QA metrics:
- Exact Match (EM)
- Token-level F1-score

`python evaluation/evaluate.py`

## Experimental Setup
- Optimizer: AdamW
- Learning rate scheduling with warmup
- Evaluation on validation and test splits
- Long-context medical QA scenarios


### Citation
```
@article{frhan2026mqa,
  title   = {Supervised Fine-Tuning Approach for Medical Question-Answering using Qwen Instruct},
  author  = {Amjad Jumaah Frhan and Mohammed A. S. Al-Hitawi and Hiba A. Abu-Alsaad},
  journal = {International Journal of Computers},
  volume  = {11},
  pages   = {57--62},
  year    = {2026},
  publisher = {IARAS}
}
```
🔗 Paper Link:
https://www.iaras.org/home/cijc/supervised-fine-tuning-approach-for-medical-question-answering-using-qwen-instruct

### Reference (APA Style)
```
Frhan, A. J., Al-Hitawi, M. A. S., & Abu-Alsaad, H. A. (2026).
Supervised Fine-Tuning Approach for Medical Question-Answering using Qwen Instruct.
International Journal of Computers, 11, 57–62.
```
