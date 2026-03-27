!pip install transformers accelerate datasets peft bitsandbytes sentencepiece evaluate

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from datasets import Dataset

df = pd.read_csv("/content/drive/MyDrive/MQA/medquad.csv")

# Keep only needed columns
df = df[["question", "answer"]]

# Sample 30%
df = df.sample(frac=0.3, random_state=42)

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)

# Split train/validation (90/10)
dataset = dataset.train_test_split(test_size=0.1)

dataset

def format_example(example):
    question = example["question"] if example["question"] is not None else ""
    answer   = example["answer"] if example["answer"] is not None else ""
    example["text"] = (
        "<|im_start|>user\n"
        + question
        + "\n<|im_end|>\n<|im_start|>assistant\n"
        + answer
        + "\n<|im_end|>"
    )
    return example

dataset = dataset.map(format_example)

dataset['test']

dataset['test'][0]

dataset['train']

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    # attn_implementation="flash_attention_2",
)

tokenizer

model

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    # Question , Answer
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="qwen-medical-qa",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=50,
    max_steps=800,

    bf16=True,
    fp16=False,

    logging_steps=20,
    # evaluation_strategy="steps", # Removed as it causes TypeError
    eval_steps=200,
    # save_strategy="steps",
    save_steps=200,
    report_to="none",
)

# Define Evaluation Metrics (F1 + EM)
# We use token-level F1 and Exact Match (SQuAD-style).

import evaluate
import numpy as np

f1_metric = evaluate.load("f1")
em_metric = evaluate.load("exact_match")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Decode
    decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    f1_scores = []
    em_scores = []

    for pred, label in zip(decoded_preds, decoded_labels):
        f1_scores.append(f1_metric.compute(predictions=[pred], references=[label])["f1"])
        em_scores.append(em_metric.compute(predictions=[pred], references=[label])["exact_match"])

    return {
        "f1": np.mean(f1_scores),
        "exact_match": np.mean(em_scores)
    }

def preprocess(batch):
    tokenized_batch = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512) # Reduced max_length from 1024 to 512
    # For causal LMs, labels are usually the input_ids themselves for training loss computation.
    tokenized_batch["labels"] = tokenized_batch["input_ids"].copy()
    return tokenized_batch

tokenized = dataset.map(preprocess, batched=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= tokenized["train"],
    eval_dataset = tokenized["test"],
    compute_metrics = compute_metrics,
)

trainer.train()

prompt = "What is Glaucoma?"

input_ids = tokenizer(
    f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n",
    return_tensors="pt"
).to(model.device)

generated = model.generate(
    **input_ids,
    max_new_tokens=300,
    temperature=0.2,
)

print(tokenizer.decode(generated[0], skip_special_tokens=True))

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Load Base Model
base_model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    load_in_4bit=True
)

# 2. Load Fine-Tuned Model (LoRA)
from peft import PeftModel

ft_model_path = "/content/qwen-medical-qa"
ft_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    load_in_4bit=True
)
ft_model = PeftModel.from_pretrained(ft_model, ft_model_path)

# 3. Define 5 Qualitative Evaluation Prompts
prompts = [
    "What is the difference between glaucoma and cataracts?",
    "How is acute appendicitis diagnosed?",
    "What are the complications of uncontrolled diabetes?",
    "Explain the treatment protocol for bacterial pneumonia.",
    "What symptoms indicate a possible heart attack?"
]

# 4. Generation Helper

def generate_answer(model, prompt):
    input_ids = tokenizer(
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n",
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **input_ids,
        max_new_tokens=250,
        temperature=0.2
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# 5. Compare Base vs Fine-Tuned

for i, prompt in enumerate(prompts, 1):
    print("="*80)
    print(f"PROMPT {i}: {prompt}")
    print("-"*80)

    base_output = generate_answer(base_model, prompt)
    ft_output   = generate_answer(ft_model, prompt)

    print("\n--- BASE MODEL RESPONSE ---\n")
    print(base_output)

    print("\n--- FINE-TUNED (SFT) MODEL RESPONSE ---\n")
    print(ft_output)

# !mv /content/qwen-medical-qa /content/drive/MyDrive/MQA