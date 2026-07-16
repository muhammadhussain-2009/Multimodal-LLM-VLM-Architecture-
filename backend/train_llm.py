"""
train_llm.py
============
Cloud-ready LoRA fine-tuning script for Llama-3-8B using the AI2D dataset.
This script is designed to run on a machine with a dedicated NVIDIA GPU (e.g., A100/H100)
and requires at least 24GB VRAM for 4-bit QLoRA.

Dependencies required on target machine:
pip install torch transformers datasets peft bitsandbytes accelerate wandb trl
"""

import os
import torch
import warnings
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
try:
    from trl import SFTTrainer
except ImportError:
    warnings.warn("Please install 'trl' library to use SFTTrainer: pip install trl")
    SFTTrainer = None

# Model Configuration
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"  # Requires HF Token/approval
OUTPUT_DIR = "./lora-llama3-ai2d"
MAX_SEQ_LENGTH = 1024

def format_ai2d_prompt(example):
    """
    Format the AI2D dataset example into a Llama-3 Instruct prompt.
    The AI2D dataset contains questions and answers about diagrams.
    """
    question = example.get("question", "")
    answer = example.get("answer", "")
    
    # In a fully multimodal scenario, you'd feed the image too (e.g. Llava).
    # Since this is an LLM fine-tune, we simulate the text-only reasoning path.
    
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert STEM diagram analyst. Answer the user's question about the diagram.\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{answer}<|eot_id|>"
    )
    return {"text": prompt}

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU is required to run this training script.")

    print("Loading AI2D Dataset...")
    # Load AI2D dataset from HuggingFace
    dataset = load_dataset("lmms-lab/ai2d", split="train")
    
    print("Formatting dataset for instruction tuning...")
    dataset = dataset.map(format_ai2d_prompt, remove_columns=dataset.column_names)
    # Shuffle and select a subset if full dataset is too large
    dataset = dataset.shuffle(seed=42)

    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.05)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, add_eos_token=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print("Configuring 4-bit Quantization (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading Base Model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    print("Configuring LoRA adapters...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=500, # Adjust based on dataset size
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="wandb",
    )

    if SFTTrainer is None:
        raise ImportError("trl library is required for SFTTrainer.")

    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting Training...")
    trainer.train()

    print("Saving Adapter Model...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Fine-tuning complete. Adapter saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
