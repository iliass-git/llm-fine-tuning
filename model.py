import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
import os
from utils import clear_gpu_memory
from config import Config

def get_model_configs():
    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=Config.LEARNING_RATE,
        fp16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        logging_dir="./logs",
        report_to="none",
    )
    return lora_config, bnb_config, training_args

def load_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
    except (TypeError, Exception):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,
                trust_remote_code=True,
                legacy=False
            )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def load_model(model_name, bnb_config, lora_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        max_memory={0: "14GB", "cpu": "32GB"},
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model

def train_model(model, dataset, tokenizer, training_args, text_field):
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=training_args,
            dataset_text_field=text_field,
            max_seq_length=Config.MAX_SEQ_LENGTH,
        )
    except TypeError:
        def tokenize_function(examples):
            return tokenizer(
                examples[text_field],
                truncation=True,
                max_length=Config.MAX_SEQ_LENGTH,
                padding="max_length",
            )
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
    trainer.train()
    return trainer

def merge_and_save_model(trainer, model_name, output_dir, merged_dir):
    trainer.model.save_pretrained(output_dir)
    clear_gpu_memory()
    del trainer.model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        offload_folder="./offload",
        offload_state_dict=True,
    )
    model_with_adapters = PeftModel.from_pretrained(
        base_model,
        output_dir,
        offload_folder="./offload",
    )
    merged_model = model_with_adapters.merge_and_unload()
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)
    return merged_model
