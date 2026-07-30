import os
import torch
from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    local_files_only=True,
)
model.config.use_cache = False  # disable cache for training
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True,
)
tokenizer.pad_token = tokenizer.eos_token
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model=get_peft_model(
    model,
    lora_config
)
model.print_trainable_parameters()
dataset=load_dataset(
    "json",
    data_files={
        "train":
        "./processed_data/None system train.jsonl",

        "validation":
        "./processed_data/None system val.jsonl"
    }
)
def data_collator(features):

    input_ids = [
        torch.tensor(f["input_ids"], dtype=torch.long)
        for f in features
    ]

    attention_mask = [
        torch.tensor(f["attention_mask"], dtype=torch.long)
        for f in features
    ]

    labels = [
        torch.tensor(f["labels"], dtype=torch.long)
        for f in features
    ]

    batch = {
        "input_ids": pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        ),

        "attention_mask": pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0
        ),

        "labels": pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )
    }
    return batch
training_args = TrainingArguments(
    output_dir="./qwen3_lora_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="steps",
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
)
trainer.train()
print(trainer.state.best_model_checkpoint)
trainer.save_model(
    "./qwen3_lora_best"
)
# 保存tokenizer
tokenizer.save_pretrained(
    "./qwen3_lora_best"
)