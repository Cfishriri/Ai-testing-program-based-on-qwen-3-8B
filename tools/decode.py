import os
from transformers import AutoTokenizer
import json
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B")
FILE = "./processed_data/None system train.jsonl"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)
# 读取第一条
with open(FILE, "r", encoding="utf-8") as f:
    sample = json.loads(f.readline())
input_ids = sample["input_ids"]
labels = sample["labels"]


print("="*50)
print("完整input:")
print("="*50)

text = tokenizer.decode(
    input_ids,
    skip_special_tokens=False
)

print(text)



print("\n"+"="*50)
print("labels对应answer部分:")
print("="*50)


# 找非-100位置
answer_start = next(
    i
    for i,x in enumerate(labels)
    if x != -100
)


answer_ids = input_ids[answer_start:]


answer_text = tokenizer.decode(
    answer_ids,
    skip_special_tokens=False
)


print(answer_text)



print("\n"+"="*50)
print("prompt部分:")
print("="*50)


prompt_ids = input_ids[:answer_start]
prompt_text = tokenizer.decode(
    prompt_ids,
    skip_special_tokens=False
)
print(prompt_text)
