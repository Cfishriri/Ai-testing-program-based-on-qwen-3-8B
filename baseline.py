import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


FINAL_NUMBER_PATTERN = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def extract_final_number(text):
    match = FINAL_NUMBER_PATTERN.search(text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_prompt(tokenizer, question):
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def run_baseline(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model.eval()

    records = load_jsonl(args.test_file)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with args.output_file.open("w", encoding="utf-8") as output:
        for start in tqdm(range(0, len(records), args.batch_size), desc="Evaluating"):
            batch = records[start : start + args.batch_size]
            prompts = [build_prompt(tokenizer, item["question"]) for item in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for item, response in zip(batch, responses):
                expected = extract_final_number(str(item.get("answer", "")))
                predicted = extract_final_number(response)
                result = {
                    **item,
                    "model_response": response,
                    "prediction": predicted,
                    "expected": expected,
                    "is_correct": predicted is not None and predicted == expected,
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a deterministic Qwen baseline on JSONL data.")
    parser.add_argument("--model-path", required=True, help="Local model directory or Hugging Face model ID.")
    parser.add_argument("--test-file", type=Path, default=Path("processed_data/test.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("baseline_results.jsonl"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_baseline(parse_args())
