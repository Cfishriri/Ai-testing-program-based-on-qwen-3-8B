import argparse
import json
import random
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def load_records(path):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    if suffix == ".csv":
        return pd.read_csv(path).to_dict("records")
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else payload["data"]
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raise ValueError(f"Unsupported input format: {suffix}")


def normalize_record(record):
    question = record.get("question")
    answer = record.get("answer")
    if question is None or answer is None:
        raise ValueError("Each record must contain 'question' and 'answer' fields.")
    return {"question": str(question).strip(), "answer": str(answer).strip()}


def token_length(tokenizer, record):
    messages = [
        {"role": "user", "content": record["question"]},
        {"role": "assistant", "content": record["answer"]},
    ]
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return len(token_ids)


def split_records(records, train_ratio, val_ratio):
    train_end = int(len(records) * train_ratio)
    val_end = train_end + int(len(records) * val_ratio)
    return records[:train_end], records[train_end:val_end], records[val_end:]


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_dataset(args):
    if args.train_ratio < 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio > 1:
        raise ValueError("Ratios must be non-negative and train_ratio + val_ratio <= 1.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    records = [normalize_record(item) for item in load_records(args.input_file)]
    records = [item for item in records if token_length(tokenizer, item) <= args.max_seq_len]

    random.Random(args.seed).shuffle(records)
    train, val, test = split_records(records, args.train_ratio, args.val_ratio)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": args.output_dir / "train.jsonl",
        "val": args.output_dir / "val.jsonl",
        "test": args.output_dir / "test.jsonl",
    }
    for name, subset in (("train", train), ("val", val), ("test", test)):
        write_jsonl(paths[name], subset)
        print(f"{name}: {len(subset)} records -> {paths[name]}")
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize and split question-answer datasets.")
    parser.add_argument("--model-path", required=True, help="Tokenizer directory or Hugging Face model ID.")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    process_dataset(parse_args())
