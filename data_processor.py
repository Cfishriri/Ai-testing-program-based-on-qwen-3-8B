import os
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from transformers import AutoTokenizer
# ===================== 全局配置（可外部传入） =====================
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B")
INPUT_FILE = os.environ.get("INPUT_FILE", "./data/train.parquet")
OUTPUT_DIR = "./processed_data"
MAX_SEQ_LEN = 1024
""""""
SYSTEM_PROMPT_GSM8K = """你是一个数学解题助手。

请不要进行长篇推理。
直接给出简短计算步骤。

最终答案必须单独一行：
#### 数字"""""""""
class DataProcessor:
    """GSM8K SFT数据集处理类：读取→格式化ChatTemplate→正确Label Mask→导出JSONL"""

    def __init__(self, model_path: str, max_length: int = 2048):
        self.max_length = max_length
        # 加载Tokenizer增加异常捕获
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True
            )
        except Exception as e:
            raise RuntimeError(f"Tokenizer加载失败：{e}")

        # 补全pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 统计容器
        self.stats = {
            "total": 0,
            "valid": 0,
            "truncated_drop": 0,
            "empty_sample": 0,
            "error": 0,
            "lengths": []
        }
    def read_data(self, file_path: str) -> pd.DataFrame:
        """读取parquet/json/jsonl/csv数据"""
        fp = Path(file_path)
        if not fp.exists():
            raise FileNotFoundError(f"文件不存在：{fp}")

        print(f"读取文件：{fp}")
        suffix = fp.suffix.lower()
        df = pd.DataFrame()

        if suffix == ".parquet":
            df = pd.read_parquet(fp)
        elif suffix == ".jsonl":
            records = []
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            df = pd.DataFrame(records)
        elif suffix == ".json":
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        elif suffix == ".csv":
            df = pd.read_csv(fp)
        else:
            raise ValueError(f"不支持格式：{suffix}")

        print(f"原始数据量：{len(df)}，字段：{list(df.columns)}")
        return df

    def map_to_messages(
            self,
            row,
            user_field: str = "question",
            assistant_field: str = "answer",
            system_prompt: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """单行转messages，过滤空问答"""
        user_content = str(row.get(user_field, "")).strip()
        assistant_content = str(row.get(assistant_field, "")).strip()

        # 过滤空样本
        if not user_content or not assistant_content:
            return None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})
        return messages

    def tokenize_messages(self, messages: List[Dict]) -> Optional[Dict]:
        """
        正确SFT Tokenize + Label Mask
        仅assistant部分计算loss，prompt部分-100
        """
        try:
            # 1. 完整对话套模板（带assistant内容，不加generation prompt）
            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
            # 末尾追加EOS（Qwen训练标配）
            full_text = full_prompt + self.tokenizer.eos_token

            # 2. Tokenize并截断
            enc = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding=False,
                enable_thinking=False,
                return_tensors=None
            )
            input_ids = enc["input_ids"]
            attn_mask = enc["attention_mask"]

            # 3. 只编码prompt部分（system+user）用于打-100标签
            prompt_only = self.tokenizer.apply_chat_template(
                [m for m in messages if m["role"] != "assistant"],
                tokenize=False,
                enable_thinking=False,
                add_generation_prompt=True
            )
            prompt_ids = self.tokenizer.encode(prompt_only, add_special_tokens=False)
            prompt_len = len(prompt_ids)

            # 4. 构造labels：前缀-100，保留回答部分
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            # 强制对齐长度（截断兜底）
            labels = labels[:len(input_ids)]

            # 超长直接丢弃（真正过滤超长样本）
            if len(input_ids) > self.max_length:
                self.stats["truncated_drop"] += 1
                return None

            self.stats["lengths"].append(len(input_ids))
            return {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "labels": labels
            }
        except Exception as e:
            print(f"Tokenize失败：{str(e)}")
            self.stats["error"] += 1
            return None

    def process(
            self,
            input_file: str,
            output_dir: str,
            user_field: str = "question",
            assistant_field: str = "answer",
            system_prompt: Optional[str] = None,
            sample_limit: Optional[int] = None,
            train_ratio: float = 0.8,
            val_ratio:float = 0.1,
            seed: int = 42
    ) -> Tuple[str, str]:
        df = self.read_data(input_file)
        self.stats["total"] = len(df)

        # 限制样本量
        if sample_limit and len(df) > sample_limit:
            df = df.head(sample_limit).copy()
            print(f"限制采样数量：{sample_limit}")

        processed = []
        for _, row in df.iterrows():
            msg = self.map_to_messages(row, user_field, assistant_field, system_prompt)
            if msg is None:
                self.stats["empty_sample"] += 1
                continue
            token_res = self.tokenize_messages(msg)
            if token_res is None:
                continue
            processed.append(token_res)
            self.stats["valid"] += 1

        # 打印统计信息
        self._print_stats()
        # 固定随机种子划分数据集
        random.seed(seed)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        random.shuffle(processed)
        train_end = int(len(processed) * train_ratio)
        val_end = int(len(processed) * val_ratio+train_end)
        train_data = processed[:train_end]
        val_data = processed[train_end:val_end]
        test_data = processed[val_end:]
        # 保存目录
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "None system train.jsonl")
        val_path = os.path.join(output_dir, "None system val.jsonl")
        test_path = os.path.join(output_dir, "None system test.jsonl")
    
        self._save_jsonl(train_data, train_path)
        self._save_jsonl(val_data, val_path)
        self._save_jsonl(test_data,test_path)
        print(f"\n✅ 导出完成：")
        print(f"训练集 {len(train_data)} 条 → {train_path}")
        print(f"验证集 {len(val_data)} 条 → {val_path}")
        print(f"测试集 {len(test_data)}条 → {test_path}")
        return train_path, val_path

    def _save_jsonl(self, data: List[Dict], save_path: str):
        """安全写入JSONL"""
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
        except IOError as e:
            raise IOError(f"写入文件失败 {save_path}: {e}")

    def _print_stats(self):
        lengths = self.stats["lengths"]
        print("\n" + "=" * 60)
        print("📊 GSM8K 数据处理统计汇总")
        print("=" * 60)
        print(f"原始总行数：{self.stats['total']}")
        print(f"有效可用样本：{self.stats['valid']}")
        print(f"空问答丢弃：{self.stats['empty_sample']}")
        print(f"超长截断丢弃：{self.stats['truncated_drop']}")
        print(f"处理异常失败：{self.stats['error']}")
        if lengths:
            lengths_sorted = sorted(lengths)
            print(f"最短Token长度：{min(lengths)}")
            print(f"最长Token长度：{max(lengths)}")
            print(f"平均Token长度：{sum(lengths)/len(lengths):.2f}")
            print(f"中位数Token长度：{lengths_sorted[len(lengths_sorted)//2]}")
        print("=" * 60 + "\n")
if __name__ == "__main__":
    # 实例化处理器
    processor = DataProcessor(model_path=MODEL_PATH, max_length=MAX_SEQ_LEN)
    # 执行处理
    processor.process(
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        user_field="question",
        assistant_field="answer",
        system_prompt=None,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42
    )