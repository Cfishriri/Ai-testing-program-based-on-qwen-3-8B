
import os
import json
import re
from xml.parsers.expat import model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
# ===================== 配置 =====================
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-8B")
TEST_FILE = "./processed_data/test.jsonl"
OUTPUT_FILE = "./baseline_results.jsonl"
SYSTEM_PROMPT = """
你是一个数学解题助手。
请不要进行长篇推理。
直接给出简短计算步骤。
最终答案必须单独一行：
#### 数字
"""
BATCH_SIZE = 32
def extract_errors(results_file: str, output_file: str = "./error_samples.jsonl"):
    """
    从基线结果中提取预测错误的样本 
    Args:
        results_file: 基线结果文件路径
        output_file: 错误样本输出路径
    """
    errors = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            if not sample["correct"]:
                errors.append(sample)
    # 保存错误样本
    with open(output_file, "w", encoding="utf-8") as f:
        for err in errors:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")
    
    # 打印统计
    with open(results_file, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)
    
    print(f"总样本: {total}")
    print(f"错误样本: {len(errors)} ({len(errors)/total*100:.1f}%)")
    print(f"已保存至: {output_file}")
    
    # 打印前 3 个错误样例
    print("\n" + "=" * 60)
    print("📋 前 3 个错误样本预览")
    print("=" * 60)
    for i, err in enumerate(errors[:3]):
        print(f"\n--- 错误样本 {i+1} (idx={err['idx']}) ---")
        print(f"预测答案: {err['pred_answer']}")
        print(f"正确答案: {err['true_answer']}")
        print(f"模型输出(前200字): {err['generated_text'][:200]}...")
        print()
# ===================== 工具函数 =====================
def extract_final_number(text: str):
    """从文本中提取 #### 后面的最终答案数字"""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return float(numbers[-1])
    return None
# ===================== 主函数 =====================
def run_baseline(test_file, output_file, model, tokenizer, system_prompt=None):
    # 1.读取测试集
    test_samples = []

    with open(test_file,"r",encoding="utf-8") as f:
        for line in f:
            test_samples.append(json.loads(line))
    print(f"📂 测试集样本数: {len(test_samples)}")
    results=[]
    correct=0
    total=0
    for start in tqdm(
        range(0,len(test_samples),BATCH_SIZE),
        desc="🔍 推理中"
    ):
        batch_samples = test_samples[start:start+BATCH_SIZE]
        batch_prompts=[]
        batch_meta=[]
        # ===============================
        # 准备batch数据
        # ===============================

        for sample in batch_samples:

            # 这批样本已经是预处理后的 token 序列。
            # input_ids 包含完整 prompt 与 answer，labels 中的 -100 表示 prompt 部分，
            # 非 -100 的位置表示 assistant 的答案位置。
            # 因此直接从 labels 找到 answer 的起点，再取前缀作为模型输入。
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            assistant_start = next(
                (
                    i
                    for i, lbl in enumerate(labels)
                    if lbl != -100
                ),
                len(input_ids)
            )
            prompt_ids = input_ids[:assistant_start]
            prompt_text = tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False
            )

            answer_ids = input_ids[assistant_start:]
            true_answer_str = tokenizer.decode(
                answer_ids,
                skip_special_tokens=True
            )
            true_answer = extract_final_number(true_answer_str)

            batch_prompts.append(prompt_text)
            batch_meta.append({
                "true_answer": true_answer
            })

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,

            )

        for i, out in enumerate(output_ids):

            generated_tokens = out[
                inputs.input_ids.shape[1]:
            ]

            generated_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

            pred_answer = extract_final_number(generated_text)
            true_answer = batch_meta[i]["true_answer"]

            is_correct = (
                pred_answer is not None
                and true_answer is not None
                and abs(pred_answer - true_answer) < 0.001
            )

            if is_correct:
                correct += 1

            total += 1

            results.append({
                "idx": start + i,
                "generated_text": generated_text,
                "pred_answer": pred_answer,
                "true_answer": true_answer,
                "correct": is_correct
            })
    accuracy=correct/total*100 if total>0 else 0.0
    # 保存
    with open(output_file,"w",encoding="utf-8") as f:

        for r in results:

            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False
                )+"\n"
            )

    print("\n"+"="*50)
    print("📊 基线测试结果")
    print("="*50)
    print(f"模型: {MODEL_PATH}")
    print(f"测试集: {test_file}")
    print(f"总样本: {total}")
    print(f"正确: {correct}")
    print(f"准确率: {accuracy:.2f}%")
    print(f"结果保存: {output_file}")
    print("="*50)
    return accuracy,results
# ===================== 入口 =====================

if __name__=="__main__":
    print("加载模型...")
    model_global = AutoModelForCausalLM.from_pretrained(

        MODEL_PATH,

        torch_dtype=torch.float16,

        device_map="auto",

        trust_remote_code=True,

        local_files_only=True,

    )
    model_global.eval()
    tokenizer_global = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer_global.pad_token = tokenizer_global.eos_token
    tokenizer_global.padding_side="left"
    print("模型加载完成。\n")
    accuracy,results=run_baseline(
        test_file=TEST_FILE,
        output_file=OUTPUT_FILE,
        model=model_global,
        tokenizer=tokenizer_global,
        system_prompt=None
    )
    extract_errors(OUTPUT_FILE, "./error_samples.jsonl")
