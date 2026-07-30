import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
from tqdm import tqdm
from xml.parsers.expat import model
from peft import PeftModel
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

BATCH_SIZE = 16
def extract_final_number(text: str):

    match = re.search(
        r"####\s*(-?[\d,]+\.?\d*)",
        text
    )
    if match:
        return float(
            match.group(1).replace(",", "")
        )
    numbers = re.findall(
        r"-?\d+\.?\d*",
        text
    )
    if numbers:
        return float(numbers[-1])

    return None
def run_baseline(
    test_file,
    output_file,
    model,
    tokenizer,
    system_prompt=None
):

    test_samples = []

    with open(
        test_file,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            test_samples.append(
                json.loads(line)
            )


    print(
        f"📂 测试集样本数: {len(test_samples)}"
    )
    results = []
    correct = 0
    total = 0
    for start in tqdm(
        range(
            0,
            len(test_samples),
            BATCH_SIZE
        ),
        desc="🔍 推理中"
    ):

        batch_samples = test_samples[
            start:start+BATCH_SIZE
        ]


        batch_prompts = []

        batch_meta = []


        for sample in batch_samples:


            input_ids = sample["input_ids"]

            labels = sample["labels"]


            # 找assistant答案开始位置
            assistant_start = next(
                (
                    i
                    for i,lbl in enumerate(labels)
                    if lbl != -100
                ),
                len(input_ids)
            )


            prompt_ids = input_ids[
                :assistant_start
            ]


            prompt_text = tokenizer.decode(
                prompt_ids,
                skip_special_tokens=False
            )


            answer_ids = input_ids[
                assistant_start:
            ]


            true_answer_text = tokenizer.decode(
                answer_ids,
                skip_special_tokens=True
            )
            true_answer = extract_final_number(
                true_answer_text
            )
            batch_prompts.append(
                prompt_text
            )
            batch_meta.append(
                {
                    "true_answer": true_answer
                }
            )


        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)



        with torch.inference_mode():

            outputs = model.generate(

                **inputs,

                max_new_tokens=1024,

                do_sample=False,

                use_cache=True,

                pad_token_id=tokenizer.pad_token_id,

                eos_token_id=tokenizer.eos_token_id,

            )
        for i,out in enumerate(outputs):


            generated_tokens = out[
                inputs.input_ids.shape[1]:
            ]


            generated_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )


            pred_answer = extract_final_number(
                generated_text
            )


            true_answer = batch_meta[i][
                "true_answer"
            ]


            is_correct = (
                pred_answer is not None
                and true_answer is not None
                and abs(
                    pred_answer-true_answer
                ) < 0.001
            )


            if is_correct:
                correct += 1


            total += 1


            results.append(
                {
                    "idx": start+i,
                    "generated_text": generated_text,
                    "pred_answer": pred_answer,
                    "true_answer": true_answer,
                    "correct": is_correct
                }
            )



    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )



    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for r in results:

            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False
                )
                + "\n"
            )



    print("\n"+"="*50)

    print("📊 微调模型测试结果")

    print("="*50)


    print(
        f"模型: {MODEL_PATH}"
    )

    print(
        f"测试集: {test_file}"
    )

    print(
        f"总样本: {total}"
    )

    print(
        f"正确: {correct}"
    )

    print(
        f"准确率: {accuracy:.2f}%"
    )

    print(
        f"结果保存: {output_file}"
    )

    print("="*50)



    return accuracy, results





if __name__ == "__main__":


    print("加载原始Qwen3-8B...")


    model_global = AutoModelForCausalLM.from_pretrained(

        MODEL_PATH,

        torch_dtype=torch.bfloat16,

        device_map="auto",

        trust_remote_code=True,

        local_files_only=True

    )
    adapter_path = os.environ.get("ADAPTER_PATH", "./qwen3_lora_best")
    model = PeftModel.from_pretrained(
        model_global,
        adapter_path
    )
    model.eval()
    model_global.eval()



    tokenizer_global = AutoTokenizer.from_pretrained(

        MODEL_PATH,

        trust_remote_code=True,

        local_files_only=True

    )


    tokenizer_global.pad_token = (
        tokenizer_global.eos_token
    )


    tokenizer_global.padding_side = "left"
    print("模型加载完成\n")
    accuracy, results = run_baseline(

        test_file=TEST_FILE,

        output_file=OUTPUT_FILE,

        model=model,

        tokenizer=tokenizer_global,

        system_prompt=SYSTEM_PROMPT

    )