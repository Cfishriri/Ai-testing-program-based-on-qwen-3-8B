# Qwen3-8B 数学推理 LoRA 实验

本分支公开的是 `blockdata` 工作区当前使用的真实数据处理、基线测试、LoRA 训练和微调后测试逻辑。除了把服务器绝对路径替换为环境变量外，没有重构训练、生成或评分算法。

## 主流程

1. `data_processor.py`：把 GSM8K 的 `question` / `answer` 转成 Qwen Chat Template token，并只对 assistant 答案计算 loss。
2. `baseline.py`：测试原始 Qwen3-8B。
3. `lora-math-reasoning.py`：训练 LoRA，并保存最佳适配器。
4. `finetuned.py`：加载基础模型和 LoRA 适配器进行评测。
5. `check_answer.py`：查找“基线正确、LoRA 错误”的样本。

辅助脚本：`decode.py` 检查 token/label 边界，`check_gpu.py` 检查运行环境。

## 安装与路径

```bash
pip install -r requirements.txt
export MODEL_PATH=/path/to/Qwen3-8B
export INPUT_FILE=/path/to/train.parquet
export ADAPTER_PATH=/path/to/qwen3_lora_best
```

脚本使用 `local_files_only=True`，模型需已下载到本地。默认模型标识是 `Qwen/Qwen3-8B`，默认输入是 `./data/train.parquet`，默认适配器是 `./qwen3_lora_best`。

## 当前真实训练参数

- `r=8`、`lora_alpha=16`、dropout `0.05`
- 目标层：`q_proj`、`v_proj`
- 1 epoch，单卡 batch 2，梯度累积 8
- 学习率 `5e-5`
- 每 200 step 保存并评估，以 `eval_loss` 选择最佳 checkpoint

## 准确率下降时首先检查

- `baseline.py` 使用 `system_prompt=None`，`finetuned.py` 使用数学 system prompt，评测输入不完全一致。
- 数据处理也使用 `system_prompt=None`，与微调后测试 prompt 不一致。
- `finetuned.py` 当前仍写入 `baseline_results.jsonl`；这是工作区真实行为，发布时未擅自修改。需要并排比较时，应改为 `finetuned_results.jsonl`。
- 只比较 checkpoint 的准确率前，应固定同一测试集、prompt、解码设置和评分函数。

模型权重、LoRA checkpoint、虚拟环境、原始数据及生成结果未上传。发布范围见 `docs/publication/file-inventory.md`。
