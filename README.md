# Ai-testing-program-based-on-qwen-3-8B

一个用于整理问答数据、运行 Qwen 3 8B 基线评测和进行多轮交互测试的最小可复现项目。

> 本仓库处于学习与测试阶段。代码可能仍有未覆盖的边界情况，欢迎反馈。

## 包含内容

- `data_processor.py`：读取 Parquet、CSV、JSON 或 JSONL，校验 `question` / `answer` 字段，按 tokenizer 长度过滤并拆分训练、验证、测试集。
- `baseline.py`：对 JSONL 测试集执行确定性生成，将回答、最终数值和正确性写入 JSONL。
- `chat_persistent.py`：保留对话历史的交互式命令行聊天。
- `check_gpu.py`：输出 PyTorch、CUDA 和显卡信息。
- `examples/sample.jsonl`：不包含隐私或业务数据的输入示例。

模型权重、完整数据集、运行结果、虚拟环境和编辑器设置不会提交到仓库。

## 环境要求

- Python 3.10 或更高版本
- Qwen 3 8B 模型可访问（本地目录或 Hugging Face 模型 ID）
- 推理建议使用支持 CUDA 的 NVIDIA GPU；显存需求取决于精度、量化方式和批大小

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell 激活命令为 `.venv\Scripts\Activate.ps1`。

## 数据格式

输入记录至少需要两个字符串字段：

```json
{"question": "题目文本", "answer": "解题过程 #### 12"}
```

GSM8K 风格答案中的 `#### 数值` 会作为最终答案；如果没有该标记，评测脚本会使用文本中最后出现的数值。完整示例见 `examples/sample.jsonl`。

## 1. 检查 GPU

```bash
python check_gpu.py
```

## 2. 整理数据

```bash
python data_processor.py \
  --model-path Qwen/Qwen3-8B \
  --input-file examples/sample.jsonl \
  --output-dir processed_data
```

如果模型和 tokenizer 已经下载到本地，可以把 `--model-path` 指向该目录，并增加 `--local-files-only`。

默认拆分比例为 80% / 10% / 10%。小型示例可能因向下取整而让部分拆分为空；正式使用时请提供足够多的记录。

## 3. 运行基线评测

```bash
python baseline.py \
  --model-path Qwen/Qwen3-8B \
  --test-file processed_data/test.jsonl \
  --output-file baseline_results.jsonl \
  --batch-size 4
```

显存不足时先降低 `--batch-size` 和 `--max-new-tokens`。输出文件会逐行保存原始记录、模型回答、预测值、期望值和 `is_correct`。

## 4. 交互式聊天

```bash
python chat_persistent.py \
  --model-path Qwen/Qwen3-8B \
  --system-prompt "You are a helpful assistant."
```

输入 `/reset` 清空对话历史，输入 `/exit` 退出。

## 验证

仓库的 GitHub Actions 使用 Python 标准库执行源代码契约检查，验证：

- 必要文件齐全；
- Python 源文件可解析；
- 脚本不包含原服务器的 `/root/` 硬编码路径；
- 主要入口提供命令行参数。

本地可运行：

```bash
python -m unittest discover -s tests -v
```

## 安全说明

不要提交访问令牌、私有数据、模型权重或包含敏感信息的输出。默认 `.gitignore` 已排除常见的大文件和本项目的本地运行产物。
