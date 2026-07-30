# blockdata publication inventory

## Included core files

- `data_processor.py`: dataset conversion and label masking
- `baseline.py`: base-model evaluation
- `lora-math-reasoning.py`: authentic LoRA training
- `finetuned.py`: adapter evaluation

## Included diagnostics

- `decode.py`: inspect tokenized prompt and answer boundaries
- `check_answer.py`: compare baseline and LoRA errors
- `check_gpu.py`: inspect Python, PyTorch, CUDA, GPU and memory

## Excluded

- `check.py`: duplicate scratch version of `check_answer.py`
- `simple agent.py`: separate Qwen2.5 tool-calling experiment, not part of the Qwen3 LoRA accuracy workflow
- chat persistence experiment: unrelated to the LoRA workflow
- conda environment listing: machine-specific
- processed datasets, JSONL results, checkpoints, model weights, caches and virtual environments: generated, large or machine-specific

## Sanitization

Only machine-specific absolute paths were replaced with `MODEL_PATH`, `INPUT_FILE` and `ADAPTER_PATH` environment variables. Training, prompting, generation and scoring behavior was otherwise preserved.
