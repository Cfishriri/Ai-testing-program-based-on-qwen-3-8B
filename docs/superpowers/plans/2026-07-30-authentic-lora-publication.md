# Authentic LoRA Project Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a draft PR containing a behaviorally faithful, minimally sanitized snapshot of the active remote `blockdata` LoRA project.

**Architecture:** Read each candidate file from the active VS Code Remote SSH workspace, classify it as source or generated artifact, and reproduce included source on the existing GitHub branch. Replace only configuration-bound absolute paths while preserving training, preprocessing, checkpoint, generation, and scoring behavior.

**Tech Stack:** Python 3.10, PyTorch, Transformers, PEFT, Datasets, pandas, tqdm, GitHub Actions.

## Global Constraints

- Target repository: `Cfishriri/Ai-testing-program-based-on-qwen-3-8B`.
- Working branch: `agent/publish-authentic-lora-project`.
- Base branch: `main`.
- Draft PR #1 must remain unchanged.
- Do not upload model weights, LoRA adapter weights, checkpoints, full datasets, full result files, virtual environments, caches, editor state, or credentials.
- Do not redesign training hyperparameters, Chat Template behavior, label masking, LoRA targets, checkpoint loading, generation, answer extraction, or accuracy calculation.
- Every intentional difference from the remote workspace must be listed in the new PR.

---

### Task 1: Inventory and classify the active workspace

**Files:**
- Inspect: `/root/blockdata`
- Record: `docs/publication/file-inventory.md`

**Interfaces:**
- Consumes: VS Code Remote SSH workspace file tree and active editor contents.
- Produces: a table with `path`, `classification`, `include`, and `reason`.

- [ ] **Step 1: Enumerate candidate files**

Use VS Code Explorer and Quick Open to enumerate top-level text source/config files and relevant subdirectories. Record all observed candidates, including `data_processor.py`, `baseline.py`, `lora-math-reasoning.py`, `finetuned.py`, `decode.py`, `check_answer.py`, `check.py`, and `simple agent.py`.

- [ ] **Step 2: Read every candidate completely**

Open each candidate in VS Code and capture its complete editor text. Do not infer file contents from names.

- [ ] **Step 3: Classify files**

Use these exact categories:

- `core-source`: required for preprocessing, training, evaluation, or inference;
- `analysis-source`: useful for inspecting results or failures;
- `scratch`: disposable one-off experiment;
- `generated`: datasets, results, checkpoints, weights, caches, or environment files.

Include `core-source` and useful `analysis-source`; exclude `scratch` and `generated` unless a small sanitized example is required to explain an interface.

- [ ] **Step 4: Create the inventory document**

Create `docs/publication/file-inventory.md` containing the classification table and explicit exclusion reasons.

- [ ] **Step 5: Commit**

Commit message:

```text
docs: inventory authentic LoRA project files
```

### Task 2: Publish faithful source with minimal sanitization

**Files:**
- Create or update: included Python source files at repository root
- Create: `config.example.json` only if multiple scripts share the same path inputs
- Test: `tests/test_publication_contract.py`

**Interfaces:**
- Consumes: exact remote-workspace source captured in Task 1.
- Produces: public source whose only behavioral differences are configuration-bound path substitutions.

- [ ] **Step 1: Write source-contract tests**

Create `tests/test_publication_contract.py` using only the Python standard library. It must:

```python
import ast
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class PublicationContractTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in ROOT.glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_no_private_absolute_paths(self):
        pattern = re.compile(r"/root/(?:blockdata|eb-public)/")
        for path in ROOT.glob("*.py"):
            self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")), path.name)

    def test_no_large_runtime_artifacts(self):
        forbidden = ("processed_data", "qwen3_lora_output", "kv_cache_env")
        for name in forbidden:
            self.assertFalse((ROOT / name).exists(), name)
```

- [ ] **Step 2: Verify the contract initially fails**

Run through GitHub Actions after committing the test alone. Expected: failure because required source is absent or private absolute paths remain in the branch baseline.

- [ ] **Step 3: Reproduce included source**

For each included file, preserve function structure, constants other than machine-specific paths, hyperparameters, prompt text, LoRA configuration, dataset transformations, generation arguments, and scoring behavior.

- [ ] **Step 4: Replace private paths narrowly**

Replace each `/root/blockdata/...` or `/root/eb-public/...` literal only at the configuration boundary. Prefer command-line arguments where the script already has an entry point. If introducing shared JSON configuration would require broad refactoring, keep per-script arguments to minimize behavioral change.

- [ ] **Step 5: Record intentional differences**

Add a table to `docs/publication/sanitization-log.md` with columns:

```text
File | Workspace value | Published value | Behavioral impact
```

Every row must state `none; configuration only` unless a real behavior change was explicitly approved.

- [ ] **Step 6: Commit**

Commit message:

```text
feat: publish authentic LoRA workflow
```

### Task 3: Add safe runtime documentation and dependency metadata

**Files:**
- Modify: `README.md`
- Create or update: `requirements.txt`
- Create: `examples/processed_sample.jsonl`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: imports and data contracts from the published source.
- Produces: reproducible setup instructions without private data or weights.

- [ ] **Step 1: Derive dependencies from imports**

Include the actual third-party packages used by published source: PyTorch, Transformers, PEFT, Datasets, pandas, pyarrow, tqdm, and any additional package found during full inspection. Do not add unused packages.

- [ ] **Step 2: Create a synthetic processed sample**

Create a single synthetic JSONL record matching the actual `input_ids`, `attention_mask`, and `labels` schema if published evaluators consume tokenized records. Clearly label it as structural only and not directly meaningful without the matching tokenizer.

- [ ] **Step 3: Expand ignore rules**

Ensure `.gitignore` excludes:

```text
.vscode/
__pycache__/
*.py[cod]
kv_cache_env/
.venv/
venv/
processed_data/
qwen3_lora_output/
qwen3_lora_best/
*_results.jsonl
error_samples.jsonl
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
*.onnx
```

- [ ] **Step 4: Document the real workflow**

README must describe the real sequence:

```text
data_processor.py -> lora-math-reasoning.py -> baseline.py / finetuned.py -> analysis scripts
```

Document configuration arguments, checkpoint selection, expected data schema, excluded artifacts, and the fact that logic was preserved while paths were sanitized.

- [ ] **Step 5: Commit**

Commit message:

```text
docs: document authentic LoRA workflow
```

### Task 4: Validate and open the new draft PR

**Files:**
- Create or update: `.github/workflows/ci.yml`
- Inspect: all files changed from `main`

**Interfaces:**
- Consumes: published source, tests, docs, and ignore rules.
- Produces: validated draft PR into `main`.

- [ ] **Step 1: Add CI**

Configure Python 3.10 and run:

```text
python -m unittest discover -s tests -v
```

- [ ] **Step 2: Run source and security validation**

Confirm:

- every Python file parses;
- no private absolute paths remain;
- no token, private-key, or credential patterns are present;
- no excluded directory or weight/result artifact is present;
- every included source file appears in the inventory and sanitization log.

- [ ] **Step 3: Inspect final branch contents**

List the exact changed filenames against `main`. Reject any file not traceable to the approved design.

- [ ] **Step 4: Verify GitHub Actions**

Wait for the workflow attached to the final head commit. Expected: `completed / success`.

- [ ] **Step 5: Open a draft PR**

Create a draft PR:

```text
agent/publish-authentic-lora-project -> main
```

The body must list included files, excluded artifacts, every sanitization category, validation evidence, and known limitations. Do not modify or close draft PR #1.

- [ ] **Step 6: Final verification**

Re-fetch PR metadata and changed filenames. Confirm it is open, draft, mergeable when GitHub reports mergeability, and targets `main`.
