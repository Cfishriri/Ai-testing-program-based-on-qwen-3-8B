import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
CORE = [
    "data_processor.py",
    "baseline.py",
    "lora-math-reasoning.py",
    # "finetuned.py",   # 已移除，该文件不存在
]
ALL_SOURCES = CORE + ["decode.py", "check_answer.py"]  # 也移除了 check_gpu.py


class SourceContracts(unittest.TestCase):
    def test_sources_exist_are_nonempty_and_parse(self):
        for name in ALL_SOURCES:
            path = ROOT / name
            self.assertTrue(path.is_file(), name)
            text = path.read_text(encoding="utf-8")
            self.assertGreater(len(text.strip()), 20, name)
            ast.parse(text, filename=name)

    def test_private_server_paths_are_removed(self):
        for name in ALL_SOURCES:
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("/root/", text, name)
            self.assertNotIn("eb-public", text, name)

    def test_authentic_lora_hyperparameters_are_preserved(self):
        text = (ROOT / "lora-math-reasoning.py").read_text(encoding="utf-8")
        for snippet in ("r=8", "lora_alpha=16", "learning_rate=5e-5"):
            self.assertIn(snippet, text)

    def test_generated_artifacts_are_ignored(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for item in ("processed_data/", "qwen3_lora_output/", "*.safetensors"):
            self.assertIn(item, ignore)


if __name__ == "__main__":
    unittest.main()
