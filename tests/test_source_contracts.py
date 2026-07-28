import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("baseline.py", "data_processor.py", "chat_persistent.py", "check_gpu.py")


class ProjectContractTests(unittest.TestCase):
    def test_required_files_exist(self):
        required = (*SCRIPTS, "README.md", "requirements.txt", ".gitignore")
        for relative_path in required:
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

    def test_python_sources_parse(self):
        for relative_path in SCRIPTS:
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            ast.parse(source, filename=relative_path)

    def test_scripts_do_not_embed_server_root_paths(self):
        for relative_path in SCRIPTS:
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("/root/", source, relative_path)

    def test_main_scripts_expose_command_line_interfaces(self):
        for relative_path in ("baseline.py", "data_processor.py", "chat_persistent.py"):
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn("argparse", source, relative_path)
            self.assertIn("if __name__ == \"__main__\":", source, relative_path)


if __name__ == "__main__":
    unittest.main()
