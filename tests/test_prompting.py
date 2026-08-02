import importlib.util
import os
import unittest
from pathlib import Path


def load_prompting_module():
    module_path = Path(__file__).resolve().parents[1] / "07_prompting.py"
    spec = importlib.util.spec_from_file_location("prompting_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PromptingApiConfigTests(unittest.TestCase):
    def test_configure_api_settings_updates_environment_values(self):
        module = load_prompting_module()

        module.configure_api_settings(api_key="test-key", model="test/model")

        self.assertEqual(module.get_api_config()[0], "test-key")
        self.assertEqual(module.get_api_config()[1], "test/model")
        self.assertEqual(os.environ.get("OPENROUTER_API_KEY"), "test-key")
        self.assertEqual(os.environ.get("OPENROUTER_MODEL"), "test/model")


if __name__ == "__main__":
    unittest.main()
