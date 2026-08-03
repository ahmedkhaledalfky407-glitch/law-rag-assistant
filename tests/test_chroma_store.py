import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ChromaStoreTests(unittest.TestCase):
    def test_create_or_update_chroma_store_accepts_repeated_calls(self):
        module = load_module("create_chroma_store_under_test", str(Path(__file__).resolve().parents[1] / "05_create_chroma_store.py"))

        try:
            tmpdir_obj = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
            tmpdir = tmpdir_obj.name
        except TypeError:
            tmpdir = tempfile.mkdtemp()
            tmpdir_obj = None

        try:
            chunks = [{
                "text": "المادة الأولى: يبدأ العمل من تاريخ التعيين.",
                "embedding": [0.1, 0.2, 0.3],
                "book": "الأول",
                "chapter": "الأول",
                "article_number": "1",
                "chunk_id": "chunk_1",
                "source_file": "demo.txt",
            }]
            first = module.create_or_update_chroma_store(chunks, persist_directory=tmpdir, rebuild=True)
            second = module.create_or_update_chroma_store(chunks, persist_directory=tmpdir, rebuild=False)
            self.assertEqual(first["persist_directory"], tmpdir)
            self.assertEqual(second["persist_directory"], tmpdir)
            self.assertGreaterEqual(second["count"], 1)
        finally:
            if tmpdir_obj:
                try:
                    tmpdir_obj.cleanup()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
