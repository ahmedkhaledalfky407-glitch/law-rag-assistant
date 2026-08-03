"""Comprehensive end-to-end test of the RAG pipeline - fixed for Windows."""
import os
import shutil
import unittest

os.environ['OPENROUTER_API_KEY'] = os.environ.get('OPENROUTER_API_KEY', 'test-key')
os.environ['OPENROUTER_MODEL'] = 'openai/gpt-4o-mini'
os.environ['EMBEDDING_PROVIDER'] = 'openai'
os.environ['EMBEDDING_MODEL'] = 'openai/text-embedding-3-small'

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


def load_mod(name, file):
    spec = spec_from_file_location(name, Path(file))
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


TEST_ROOT = Path(__file__).resolve().parents[1] / 'tmp_test_rag'
if TEST_ROOT.exists():
    shutil.rmtree(TEST_ROOT)
TEST_ROOT.mkdir()


def cleanup_test_dir(name):
    d = TEST_ROOT / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    return d


class TestRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.docs_mod = load_mod('docs', cls.root / '01_documents.py')
        cls.pre_mod = load_mod('pre', cls.root / '02_preprocessing.py')
        cls.chunk_mod = load_mod('chunk', cls.root / '03_chunking.py')
        cls.vec_mod = load_mod('vec', cls.root / '04_vector_representation.py')
        cls.store_mod = load_mod('store', cls.root / '05_create_chroma_store.py')
        cls.retrieve_mod = load_mod('retrieve', cls.root / '06_retrieve_context.py')
        cls.prompt_mod = load_mod('prompt', cls.root / '07_prompting.py')

    def test_preprocessing_preserves_source_file(self):
        doc = {'raw_text': 'المادة 1: نص القانون.', 'source_file': 'test.txt'}
        articles = self.pre_mod.process_documents([doc])
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['source_file'], 'test.txt')

    def test_chunking_preserves_source_file(self):
        articles = [{'book': '1', 'chapter': '1', 'article_number': '1', 'article_text': 'نص طويل ' * 200, 'source_file': 'test.txt'}]
        chunks = self.chunk_mod.chunk_articles(articles, max_tokens=50, overlap_ratio=0.2)
        self.assertTrue(len(chunks) > 0)
        for chunk in chunks:
            self.assertEqual(chunk.get('source_file'), 'test.txt')

    def test_arabic_word_number_normalization(self):
        self.assertEqual(self.pre_mod._normalize_article_number('الأولى'), '1')
        self.assertEqual(self.pre_mod._normalize_article_number('الثانية'), '2')
        self.assertEqual(self.pre_mod._normalize_article_number('الثالثة عشرة'), '13')
        self.assertEqual(self.pre_mod._normalize_article_number('العاشرة'), '10')
        self.assertEqual(self.pre_mod._normalize_article_number('١٢٩'), '129')
        self.assertEqual(self.pre_mod._normalize_article_number('المادة 1'), '1')
        self.assertEqual(self.pre_mod._normalize_article_number('رقم 5'), '5')

    def test_chroma_store_appends_without_deleting(self):
        tmpdir = str(cleanup_test_dir('test_append'))
        try:
            chunks1 = [{'text': 'نص أول', 'embedding': [0.1]*1536, 'article_number': '1', 'source_file': 'a.txt'}]
            result1 = self.store_mod.create_or_update_chroma_store(chunks1, persist_directory=tmpdir, collection_name='test_append', rebuild=True)
            self.assertEqual(result1['count'], 1)

            chunks2 = [{'text': 'نص ثاني', 'embedding': [0.2]*1536, 'article_number': '2', 'source_file': 'b.txt'}]
            result2 = self.store_mod.create_or_update_chroma_store(chunks2, persist_directory=tmpdir, collection_name='test_append', rebuild=False)
            self.assertEqual(result2['count'], 1)

            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(path=tmpdir, settings=Settings(allow_reset=True, anonymized_telemetry=False))
            col = client.get_collection('test_append')
            self.assertEqual(col.count(), 2)
        finally:
            cleanup_test_dir('test_append')

    def test_chroma_store_rebuild_deletes_and_creates(self):
        tmpdir = str(cleanup_test_dir('test_rebuild'))
        try:
            chunks1 = [{'text': 'نص أول', 'embedding': [0.1]*1536, 'article_number': '1', 'source_file': 'a.txt'}]
            self.store_mod.create_or_update_chroma_store(chunks1, persist_directory=tmpdir, collection_name='test_rebuild', rebuild=True)

            chunks2 = [{'text': 'نص ثاني', 'embedding': [0.2]*1536, 'article_number': '2', 'source_file': 'b.txt'}]
            self.store_mod.create_or_update_chroma_store(chunks2, persist_directory=tmpdir, collection_name='test_rebuild', rebuild=True)

            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(path=tmpdir, settings=Settings(allow_reset=True, anonymized_telemetry=False))
            col = client.get_collection('test_rebuild')
            self.assertEqual(col.count(), 1)
        finally:
            cleanup_test_dir('test_rebuild')

    def test_retrieval_returns_results(self):
        tmpdir = str(cleanup_test_dir('test_retrieval'))
        try:
            chunks = [
                {'text': 'للعامل إجازة سنوية بأجر.', 'embedding': [0.1]*1536, 'article_number': '124', 'source_file': 'test.txt'},
                {'text': 'للعامل إجازة بأجر في العطلات.', 'embedding': [0.2]*1536, 'article_number': '129', 'source_file': 'test.txt'},
            ]
            self.store_mod.create_or_update_chroma_store(chunks, persist_directory=tmpdir, collection_name='test_retrieval', rebuild=True)

            os.environ['CHROMA_DB_PATH'] = tmpdir
            os.environ['CHROMA_COLLECTION'] = 'test_retrieval'

            results = self.retrieve_mod.retrieve_context('إجازات العمال', top_k=2)
            self.assertTrue(len(results) > 0, f'Expected results, got: {results}')
        finally:
            cleanup_test_dir('test_retrieval')

    def test_full_pipeline_end_to_end(self):
        tmpdir = str(cleanup_test_dir('test_e2e'))
        try:
            sample = 'الكتاب الأول\nالباب الأول\nالمادة الأولى: يبدأ العمل من تاريخ التعيين.\nالمادة الثانية: تلتزم الجهات بأحكام هذا القانون.\nالمادة 3: يعتبر هذا القانون هو القانون العام.\n'

            import tempfile as tf
            with tf.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                f.write(sample)
                f.flush()
                documents = self.docs_mod.load_documents(f.name)

            articles = self.pre_mod.process_documents(documents)
            self.assertTrue(len(articles) > 0)

            chunks = self.chunk_mod.chunk_articles(articles)
            self.assertTrue(len(chunks) > 0)

            embedded = self.vec_mod.build_embeddings(chunks)
            self.assertTrue(len(embedded) > 0)
            for chunk in embedded:
                self.assertEqual(len(chunk['embedding']), 1536)

            result = self.store_mod.create_or_update_chroma_store(
                embedded,
                persist_directory=tmpdir,
                collection_name='test_e2e',
                rebuild=True,
            )
            self.assertEqual(result['count'], len(embedded))

            os.environ['CHROMA_DB_PATH'] = tmpdir
            os.environ['CHROMA_COLLECTION'] = 'test_e2e'

            results = self.retrieve_mod.retrieve_context('تاريخ التعيين', top_k=3)
            self.assertTrue(len(results) > 0, f'Expected retrieval results, got: {results}')
        finally:
            cleanup_test_dir('test_e2e')


if __name__ == '__main__':
    unittest.main()
