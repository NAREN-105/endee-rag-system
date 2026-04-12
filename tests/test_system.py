"""
EndeeRAG - System Tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import DocumentProcessor
from utils.text_splitter import SmartTextSplitter


def test_text_splitter():
    print("\n🧪 Test 1: Text Splitter")
    splitter = SmartTextSplitter(chunk_size=200, chunk_overlap=20)
    text = """
    Artificial Intelligence is transforming the world. Machine learning models 
    are being used in healthcare, finance, education, and many other sectors.
    Natural Language Processing allows computers to understand human language.
    Vector databases store high-dimensional embeddings for semantic search.
    Retrieval-Augmented Generation combines search with language model generation.
    """
    chunks = splitter.split_text(text)
    assert len(chunks) > 0, "Should produce at least one chunk"
    assert all(isinstance(c, dict) for c in chunks), "Chunks should be dicts"
    assert all("text" in c for c in chunks), "Each chunk needs text key"
    print(f"   ✅ Produced {len(chunks)} chunks from sample text")
    return True

def test_document_processor_text():
    print("\n🧪 Test 2: Document Processor (raw text)")
    processor = DocumentProcessor()
    doc = processor.process_text(
        "Vector databases enable semantic search by storing embeddings. "
        "Endee is a lightweight vector database for AI applications. "
        "RAG systems use vector search to retrieve relevant context.",
        source_name="test_input.txt"
    )
    assert doc.doc_id is not None
    assert doc.chunk_count > 0
    assert doc.filename == "test_input.txt"
    assert len(doc.chunks) == doc.chunk_count
    print(f"   ✅ doc_id: {doc.doc_id}")
    print(f"   ✅ chunks: {doc.chunk_count}")
    print(f"   ✅ words: {doc.metadata['word_count']}")
    return True


def test_document_processor_txt_file():
    print("\n🧪 Test 3: Document Processor (TXT file)")
    # Create a temp test file
    test_file = "tests/sample_test.txt"
    with open(test_file, "w") as f:
        f.write("""
Introduction to RAG Systems

RAG stands for Retrieval-Augmented Generation. It is a technique that combines
information retrieval with text generation using large language models.

How RAG Works

First, documents are split into chunks and converted to embeddings.
These embeddings are stored in a vector database like Endee.
When a user asks a question, the query is also embedded.
The system retrieves the most similar chunks from the vector database.
Finally, an LLM generates an answer using the retrieved context.

Benefits of RAG

RAG reduces hallucinations because the model uses real retrieved data.
It allows LLMs to answer questions about private or recent documents.
RAG is more cost-effective than fine-tuning a model on custom data.
        """)

    processor = DocumentProcessor()
    doc = processor.process_file(test_file)

    assert doc.file_type == "txt"
    assert doc.chunk_count > 0
    assert doc.metadata["word_count"] > 10
    print(f"   ✅ File type: {doc.file_type}")
    print(f"   ✅ Chunks: {doc.chunk_count}")
    print(f"   ✅ File size: {doc.file_size} bytes")

    # Cleanup
    os.remove(test_file)
    return True


def test_chunk_metadata():
    print("\n🧪 Test 4: Chunk Metadata Integrity")
    processor = DocumentProcessor()
    doc = processor.process_text(
        "Endee vector database supports cosine similarity search. "
        "It stores embeddings with metadata for filtered retrieval. "
        "Batch operations allow inserting thousands of vectors efficiently. "
        "The CRUD API supports create, read, update, and delete operations.",
        source_name="metadata_test.txt"
    )
    for chunk in doc.chunks:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "chunk_index" in chunk
        assert "source" in chunk
        assert "doc_id" in chunk
        assert chunk["doc_id"] == doc.doc_id
    print(f"   ✅ All {len(doc.chunks)} chunks have correct metadata")
    return True


def test_processor_stats():
    print("\n🧪 Test 5: Processor Statistics")
    processor = DocumentProcessor()
    processor.process_text("First document about AI and machine learning.", "doc1.txt")
    processor.process_text("Second document about vector databases and embeddings.", "doc2.txt")
    processor.process_text("Third document about RAG pipelines and Groq LLM.", "doc3.txt")

    stats = processor.get_stats()
    assert stats["total_documents"] == 3
    assert stats["total_chunks"] > 0
    assert stats["total_words"] > 0
    print(f"   ✅ Documents: {stats['total_documents']}")
    print(f"   ✅ Total chunks: {stats['total_chunks']}")
    print(f"   ✅ Total words: {stats['total_words']}")
    return True


def test_multi_format_support():
    print("\n🧪 Test 6: Multi-Format Detection")
    from src.document_processor import DocumentProcessor
    processor = DocumentProcessor()
    supported = processor.SUPPORTED_FORMATS
    assert ".pdf" in supported
    assert ".docx" in supported
    assert ".txt" in supported
    assert ".md" in supported
    print(f"   ✅ Supported formats: {supported}")
    return True


def run_all_tests():
    print("""
╔══════════════════════════════════════╗
║      EndeeRAG - Test Suite           ║
╚══════════════════════════════════════╝
""")
    tests = [
        test_text_splitter,
        test_document_processor_text,
        test_document_processor_txt_file,
        test_chunk_metadata,
        test_processor_stats,
        test_multi_format_support,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            failed += 1

    print(f"""
══════════════════════════════════════
Results: {passed} passed, {failed} failed
{'✅ ALL TESTS PASSED!' if failed == 0 else '⚠️  Some tests failed'}
══════════════════════════════════════
""")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)