"""
EndeeRAG - Main Entry Point
Advanced Document Intelligence System
"""

import os
import sys
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline
from src.agents import MultiAgentSystem


def validate_setup():
    """Check all required config is present"""
    print("🔍 Validating setup...")
    issues = []

    if not Config.GROQ_API_KEY or Config.GROQ_API_KEY == "your_groq_api_key_here":
        issues.append("❌ GROQ_API_KEY not set in .env")
    if not Config.ENDEE_API_KEY or Config.ENDEE_API_KEY == "your_endee_api_key_here":
        issues.append("❌ ENDEE_API_KEY not set in .env")

    if issues:
        print("\n".join(issues))
        print("\n💡 Edit your .env file and add the missing API keys.")
        return False

    print("✅ Config valid")
    return True


def demo_mode():
    """Run a demo without API keys using mock responses"""
    print("\n🎭 DEMO MODE (no API keys required)")
    print("=" * 50)

    processor = DocumentProcessor()

    # Create a sample document
    sample_text = """
    Artificial Intelligence (AI) is transforming industries worldwide.
    Machine learning models can now process vast amounts of data to identify patterns.
    Natural Language Processing (NLP) enables computers to understand human language.
    Vector databases store embeddings for efficient semantic search.
    Retrieval-Augmented Generation (RAG) combines search with language models.
    This hybrid approach improves accuracy and reduces hallucinations in AI systems.
    Companies like OpenAI, Anthropic, and Google are leading AI research.
    The future of AI includes multimodal models that process text, images, and audio.
    """

    doc = processor.process_text(sample_text, source_name="ai_overview.txt")
    print(f"\n📄 Processed sample document:")
    print(f"   Chunks: {doc.chunk_count}")
    print(f"   Words: {doc.metadata['word_count']}")

    print("\n✅ Demo complete! System is working correctly.")
    print("\nTo use the full system:")
    print("  1. Add API keys to .env")
    print("  2. Run: python main.py --mode cli")
    print("  3. Or run: python ui/web_app.py")


def cli_mode():
    """Interactive command-line interface"""
    print("\n🖥️  CLI MODE")
    print("=" * 50)

    if not validate_setup():
        return

    pipeline = RAGPipeline()
    processor = DocumentProcessor()

    print("\nEndeeRAG CLI - Type 'help' for commands\n")

    while True:
        try:
            user_input = input(">>> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("👋 Goodbye!")
                break

            elif user_input.lower() == "help":
                print("""
Commands:
  load <path>     - Load a document or directory
  ask <question>  - Ask a question
  stats           - Show system statistics
  clear           - Clear conversation history
  sources         - List all document sources
  exit            - Quit
""")

            elif user_input.lower().startswith("load "):
                path = user_input[5:].strip()
                if os.path.isfile(path):
                    doc = processor.process_file(path)
                    from src.vector_store import VectorStore
                    vs = VectorStore()
                    vs.insert_document(doc)
                    print(f"✅ Loaded: {doc.filename}")
                elif os.path.isdir(path):
                    docs = processor.process_directory(path)
                    from src.vector_store import VectorStore
                    vs = VectorStore()
                    vs.insert_many(docs)
                else:
                    print(f"❌ Path not found: {path}")

            elif user_input.lower().startswith("ask "):
                question = user_input[4:].strip()
                response = pipeline.query(question)
                print(f"\n💬 Answer:\n{response.answer}")
                print(f"\n📚 Sources: {[s['source'] for s in response.sources]}")
                print(f"⏱️  {response.latency_ms}ms | 🎯 Confidence: {response.confidence:.0%}\n")

            elif user_input.lower() == "stats":
                rag_stats = pipeline.get_stats()
                vs_stats = pipeline.vector_store.get_store_stats()
                print(f"\n📊 Stats:")
                print(f"  Queries: {rag_stats['total_queries']}")
                print(f"  Vectors: {vs_stats['total_vectors']}")
                print(f"  Sources: {vs_stats['unique_sources']}")

            elif user_input.lower() == "clear":
                pipeline.clear_history()

            elif user_input.lower() == "sources":
                sources = pipeline.vector_store.get_all_sources()
                if sources:
                    print("\n📁 Documents:")
                    for s in sources:
                        print(f"  • {s}")
                else:
                    print("No documents loaded yet.")

            else:
                # Treat any other input as a question
                response = pipeline.query(user_input)
                print(f"\n💬 {response.answer}")
                print(f"📚 Sources: {[s['source'] for s in response.sources]}\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def quick_test():
    """Quick system test"""
    print("\n🧪 QUICK TEST")
    print("=" * 50)

    if not validate_setup():
        return

    print("\n1. Testing Document Processor...")
    processor = DocumentProcessor()
    doc = processor.process_text(
        "Machine learning is a subset of artificial intelligence. "
        "Deep learning uses neural networks with many layers.",
        source_name="test.txt"
    )
    print(f"   ✅ Processed: {doc.chunk_count} chunks")

    print("\n2. Testing Vector Store...")
    vs = VectorStore()
    result = vs.insert_document(doc)
    print(f"   ✅ Inserted: {result['chunks_inserted']} vectors")

    print("\n3. Testing RAG Pipeline...")
    pipeline = RAGPipeline()
    response = pipeline.query("What is machine learning?")
    print(f"   ✅ Answer: {response.answer[:100]}...")

    print("\n4. Testing Multi-Agent System...")
    mas = MultiAgentSystem()
    agent_result = mas.run("Explain deep learning", verbose=False)
    print(f"   ✅ Agents used: {agent_result.agents_used}")

    print("\n🎉 All tests passed!")


def main():
    """Main entry point"""
    print("""
╔════════════════════════════════════════╗
║         EndeeRAG System v1.0           ║
║   Advanced Document Intelligence       ║
╚════════════════════════════════════════╝
""")

    mode = "demo"
    if len(sys.argv) > 1:
        flag = sys.argv[1].lower()
        if flag == "--mode" and len(sys.argv) > 2:
            mode = sys.argv[2].lower()
        elif flag in ("--test", "-t"):
            mode = "test"
        elif flag in ("--cli", "-c"):
            mode = "cli"
        elif flag in ("--demo", "-d"):
            mode = "demo"

    if mode == "demo":
        demo_mode()
    elif mode == "cli":
        cli_mode()
    elif mode == "test":
        quick_test()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python main.py [--mode demo|cli|test]")


if __name__ == "__main__":
    main()