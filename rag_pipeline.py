"""
RAG Pipeline — Wikipedia + ChromaDB + Groq (llama-3.3-70b-versatile)

Flow:
  1. Fetch a Wikipedia article
  2. Split into overlapping chunks
  3. Embed chunks with sentence-transformers
  4. Store in a ChromaDB vector database
  5. Answer questions by retrieving relevant chunks first, then querying Groq
"""

import os
import textwrap

import chromadb
import groq
import wikipediaapi
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"
EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, lightweight sentence embeddings
CHUNK_SIZE = 500                    # characters per chunk
CHUNK_OVERLAP = 100                 # overlap between consecutive chunks
TOP_K = 4                           # number of chunks to retrieve per query
CHROMA_COLLECTION = "wikipedia_rag"

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_wikipedia(topic: str) -> tuple[str, str]:
    """Return (title, full_text) for a Wikipedia article."""
    wiki = wikipediaapi.Wikipedia(
        user_agent="llm-reliability-rag/1.0",
        language="en",
    )
    page = wiki.page(topic)
    if not page.exists():
        raise ValueError(f"Wikipedia page not found: '{topic}'")
    return page.title, page.text


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


def build_vector_store(
    chunks: list[str],
    embedder: SentenceTransformer,
    collection_name: str = CHROMA_COLLECTION,
) -> chromadb.Collection:
    """Embed chunks and store them in an in-memory ChromaDB collection."""
    client = chromadb.Client()

    # Clear existing collection if it already exists
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(collection_name)

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk-{i}" for i in range(len(chunks))],
    )
    return collection


def retrieve(
    query: str,
    collection: chromadb.Collection,
    embedder: SentenceTransformer,
    top_k: int = TOP_K,
) -> list[str]:
    """Return the top-k most relevant chunks for a query."""
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0]


def answer(
    question: str,
    context_chunks: list[str],
    groq_client: groq.Groq,
) -> str:
    """Send retrieved context + question to Groq and return the answer."""
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        f"You are a helpful assistant. Answer the question using only the context below.\n"
        f"If the context doesn't contain enough information, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(topic: str, questions: list[str]) -> None:
    groq_client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
    embedder = SentenceTransformer(EMBED_MODEL)

    # 1. Fetch Wikipedia article
    print(f"\nFetching Wikipedia article: '{topic}'...")
    title, text = fetch_wikipedia(topic)
    print(f"  Retrieved '{title}' — {len(text):,} characters")

    # 2. Chunk
    chunks = chunk_text(text)
    print(f"  Split into {len(chunks)} chunks ({CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap)")

    # 3 & 4. Embed + store
    collection = build_vector_store(chunks, embedder)
    print(f"  Stored in ChromaDB collection '{CHROMA_COLLECTION}'\n")

    # 5. Answer each question
    for i, question in enumerate(questions, start=1):
        print(f"{'─' * 60}")
        print(f"Q{i}: {question}")
        print(f"{'─' * 60}")

        relevant_chunks = retrieve(question, collection, embedder)
        print(f"[Retrieved {len(relevant_chunks)} chunks]\n")

        response = answer(question, relevant_chunks, groq_client)
        print(textwrap.fill(response, width=80))
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TOPIC = "Artificial intelligence"

    QUESTIONS = [
        "What are the main applications of artificial intelligence?",
        "What ethical concerns are associated with AI?",
        "How does machine learning differ from traditional programming?",
        "Who are some of the key pioneers in AI research?",
        "What is the Turing test and why does it matter?",
    ]

    run_pipeline(TOPIC, QUESTIONS)
