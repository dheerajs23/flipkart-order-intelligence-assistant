import os
import pickle
import faiss

from sentence_transformers import SentenceTransformer
from policy_kb.policies import POLICY_DOCUMENTS


# ==========================================
# Configuration
# ==========================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

VECTOR_DIR = "vector_store"

INDEX_PATH = os.path.join(
    VECTOR_DIR,
    "policy.index"
)

METADATA_PATH = os.path.join(
    VECTOR_DIR,
    "metadata.pkl"
)


# ==========================================
# Create vector-store directory
# ==========================================

os.makedirs(
    VECTOR_DIR,
    exist_ok=True
)


# ==========================================
# Sentence-wise chunking
# ==========================================

chunks = []

for document in POLICY_DOCUMENTS:

    sentences = [
        sentence.strip()
        for sentence in document["text"].split(".")
        if sentence.strip()
    ]

    for sentence in sentences:

        chunks.append({
            "doc_id": document["doc_id"],
            "title": document["title"],
            "text": sentence + "."
        })


print("\n========== RAG DOCUMENTS ==========")

print(
    "Documents:",
    len(POLICY_DOCUMENTS)
)

print(
    "Chunks:",
    len(chunks)
)


# ==========================================
# Load embedding model
# ==========================================

print("\n========== LOADING EMBEDDING MODEL ==========")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    "Embedding model:",
    EMBEDDING_MODEL
)


# ==========================================
# Generate embeddings
# ==========================================

texts = [
    chunk["text"]
    for chunk in chunks
]

print("\n========== GENERATING EMBEDDINGS ==========")

embeddings = embedding_model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

print(
    "Embedding shape:",
    embeddings.shape
)


# ==========================================
# Build FAISS index
# ==========================================

print("\n========== BUILDING FAISS INDEX ==========")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings.astype("float32")
)

print(
    "FAISS vectors:",
    index.ntotal
)


# ==========================================
# Save FAISS index
# ==========================================

faiss.write_index(
    index,
    INDEX_PATH
)


# ==========================================
# Save metadata
# ==========================================

with open(
    METADATA_PATH,
    "wb"
) as file:

    pickle.dump(
        chunks,
        file
    )


# ==========================================
# Final result
# ==========================================

print("\n========== RAG INDEX CREATED ==========")

print(
    "Index saved to:",
    INDEX_PATH
)

print(
    "Metadata saved to:",
    METADATA_PATH
)

print(
    "Documents indexed:",
    len(POLICY_DOCUMENTS)
)

print(
    "Chunks indexed:",
    len(chunks)
)