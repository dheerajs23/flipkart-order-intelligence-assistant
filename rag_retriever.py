import pickle
import faiss

from sentence_transformers import SentenceTransformer


# ==========================================
# Configuration
# ==========================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

INDEX_PATH = "vector_store/policy.index"
METADATA_PATH = "vector_store/metadata.pkl"


# ==========================================
# Load index and metadata
# ==========================================

print("Loading policy vector index...")

index = faiss.read_index(
    INDEX_PATH
)

with open(
    METADATA_PATH,
    "rb"
) as file:

    metadata = pickle.load(file)


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    "Policy index loaded successfully."
)


# ==========================================
# Retrieve policies
# ==========================================

def retrieve_policy(
    query: str,
    top_k: int = 3
):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding.astype("float32"),
        top_k
    )

    results = []

    # --------------------------------------
    # Parent-document mapping
    # --------------------------------------

    seen_documents = set()

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        if index_position < 0:
            continue

        chunk = metadata[index_position]

        doc_id = chunk["doc_id"]

        # Avoid returning duplicate parent documents
        if doc_id in seen_documents:
            continue

        seen_documents.add(doc_id)

        results.append({
            "doc_id": doc_id,
            "title": chunk["title"],
            "text": chunk["text"],
            "score": float(score)
        })

    return results


# ==========================================
# Simple test
# ==========================================

if __name__ == "__main__":

    query = "How long can I return an apparel item?"

    print("\n========== POLICY RETRIEVAL ==========")

    print(
        "Query:",
        query
    )

    results = retrieve_policy(
        query,
        top_k=3
    )

    for number, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nResult {number}"
        )

        print(
            "Document:",
            result["doc_id"]
        )

        print(
            "Title:",
            result["title"]
        )

        print(
            "Score:",
            round(result["score"], 4)
        )

        print(
            "Text:",
            result["text"]
        )