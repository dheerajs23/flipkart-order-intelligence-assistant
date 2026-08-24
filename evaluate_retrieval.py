import os
import pickle
import faiss

from sentence_transformers import SentenceTransformer


# ==========================================
# Configuration
# ==========================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

INDEX_PATH = os.path.join(
    "vector_store",
    "policy.index"
)

METADATA_PATH = os.path.join(
    "vector_store",
    "metadata.pkl"
)

TOP_K = 3


# ==========================================
# Evaluation Queries
# ==========================================
#
# Each query has an expected policy document.
# The evaluation checks whether the expected
# document is retrieved in the top 1 and
# top 3 results.
# ==========================================

EVALUATION_QUERIES = [

    {
        "query":
            "How long can I return an apparel item?",
        "expected_doc":
            "POL001"
    },

    {
        "query":
            "What is the return window for footwear?",
        "expected_doc":
            "POL002"
    },

    {
        "query":
            "How long can eligible home products be returned?",
        "expected_doc":
            "POL004"
    },

    {
        "query":
            "Can I return an electronics item?",
        "expected_doc":
            "POL003"
    },

    {
        "query":
            "What are the return conditions for an apparel order?",
        "expected_doc":
            "POL001"
    },

    {
        "query":
            "What is the return period for shoes?",
        "expected_doc":
            "POL002"
    },

    {
        "query":
            "What is the return policy for home products?",
        "expected_doc":
            "POL004"
    },

    {
        "query":
            "What is the return policy for electronic products?",
        "expected_doc":
            "POL003"
    }
]


# ==========================================
# Load FAISS index
# ==========================================

print(
    "\n========== LOADING RAG INDEX =========="
)

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(
        f"FAISS index not found: {INDEX_PATH}"
    )

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(
        f"Metadata not found: {METADATA_PATH}"
    )

index = faiss.read_index(
    INDEX_PATH
)

with open(
    METADATA_PATH,
    "rb"
) as file:

    metadata = pickle.load(
        file
    )

print(
    "FAISS vectors:",
    index.ntotal
)

print(
    "Metadata chunks:",
    len(metadata)
)


# ==========================================
# Load embedding model
# ==========================================

print(
    "\n========== LOADING EMBEDDING MODEL =========="
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print(
    "Embedding model:",
    EMBEDDING_MODEL
)


# ==========================================
# Retrieval evaluation
# ==========================================

top1_hits = 0
top3_hits = 0

reciprocal_ranks = []

print(
    "\n========== RETRIEVAL EVALUATION =========="
)

for item in EVALUATION_QUERIES:

    query = item["query"]
    expected_doc = item["expected_doc"]

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding.astype("float32"),
        TOP_K
    )

    retrieved_docs = []

    for index_position in indices[0]:

        chunk = metadata[
            int(index_position)
        ]

        doc_id = chunk["doc_id"]

        if doc_id not in retrieved_docs:
            retrieved_docs.append(
                doc_id
            )

    # --------------------------------------
    # Top-1
    # --------------------------------------

    top1_hit = (
        len(retrieved_docs) >= 1
        and retrieved_docs[0] == expected_doc
    )

    if top1_hit:
        top1_hits += 1

    # --------------------------------------
    # Top-3
    # --------------------------------------

    top3_hit = (
        expected_doc in retrieved_docs[:3]
    )

    if top3_hit:
        top3_hits += 1

    # --------------------------------------
    # Reciprocal Rank
    # --------------------------------------

    rank = None

    for position, doc_id in enumerate(
        retrieved_docs,
        start=1
    ):

        if doc_id == expected_doc:

            rank = position

            break

    if rank is not None:

        reciprocal_ranks.append(
            1 / rank
        )

    else:

        reciprocal_ranks.append(
            0
        )

    # --------------------------------------
    # Print query result
    # --------------------------------------

    print("\nQuery:")
    print(query)

    print(
        "Expected document:",
        expected_doc
    )

    print(
        "Retrieved documents:",
        retrieved_docs
    )

    print(
        "Top-1:",
        "PASS" if top1_hit else "FAIL"
    )

    print(
        "Top-3:",
        "PASS" if top3_hit else "FAIL"
    )

    if rank is not None:

        print(
            "Rank:",
            rank
        )

    else:

        print(
            "Rank: NOT FOUND"
        )


# ==========================================
# Calculate metrics
# ==========================================

total_queries = len(
    EVALUATION_QUERIES
)

recall_at_1 = (
    top1_hits /
    total_queries
)

recall_at_3 = (
    top3_hits /
    total_queries
)

mrr = (
    sum(reciprocal_ranks) /
    total_queries
)


# ==========================================
# Print final results
# ==========================================

print(
    "\n========== RETRIEVAL EVALUATION RESULTS =========="
)

print(
    "Queries evaluated:",
    total_queries
)

print(
    "Recall@1:",
    round(recall_at_1, 4)
)

print(
    "Recall@3:",
    round(recall_at_3, 4)
)

print(
    "MRR:",
    round(mrr, 4)
)


# ==========================================
# Save evaluation results
# ==========================================

OUTPUT_PATH = (
    "retrieval_evaluation_results.txt"
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "RAG Retrieval Evaluation\n"
    )

    file.write(
        "========================\n\n"
    )

    file.write(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}\n"
    )

    file.write(
        f"Queries evaluated: "
        f"{total_queries}\n"
    )

    file.write(
        f"Recall@1: "
        f"{recall_at_1:.4f}\n"
    )

    file.write(
        f"Recall@3: "
        f"{recall_at_3:.4f}\n"
    )

    file.write(
        f"MRR: "
        f"{mrr:.4f}\n"
    )

print(
    "\nEvaluation results saved to:"
)

print(
    OUTPUT_PATH
)