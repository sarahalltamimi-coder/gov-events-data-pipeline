# rag pipeline: chunking -> embeddings -> vector index -> hybrid search -> reranking

import re

import chromadb
import numpy as np
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

CSV_FILE = "Public_Sector_Events_Q2_2026_CSV.csv"

COLS = {
    "اسم الجهة الحكومية": "entity",
    "نوع الفعالية": "event_type",
    "عنوان الفعالية": "title",
    "تاريخ بداية الفعالية": "start_date",
    "تاريخ نهاية الفعالية": "end_date",
    "نوع الموقع": "venue_type",
    "موقع الفعالية": "venue",
    "المدينة": "city",
    "رقم الطلب": "request_id",
}


def make_documents():
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig", dtype=str)
    df = df.rename(columns=COLS)
    df = df.drop_duplicates(subset=["request_id"])

    docs = []
    for _, r in df.iterrows():
        title = re.sub(r"\s+", " ", r["title"].strip())
        text = (f"نظمت {r['entity']} فعالية {r['event_type']} بعنوان {title}. "
                f"المكان: {r['venue']} ({r['venue_type']}) في مدينة {r['city']}. "
                f"التاريخ من {r['start_date']} الى {r['end_date']} ورقم الطلب {r['request_id']}.")
        docs.append({"id": r["request_id"], "text": text})
    logger.info(f"made {len(docs)} documents")
    return docs


def chunk_documents(docs):
    # split every document into chunks of 2 sentences with 1 sentence overlap
    chunks = []
    for doc in docs:
        sentences = re.split(r"(?<=[.!?])\s+", doc["text"].strip())
        for i in range(0, len(sentences)):
            text = " ".join(sentences[i:i + 2])
            if text.strip():
                chunks.append({"id": f"{doc['id']}_{i}", "text": text})
    return chunks


def build_vector_index(chunks):
    # chromadb stores the embeddings and searches with hnsw
    logger.info("building the vector index, first run downloads the model...")
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.Client()
    try:
        client.delete_collection("events")  # so we can rerun without errors
    except Exception:
        pass
    collection = client.get_or_create_collection("events", embedding_function=ef)
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
    )
    logger.success(f"indexed {len(chunks)} chunks")
    return collection


def bm25_search(bm25, chunks, query, top_k=6):
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [chunks[i] for i, s in ranked[:top_k]]


def hybrid_search(vector_hits, bm25_hits, top_k=6):
    # reciprocal rank fusion: combine the two lists, k=60 is the standard value
    k = 60
    scores = {}
    texts = {}
    for rank, hit in enumerate(vector_hits):
        scores[hit["id"]] = scores.get(hit["id"], 0) + 1 / (k + rank + 1)
        texts[hit["id"]] = hit["text"]
    for rank, hit in enumerate(bm25_hits):
        scores[hit["id"]] = scores.get(hit["id"], 0) + 1 / (k + rank + 1)
        texts[hit["id"]] = hit["text"]
    best = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    return [{"id": i, "text": texts[i]} for i in best]


def rerank(query, candidates, top_k=3):
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for s, c in ranked[:top_k]]


def build_prompt(query, docs):
    context = "\n".join(f"[{i+1}] {d['text']}" for i, d in enumerate(docs))
    return (f"اجب على السؤال من السياق التالي فقط.\n\n"
            f"السياق:\n{context}\n\nالسؤال: {query}\nالجواب:")


def evaluate(query, docs, model):
    # simple version of the ragas metrics using cosine similarity
    q = model.encode(query, normalize_embeddings=True)
    sims = [float(np.dot(q, model.encode(d["text"], normalize_embeddings=True)))
            for d in docs]
    precision = sum(s > 0.3 for s in sims) / len(sims)
    return {"context_precision": round(precision, 2),
            "avg_similarity": round(sum(sims) / len(sims), 2)}


def main():
    print("---- rag pipeline ----")

    docs = make_documents()
    chunks = chunk_documents(docs)
    print(f"{len(docs)} documents -> {len(chunks)} chunks")

    collection = build_vector_index(chunks)
    bm25 = BM25Okapi([c["text"].lower().split() for c in chunks])
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    queries = [
        "ما هي ورش العمل في مدينة جدة؟",
        "ما الفعاليات التي نظمها معهد الادارة العامة؟",
        "ما تفاصيل الفعالية رقم GOV-2026-0040؟",
    ]

    for query in queries:
        print("\n==============================")
        print("query:", query)

        # 1) vector search
        res = collection.query(query_texts=[query], n_results=6)
        vector_hits = [{"id": i, "text": t}
                       for i, t in zip(res["ids"][0], res["documents"][0])]

        # 2) keyword search
        bm25_hits = bm25_search(bm25, chunks, query)

        # 3) combine with rrf then rerank
        combined = hybrid_search(vector_hits, bm25_hits)
        top_docs = rerank(query, combined)

        print("top 3 results after reranking:")
        for i, d in enumerate(top_docs, 1):
            print(f"  {i}. {d['text'][:100]}")

        prompt = build_prompt(query, top_docs)
        print("prompt for the llm (first 200 chars):")
        print(" ", prompt[:200].replace("\n", " "))

        metrics = evaluate(query, top_docs, embed_model)
        print("metrics:", metrics)
        if metrics["context_precision"] >= 0.5:
            logger.success("retrieval looks good")
        else:
            logger.warning("low precision, an arabic embedding model would help here")

    logger.success("rag pipeline done")


if __name__ == "__main__":
    main()
