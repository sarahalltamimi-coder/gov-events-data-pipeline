# Government Events Data Pipeline

An end to end data engineering pipeline built on Saudi public sector events
data for 2026 (601 events: the government entity, event type, title, dates,
venue and city).

The idea: events arrive as messages, get validated, stored in a lakehouse,
checked for quality, and at the end you can search them in natural language
(arabic) with a RAG pipeline.

## Pipeline

```
csv events
   |
   v
ingestion.py ........ kafka producer + consumer, pydantic schema gate
   |                  (2 bad messages are sent on purpose to prove
   v                   the validation rejects them)
lakehouse.py ........ delta lake: bronze (raw) -> silver (clean, no
   |                  duplicates, real dates) -> gold (summary tables)
   |                  + MERGE (upsert) + schema enforcement test
   v
quality_check.py .... 6 DAMA quality checks + a real great expectations
   |                  checkpoint + openlineage events, then the data
   |                  goes to production/ or quarantine/
   v
rag_pipeline.py ..... every event becomes a document: chunking ->
                      chromadb (vector) + bm25 (keywords) -> hybrid
                      search with RRF -> cross encoder reranking

pipeline_dag.py ..... airflow dag that runs the 4 steps in this order.
                      if airflow is not installed it runs them manually
                      in the same order.
```

## What each file does

| File | What it covers |
|------|----------------|
| `ingestion.py` | kafka producer + consumer with schema validation (pydantic) |
| `lakehouse.py` | delta lakehouse bronze/silver/gold + MERGE + schema enforcement |
| `rag_pipeline.py` | chunking, embeddings, vector index, hybrid search + reranking |
| `pipeline_dag.py` | airflow dag connecting all the modules end to end |
| `quality_check.py` | DAMA quality checks + great expectations + openlineage |

## How to run

Easiest way: open `project.ipynb` in google colab, upload the 5 .py files
and the csv from the files panel, then Run all. The notebook in this repo
already contains the outputs of a full run so you can read the results
without running anything.

Or locally:

```
pip install -r requirements.txt
python ingestion.py
python lakehouse.py
python quality_check.py
python rag_pipeline.py
python pipeline_dag.py
```
