# Public Sector Events Q2-2026

## Deliverables

1. `ingestion.py` - kafka producer and consumer with schema validation (pydantic).
   If kafka is not running it uses a simple mock so the code still works.
2. `lakehouse.py` - delta lake with bronze / silver / gold zones, plus MERGE (upsert)
   and schema enforcement (delta rejects a write with an extra column).
3. `rag_pipeline.py` - chunking, embeddings with chromadb, bm25 keyword search,
   hybrid search (reciprocal rank fusion) and cross encoder reranking.
4. `pipeline_dag.py` - airflow dag that connects everything:
   ingest -> lakehouse -> quality gate -> rag.
   If airflow is not installed it runs the tasks in the same order manually.
5. `quality_check.py` - quality checks based on the 6 DAMA dimensions, then the same
   checks with the real great expectations library, and openlineage events at the end.

## How to run

```
pip install -r requirements.txt
```

Then open `project.ipynb` and run the cells in order, it runs every deliverable and
shows the output. You can also run each file alone:

```
python ingestion.py
python lakehouse.py
python quality_check.py
python rag_pipeline.py
python pipeline_dag.py
```

Notes:
- keep the csv file in the same folder as the code
- pyspark needs java installed
- the logs are printed with loguru and also saved to pipeline.log
- the embedding model in the rag part is english (same one from the lab), the bm25
  search handles the exact arabic words like city names and request numbers
