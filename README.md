# Semantic Spotter — Insurance Policy RAG System

This repository implements a Retrieval-Augmented Generation (RAG) pipeline for answering questions over insurance policy PDFs. It uses LangChain components for document loading, chunking, embeddings, FAISS for vector search, and an agent-based LLM workflow to produce grounded answers. Evaluation hooks for arize-phoenix are included.

## Contents
- `sematic_spotter.ipynb` — Main exploratory notebook with ingestion, indexing, retriever + agent, dataset builder, and evaluation cells.
- `dataset_builder.py` — (optional) Script-style dataset generation utilities (if present).
- `evaluation_dataset.jsonl` — Example dataset output for evaluation.
- `faiss_store/` — Persisted FAISS index (created by the notebook).
- `cache/` — Embedding cache directory used by the project.

## Quick Overview
1. Load / parse PDF policy documents into chunks.
2. Create embeddings and store them in a FAISS vectorstore.
3. Use a compressed/reranked retriever and an agent (LLM) to answer queries.
4. Build an evaluation dataset (question, ground_truth, spans) and generate RAG answers.

## Requirements
- Python 3.10+ recommended
- Windows (tested) — project uses PowerShell commands in examples

## Environment variables
- Create a `.env` file or set environment variables required by OpenAI / LangSmith, e.g.:

```
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=...
```

Load them before running the notebook (the notebook calls `load_dotenv()`).

## Running the Notebook
Open `sematic_spotter.ipynb` in Jupyter / VS Code and run cells in order. High-level steps in the notebook:

1. Install dependencies (optional cell with `%pip install ...`).
2. Import libs and load `.env`.
3. Load / parse PDFs (DoclingLoader is used when present) and create `documents`.
4. Split documents using `RecursiveCharacterTextSplitter` to create `splits`.
5. Initialize embeddings (OpenAI Embeddings) and a cache-backed embedder.
6. Create or load FAISS vector store (`create_vector_store_faiss`).
7. Create a compression + reranking retriever (`ContextualCompressionRetriever`, `FlashrankRerank`).
8. Create agent using LangChain `create_agent(...)` and a safe `system_prompt_text` extracted from LangSmith prompts.
9. Use `insurance_agent(query)` wrapper to invoke the agent safely.

Important: run expensive steps (embeddings, FAISS build) only when necessary — the notebook caches documents and the FAISS index to disk.

## Generating an Evaluation Dataset
- The notebook includes dataset-building utilities to extract facts, generate question-answer pairs, and write a `evaluation_dataset.jsonl` file.
- A `DatasetConfig` pattern is used to limit API calls (e.g., `MAX_PDFS`, `MAX_FACTS_PER_PDF`, `QUESTIONS_PER_FACT`, `ENABLE_LLM_GENERATION`). Use these settings to avoid excess cost during development.

## Generating RAG Answers for Evaluation
- Use `build_dataset()` (or the notebook cell) to produce RAG answers for rows in your evaluation dataset.
- Important: Do NOT overwrite your labeled `ground_truth` column. The notebook follows the safer pattern of storing the actual retrieved content per-row in `retrieved_context` (or `context` if you explicitly choose to replace it). Keep `ground_truth` separate so Phoenix evaluators can use it for correctness evaluation.


## Minimal Run Examples (PowerShell)

Run the notebook in JupyterLab / VS Code. To run only a few non-API checks:

```powershell
# start jupyter lab
jupyter notebook
```

To run a single notebook cell manually from PowerShell (for debugging), consider using `papermill` or executing a small script that imports functions from `dataset_builder.py`.

## Developer Tips
- When iterating on prompts, avoid editing the live `prompt` object in the notebook; use the provided `extract_prompt_text` helper and keep the system prompt stable.
- Cache embeddings and the FAISS index to speed up repeated runs — the notebook already saves `processed_documents.pkl` and the FAISS store to `./faiss_store`.
- Limit LLM usage during testing by toggling `DatasetConfig.ENABLE_LLM_GENERATION = False`.

## Next Steps / Optional Improvements
- Add a `requirements.txt` or `pyproject.toml` for reproducible installs.
- Add test harnesses / smoke tests that run without API calls (set `ENABLE_LLM_GENERATION=False`) to validate data flow.
- Consider adding CI checks that verify notebook cells with `nbval` or convert critical sections to Python modules for unit testing.

