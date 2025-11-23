# ---------------------------
# FIXED VERSION (supports multiple Document objects)
# ---------------------------

import os
import json
import glob
from typing import List, Dict

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

PDF_DIR = "./Policy+Documents"
OUTPUT_JSONL = "evaluation_dataset.jsonl"


# ---------------------------
# Helper: Write JSONL
# ---------------------------
def write_jsonl(path, data_rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in data_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------
# FIX: Extract facts from PDFs
# Supports multiple Document objects
# ---------------------------
def extract_facts_from_pdf(pdf_path: str) -> List[Dict]:

    loader = DoclingLoader(
        file_path=pdf_path,
        export_type=ExportType.DOC_CHUNKS
    )

    docs = loader.load()     # <-- returns LIST of Document

    facts = []

    patterns = {
        "coverage_amount": ["coverage", "insured amount", "sum assured"],
        "eligibility": ["eligible", "eligibility", "age"],
        "claim_process": ["claim", "how to file", "process", "steps"],
        "waiting_period": ["waiting period", "cooling period"],
        "exclusions": ["not covered", "exclusion", "excluded"],
        "renewal": ["renewal", "renew"],
    }

    for doc in docs:
        text = doc.page_content
        metadata = doc.metadata
        tables = metadata.get("tables", []) if isinstance(metadata, dict) else []

        # TEXT facts
        for field, keywords in patterns.items():
            for kw in keywords:
                if kw.lower() in text.lower():
                    idx = text.lower().find(kw.lower())
                    snippet = text[max(0, idx - 150): idx + 300]

                    facts.append({
                        "field": field,
                        "value": snippet.strip(),
                        "doc_id": os.path.basename(pdf_path),
                        "source_type": "text"
                    })
                    break

        # TABLE facts (if exists)
        if tables:
            for i, table in enumerate(tables):
                facts.append({
                    "field": f"table_{i}",
                    "value": str(table),
                    "doc_id": os.path.basename(pdf_path),
                    "source_type": "table"
                })

    return facts


# ---------------------------
# LLM questions
# ---------------------------
def generate_questions(fact_value: str, field: str):
    prompt = f"""
    Generate 5 diverse user questions based on this FACT:

    {fact_value}

    Field: {field}
    """

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    lines = response.output_text.split("\n")

    cleaned = []
    for line in lines:
        line = line.strip().lstrip("-• ").strip()
        if line:
            cleaned.append(line)

    return cleaned[:5]


# ---------------------------
# Hallucination checks
# ---------------------------
NO_ANSWER_QUESTIONS = [
    "Does this policy cover dental implants?",
    "Is pregnancy covered from day 1?",
    "Can I claim twice in the same day?",
    "Is cosmetic surgery included in the policy?",
    "Does this policy include home nursing?"
]

def build_no_answer_items(pdf_name: str):
    return [
        {
            "query": q,
            "expected_answer": "Not present in the document.",
            "type": "no_answer",
            "doc_id": pdf_name,
            "ground_truth_span": "",
            "category": "hallucination_check"
        }
        for q in NO_ANSWER_QUESTIONS
    ]


# ---------------------------
# Dataset row
# ---------------------------
def build_dataset_row(query, fact_value, pdf_name, field):
    return {
        "query": query,
        "expected_answer": fact_value[:300],
        "doc_id": pdf_name,
        "ground_truth_span": fact_value,
        "type": "fact",
        "category": field
    }


# ---------------------------
# Build dataset
# ---------------------------
def build_dataset():
    dataset = []
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))

    for pdf in pdf_files:
        print(f"Processing {pdf}...")

        facts = extract_facts_from_pdf(pdf)

        for fact in facts:
            fact_value = fact["value"]
            field = fact["field"]
            doc_name = fact["doc_id"]

            questions = generate_questions(fact_value, field)

            for q in questions:
                dataset.append(build_dataset_row(q, fact_value, doc_name, field))

        dataset.extend(build_no_answer_items(os.path.basename(pdf)))

    write_jsonl(OUTPUT_JSONL, dataset)
    print(f"Saved {len(dataset)} items → {OUTPUT_JSONL}")


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    build_dataset()
