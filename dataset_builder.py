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

# ====================================================================================
# CONFIGURATION: Control API calls and dataset size
# ====================================================================================
class DatasetConfig:
    """Production-grade configuration for dataset generation."""
    
    # PDF processing
    MAX_PDFS = None                          # Process only first N PDFs (None = all)
    MAX_FACTS_PER_PDF = 5                 # Extract max N facts per document
    
    # LLM generation
    QUESTIONS_PER_FACT = 3                # Generate N questions per fact (was 5)
    ENABLE_LLM_GENERATION = True          # Toggle LLM calls on/off
    
    # Hallucination checks
    INCLUDE_NO_ANSWER_CHECKS = True       # Include queries with no answer
    NO_ANSWER_SAMPLE_SIZE = 3             # How many no-answer questions per PDF
    
    # Output
    OUTPUT_JSONL = "evaluation_dataset.jsonl"
    BATCH_SIZE = 10                       # Process facts in batches for monitoring
    
    @classmethod
    def get_summary(cls):
        """Print configuration summary."""
        return f"""
╔════════════════════════════════════════╗
║    DATASET GENERATION CONFIG           ║
╠════════════════════════════════════════╣
║ Max PDFs:              {cls.MAX_PDFS or 'ALL'}
║ Max Facts/PDF:         {cls.MAX_FACTS_PER_PDF}
║ Questions/Fact:        {cls.QUESTIONS_PER_FACT}
║ LLM Generation:        {'✓ ENABLED' if cls.ENABLE_LLM_GENERATION else '✗ DISABLED'}
║ Hallucination Checks:  {'✓ ENABLED' if cls.INCLUDE_NO_ANSWER_CHECKS else '✗ DISABLED'}
║ Output File:           {cls.OUTPUT_JSONL}
╚════════════════════════════════════════╝
        """


PDF_DIR = "./Policy+Documents"


# ====================================================================================
# Helper: Write JSONL with progress tracking
# ====================================================================================
def write_jsonl(path, data_rows):
    """Write dataset rows to JSONL format with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        for row in data_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"✓ Wrote {len(data_rows)} rows to {path}")


# ====================================================================================
# FIX: Extract facts from PDFs with size limits
# Supports multiple Document objects + limits
# ====================================================================================
def extract_facts_from_pdf(pdf_path: str, max_facts: int = None) -> List[Dict]:
    """
    Extract structured facts from PDF using DoclingLoader.
    
    Args:
        pdf_path: Path to PDF file
        max_facts: Maximum number of facts to extract (None = unlimited)
        
    Returns:
        List of fact dictionaries with field, value, doc_id, source_type
    """
    try:
        loader = DoclingLoader(
            file_path=pdf_path,
            export_type=ExportType.DOC_CHUNKS
        )

        docs = loader.load()     # Returns LIST of Document objects

        facts = []
        fact_count = 0

        # Keyword patterns for structured fact extraction
        patterns = {
            "coverage_amount": ["coverage", "insured amount", "sum assured"],
            "eligibility": ["eligible", "eligibility", "age", "requirements"],
            "claim_process": ["claim", "how to file", "process", "steps"],
            "waiting_period": ["waiting period", "cooling period"],
            "exclusions": ["not covered", "exclusion", "excluded"],
            "renewal": ["renewal", "renew", "premium"],
        }

        # Extract facts from each document
        for doc in docs:
            # STOP if we've hit the max facts limit
            if max_facts and fact_count >= max_facts:
                break
                
            text = doc.page_content
            metadata = doc.metadata

            # TEXT facts: keyword-based extraction
            for field, keywords in patterns.items():
                if max_facts and fact_count >= max_facts:
                    break
                    
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
                        fact_count += 1
                        break

            # TABLE facts: extract if available
            tables = metadata.get("tables", []) if isinstance(metadata, dict) else []
            if tables and (not max_facts or fact_count < max_facts):
                for i, table in enumerate(tables):
                    if max_facts and fact_count >= max_facts:
                        break
                    facts.append({
                        "field": f"table_{i}",
                        "value": str(table)[:500],  # Truncate large tables
                        "doc_id": os.path.basename(pdf_path),
                        "source_type": "table"
                    })
                    fact_count += 1

        print(f"  → Extracted {len(facts)} facts from {os.path.basename(pdf_path)}")
        return facts
    
    except Exception as e:
        print(f"  ✗ Error processing {pdf_path}: {str(e)}")
        return []


# ====================================================================================
# LLM question generation with API cost awareness
# ====================================================================================
def generate_questions(fact_value: str, field: str, num_questions: int = 3) -> List[str]:
    """
    Generate diverse questions from a fact using GPT.
    
    Args:
        fact_value: The fact/context to generate questions from
        field: Category field (e.g., 'coverage_amount')
        num_questions: Number of questions to generate
        
    Returns:
        List of generated questions
        
    Note:
        This function makes API calls. Use DatasetConfig to control frequency.
    """
    if not DatasetConfig.ENABLE_LLM_GENERATION:
        # Return placeholder questions if LLM is disabled
        return [f"Question {i+1} about {field}" for i in range(num_questions)]
    
    try:
        prompt = f"""Generate {num_questions} diverse, specific user questions based on this FACT from an insurance policy:

FACT: {fact_value[:500]}

Field Category: {field}

Requirements:
- Each question should be answerable using ONLY the provided fact
- Questions should be natural and conversational
- Avoid yes/no questions; ask for specific information
- Format: one question per line, numbered

Questions:"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )

        text = response.choices[0].message.content
        lines = text.split("\n")

        cleaned = []
        for line in lines:
            # Remove numbering and bullets
            line = line.strip().lstrip("0123456789.-• ").strip()
            if line and len(line) > 10:  # Filter out short noise
                cleaned.append(line)

        print(f"    → Generated {len(cleaned[:num_questions])} questions (API call made)")
        return cleaned[:num_questions]
    
    except Exception as e:
        print(f"    ✗ Error generating questions: {str(e)}")
        return []


# ====================================================================================
# Hallucination checks: queries with no answer in the document
# ====================================================================================
NO_ANSWER_QUESTIONS = [
    "Does this policy cover dental implants?",
    "Is pregnancy covered from day 1?",
    "Can I claim twice in the same day?",
    "Is cosmetic surgery included?",
    "Does this policy include home nursing?",
    "Are pre-existing conditions covered immediately?",
    "Can non-residents file claims?",
]

def build_no_answer_items(pdf_name: str, sample_size: int = None) -> List[Dict]:
    """
    Build evaluation items for questions that should NOT be answered from the document.
    
    Args:
        pdf_name: Name of the PDF document
        sample_size: Limit number of no-answer questions (None = all)
        
    Returns:
        List of evaluation records with no expected answer
    """
    questions = NO_ANSWER_QUESTIONS[:sample_size] if sample_size else NO_ANSWER_QUESTIONS
    
    return [
        {
            "query": q,
            "expected_answer": "I don't know — not found in provided documents.",
            "type": "no_answer",
            "doc_id": pdf_name,
            "ground_truth_span": "",
            "category": "hallucination_check"
        }
        for q in questions
    ]


# ====================================================================================
# Build dataset row: query + context + expected answer
# ====================================================================================
def build_dataset_row(query: str, fact_value: str, pdf_name: str, field: str) -> Dict:
    """
    Construct a single evaluation record.
    
    Args:
        query: User question
        fact_value: Ground truth context
        pdf_name: Source document name
        field: Fact category
        
    Returns:
        Evaluation record dictionary
    """
    return {
        "query": query,
        "expected_answer": fact_value[:300],
        "doc_id": pdf_name,
        "ground_truth_span": fact_value,
        "type": "fact",
        "category": field
    }


# ====================================================================================
# Main: Build dataset with controlled API calls
# ====================================================================================
def build_dataset():
    """
    Main pipeline: extract facts → generate questions → build evaluation dataset.
    
    API Call Reduction Strategy:
    - MAX_PDFS: Process fewer PDFs
    - MAX_FACTS_PER_PDF: Limit facts extracted per document
    - QUESTIONS_PER_FACT: Reduce questions per fact
    - ENABLE_LLM_GENERATION: Disable LLM calls for testing
    """
    print(DatasetConfig.get_summary())
    
    dataset = []
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    
    # Limit PDF files if configured
    if DatasetConfig.MAX_PDFS:
        pdf_files = pdf_files[:DatasetConfig.MAX_PDFS]
    
    print(f"\n📄 Processing {len(pdf_files)} PDF files...\n")

    for pdf_idx, pdf in enumerate(pdf_files, 1):
        print(f"[{pdf_idx}/{len(pdf_files)}] {os.path.basename(pdf)}")
        
        # Extract facts with limit
        facts = extract_facts_from_pdf(pdf, max_facts=DatasetConfig.MAX_FACTS_PER_PDF)

        # Generate questions per fact
        for fact_idx, fact in enumerate(facts, 1):
            fact_value = fact["value"]
            field = fact["field"]
            doc_name = fact["doc_id"]

            # Generate questions (makes API calls)
            questions = generate_questions(
                fact_value, 
                field, 
                num_questions=DatasetConfig.QUESTIONS_PER_FACT
            )

            # Build dataset rows
            for q in questions:
                dataset.append(build_dataset_row(q, fact_value, doc_name, field))

        # Add hallucination check questions
        if DatasetConfig.INCLUDE_NO_ANSWER_CHECKS:
            dataset.extend(
                build_no_answer_items(
                    os.path.basename(pdf),
                    sample_size=DatasetConfig.NO_ANSWER_SAMPLE_SIZE
                )
            )

        print()

    # Write output
    write_jsonl(DatasetConfig.OUTPUT_JSONL, dataset)
    
    print(f"\n✓ Dataset generation complete!")
    print(f"  Total records: {len(dataset)}")
    print(f"  Estimated API calls: {len([d for d in dataset if d['type'] == 'fact']) // DatasetConfig.QUESTIONS_PER_FACT}")


# ====================================================================================
# Run
# ====================================================================================
if __name__ == "__main__":
    build_dataset()
