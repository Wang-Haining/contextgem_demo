"""
Demo: Neuropathology Report Extraction Pipeline using ContextGem

This module demonstrates how to apply ContextGem to extract structured information
from clinical neuropathology reports in PDF format.

Three extraction targets are defined:
1. NIA-AA ABC Staging (Alzheimer's disease neuropathology framework).
2. Anatomical entities using Foundational Model of Anatomy (FMA).
3. Anatomical asymmetries across hemispheres or sides.

LLM backend: LLaMA 3.1 8B Instruct served via vLLM (OpenAI-compatible API).
This pipeline supports local fully-private deployment.

Usage:
# activate venv
source .venv/bin/activate

# start vllm backend
OMP_NUM_THREADS=1 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --max-model-len 16384

# start gradio interface
python app.py

# ssh tunnel for browser access
ssh -J hw56@quartz.uits.iu.edu hw56@g13.quartz.uits.iu.edu -L 7860:localhost:7860
"""

__author__ = "hw56@iu.edu"
__version__ = "0.0.2"
__license__ = "MIT"


import json
import os

import fitz
import gradio as gr
from contextgem import Document, DocumentLLM, JsonObjectConcept


def parse_pdf(pdf_file):
    """extract raw text from pdf using pymupdf (more reliable for scientific pdfs)."""
    doc = fitz.open(pdf_file)
    text = "\n".join(page.get_text() for page in doc)
    return text


# define extraction concept: nia-aa abc staging
staging_concept = JsonObjectConcept(
    name="ABC_Staging",
    description="Extract NIA‑AA ABC score: A (Aβ plaques), B (Braak stage), C (CERAD) and overall likelihood.",
    structure={"A": int, "B": int, "C": int, "likelihood": str},
    add_references=True,
    reference_depth="sentences",
    add_justifications=True,
    justification_depth="brief",
)

anat_concept = JsonObjectConcept(
    name="Anatomical_Entities",
    description="List anatomical structures with FMA ID and description.",
    structure={"term": str, "fma_id": str, "description": str},
    add_references=True,
    reference_depth="sentences",
    add_justifications=True,
    justification_depth="brief",
)

asymmetry_concept = JsonObjectConcept(
    name="Anatomical_Asymmetries",
    description="Extract anatomical asymmetries.",
    structure={"structure": str, "left": str, "right": str, "comment": str},
    add_references=True,
    reference_depth="sentences",
    add_justifications=True,
    justification_depth="brief",
)


# initialize fallback llm (OpenAI GPT-4o-mini)
openai_api_key = os.getenv("OPENAI_API_KEY")

fallback_llm = DocumentLLM(
    model="openai/gpt-4o-2024-05-13",
    api_key=openai_api_key,
)

# initialize primary llm (local vLLM llama3.1)
primary_llm = DocumentLLM(
    model="vllm/meta-llama/Llama-3.1-8B-Instruct",
    api_base="http://localhost:8000/v1",
    fallback_llm=fallback_llm,
)


def extract_concepts(pdf_file, extraction_target, show_prompt):
    text = parse_pdf(pdf_file)
    doc = Document(raw_text=text)

    concept_map = {
        "ABC Staging": [staging_concept],
        "Anatomical Entities": [anat_concept],
        "Anatomical Asymmetries": [asymmetry_concept],
        "All": [staging_concept, anat_concept, asymmetry_concept],
    }
    doc.concepts = concept_map[extraction_target]

    doc = primary_llm.extract_all(doc)

    # OPTIONAL: print what ContextGem actually sent to LLM
    prompt_debug_output = ""
    if show_prompt:
        chat_request = doc._internal_metadata.get("last_chat_request")
        if chat_request:
            prompt_debug_output = json.dumps(chat_request.to_dict(), indent=2)

    # parse results
    output = ""

    if extraction_target in ["ABC Staging", "All"]:
        abc_items = doc.get_concept_by_name("ABC_Staging").extracted_items
        if abc_items:
            abc = abc_items[0]
            val = abc.value
            output += f"ABC Staging:\nA: {val['A']}, B: {val['B']}, C: {val['C']}, Likelihood: {val['likelihood']}\n"
            output += f"Justification: {abc.justification}\n"
            for ref in abc.references:
                output += f"Source: {ref.text}\n"
            output += "\n"

    if extraction_target in ["Anatomical Entities", "All"]:
        anat_items = doc.get_concept_by_name("Anatomical_Entities").extracted_items
        output += "Anatomical Entities:\n"
        for item in anat_items:
            val = item.value
            output += f"- {val['term']} (FMA: {val['fma_id']}): {val['description']}\n"
            output += f"  Justification: {item.justification}\n"
            for ref in item.references:
                output += f"  Source: {ref.text}\n"
        output += "\n"

    if extraction_target in ["Anatomical Asymmetries", "All"]:
        asymmetry_items = doc.get_concept_by_name(
            "Anatomical_Asymmetries"
        ).extracted_items
        output += "Anatomical Asymmetries:\n"
        for item in asymmetry_items:
            val = item.value
            output += (
                f"- Structure: {val['structure']}\n"
                f"  Left: {val['left']}\n"
                f"  Right: {val['right']}\n"
                f"  Comment: {val.get('comment','')}\n"
                f"  Justification: {item.justification}\n"
            )
            for ref in item.references:
                output += f"  Source: {ref.text}\n"
        output += "\n"

    # append prompt debugging output if requested
    if show_prompt:
        output += "\n" + "=" * 30 + "\nLLM Prompt Sent:\n" + prompt_debug_output

    return output


# build gradio interface
gr.Interface(
    fn=extract_concepts,
    inputs=[
        gr.File(label="upload neuropathology pdf"),
        gr.Dropdown(
            choices=[
                "ABC Staging",
                "Anatomical Entities",
                "Anatomical Asymmetries",
                "All",
            ],
            label="Extraction Target",
            value="ABC Staging",
        ),
        gr.Checkbox(label="show full llm prompt", value=False),
    ],
    outputs=gr.Textbox(label="extraction result", lines=40),
    title="Demo: Neuropathology Report Concept Extraction",
    description="Upload PDF. Extract concepts using ContextGem with llama3.1 local + GPT-4o backup.",
).launch()
