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
```bash
# init backend llama3.1
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --port 8000
```

"""

__author__ = 'hw56@iu.edu'
__version__ = '0.0.1'
__license__ = 'MIT'


import gradio as gr
import fitz  # pymupdf for robust pdf parsing
from contextgem import Document, DocumentLLM, SegmentAndTagSegmenter, JsonObjectConcept


def parse_pdf(pdf_file):
    """extract raw text from pdf using pymupdf (more reliable for scientific pdfs)."""
    doc = fitz.open(pdf_file)
    text = "\n".join(page.get_text() for page in doc)
    return text


# define extraction concept: nia-aa abc staging
abc_concept = JsonObjectConcept(
    name="ABC_Staging",
    description="Extract NIA‑AA ABC score: A (Aβ plaques), B (Braak stage), C (CERAD) and overall likelihood.",
    json_schema={
        "type": "object",
        "properties": {
            "A": {"type": "integer", "minimum": 0, "maximum": 3},
            "B": {"type": "integer", "minimum": 0, "maximum": 3},
            "C": {"type": "integer", "minimum": 0, "maximum": 3},
            "likelihood": {"type": "string"}
        },
        "required": ["A", "B", "C", "likelihood"]
    },
    add_references=True,
    reference_depth="sentences",
    add_justifications=True,
    justification_depth="brief",
)

# define extraction concept: anatomical entities
anat_concept = JsonObjectConcept(
    name="Anatomical_Entities",
    description="List anatomical structures with FMA ID and description.",
    json_schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "term": {"type": "string"},
                "fma_id": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["term", "fma_id", "description"],
        },
    },
    add_references=True,
    reference_depth="sentences",
    add_justifications=True,
    justification_depth="brief",
)

# define extraction concept: anatomical asymmetries
asymmetry_concept = JsonObjectConcept(
    name="Anatomical_Asymmetries",
    description="Extract all mentions of anatomical asymmetries (left vs right, hemisphere differences, side-specific findings).",
    json_schema={
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "structure": {"type": "string"},
                "left": {"type": "string"},
                "right": {"type": "string"},
                "comment": {"type": "string"}
            },
            "required": ["structure", "left", "right"],
        },
    },
    add_references=True,
    reference_depth="sentences",
    add_justifications=True,
    justification_depth="brief",
)


def extract_concepts(pdf_file, show_prompt=False):
    """full pipeline: parse pdf → segment → extract → format results."""

    # parse pdf content
    text = parse_pdf(pdf_file)

    # initialize contextgem document and apply neural segmentation
    doc = Document(raw_text=text)
    segmenter = SegmentAndTagSegmenter(model_name="sat-base")
    doc = segmenter.segment(doc)

    # assign extraction concepts
    doc.concepts = [abc_concept, anat_concept, asymmetry_concept]

    # connect contextgem to local vllm (openai-compatible api)
    llm = DocumentLLM(
        model="llm-dummy",  # label only (ContextGem never uses this value to control routing)
        api_key="sk-dummy",  # dummy (required by ContextGem's OpenAI interface wrapper)
        api_base="http://localhost:8000/v1",  # local vLLM API endpoint
        model_type="openai"  # use OpenAI-compatible chat completion format for llama3.1
    )

    # run extraction
    doc = llm.extract_all(doc)

    # format results for output
    output = ""

    # abc staging results
    abc_items = doc.get_concept_by_name("ABC_Staging").extracted_items
    if abc_items:
        abc = abc_items[0]
        val = abc.value
        output += f"ABC Staging:\nA: {val['A']}, B: {val['B']}, C: {val['C']}, Likelihood: {val['likelihood']}\n"
        output += f"Justification: {abc.justification}\n"
        for ref in abc.references:
            output += f"Source: {ref.text}\n"
        output += "\n"

    # anatomical entities results
    anat_items = doc.get_concept_by_name("Anatomical_Entities").extracted_items
    output += "Anatomical Entities:\n"
    for item in anat_items:
        val = item.value
        output += f"- {val['term']} (FMA: {val['fma_id']}): {val['description']}\n"
        output += f"  Justification: {item.justification}\n"
        for ref in item.references:
            output += f"  Source: {ref.text}\n"
    output += "\n"

    # anatomical asymmetries results
    asymmetry_items = doc.get_concept_by_name("Anatomical_Asymmetries").extracted_items
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

    # optionally display full llm prompts used internally
    if show_prompt:
        internal = doc._internal_metadata.get("last_chat_prompts", [])
        if internal:
            full_prompts = "\n".join(f"[{p.role}] {p.content}" for p in internal)
            output += "\n\nLLM Prompts:\n" + full_prompts
        else:
            output += "\n\n[No prompt metadata available]"

    return output


# build gradio interface
gr.Interface(
    fn=extract_concepts,
    inputs=[
        gr.File(label="upload neuropathology pdf"),
        gr.Checkbox(label="show llm prompt chain", value=False)
    ],
    outputs=gr.Textbox(label="extraction result", lines=40),
    title="Demo: Neuropathology Report Concept Extraction",
    description=(
        "upload a neuropathology report pdf. "
        "extract nia-aa abc staging, anatomical structures, and anatomical asymmetries "
        "using contextgem with llama 3.1-8b instruct served locally via vllm."
    )
).launch()
