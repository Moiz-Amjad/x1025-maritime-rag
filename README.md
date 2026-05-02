# x1025-maritime-rag — Autonomous Maritime Intelligence
**Layer 1 Prototype: Industrial-Grade RAG for Complex Engineering Documents**

Welcome to the foundational AI infrastructure for **x1025-maritime-rag**. This repository contains a production-grade Retrieval-Augmented Generation (RAG) pipeline built to ingest, search, and reason over highly technical maritime engineering manuals, such as the *N.S. SAVANNAH Safety Analysis Report*.

Standard "out-of-the-box" RAG systems fail spectacularly on dense, multi-page maritime documents. This prototype was explicitly engineered to solve those failures, providing 100% grounded answers to complex procedural and tabular questions without hallucinating—a critical safety requirement under the ISM Code.

## 🚀 Key Architectural Innovations

* **Macro-Chunking:** Instead of naive line-splitting, we group text under shared Markdown headers up to a strict `1000` word limit. This ensures large tables are ingested as single, cohesive units, preventing the fragmentation of rows from their column headers.
* **Vision-Language Processing (VLM):** We bypass faulty OCR and extract pristine native text via Docling. For diagrams, we use quantized **InternVL2.5-38B-AWQ** to dynamically translate imagery into highly accurate text descriptions.
* **Two-Stage Hybrid Retrieval:**
  * **Stage 1 (Recall):** `LanceDB` + `NV-Embed-v2` performs a lightning-fast vector search to fetch 100 candidates.
  * **Stage 2 (Precision):** A massive `Qwen2.5-32B` Cross-Encoder reranker scores and extracts the exact top 15 most relevant chunks.
* **100% Self-Hosted Security:** Designed specifically to run on local **NVIDIA H200 MIG slices**. No sensitive fleet data or proprietary company manuals are ever sent to third-party APIs like OpenAI.

## 💻 Hardware Requirements

This is an enterprise-grade pipeline designed for heavy GPU computation. 
* **Minimum Requirement:** 1x NVIDIA H200 MIG slice (approx. `1g.35gb`) or equivalent GPU with at least ~35GB of VRAM.
* **Why?** Running massive embedding models (`NV-Embed-v2` at ~15.6 GB) and 32-Billion parameter rerankers/vision models locally requires significant memory. 

## ⚙️ Installation & Setup

We have provided Conda environments capturing the exact working state. 

```bash
# Clone the repository
git clone https://github.com/Moiz-Amjad/x1025-maritime-rag.git
cd x1025-maritime-rag

# Create the Conda environment from the exported file
conda env create -f environment.yml

# Activate the environment
conda activate x1025
```
*(Note: If you are building on a different architecture, you can use `requirements.txt` to install dependencies without OS-specific hashes).*

## 🚀 Quickstart Guide

The pipeline executes in two distinct phases: **Ingestion** (run once per document) and **Retrieval** (interactive). Note: Make sure to copy `.env.example` to `.env` and configure your `HF_HOME` and `HF_TOKEN` variables so that huggingface models download to your designated persistent cache.

### Phase 1: Data Ingestion & Indexing
Place your raw PDF in the `data/` directory, then run the extraction scripts:

1. **PDF to Markdown:** Converts raw PDFs into structured Markdown using Docling, saving complex figures as PNGs.
   ```bash
   python src/convert_to_markdown.py "data/N.S._SAVANNAH_UPDATED_FINAL_SAFETY_ANALYSIS_REPORT.pdf" --output-dir data/
   ```
2. **Vision Extraction:** Uses InternVL2.5-38B-AWQ to analyze extracted images and writes descriptions into a manifest.
   ```bash
   python src/describe_images_lmdeploy.py data/
   ```
3. **Markdown Injection:** Patches the generated image descriptions back into the master `manual.md` document.
   ```bash
   python src/update_md_from_manifest.py data/
   ```
4. **Embedding & Indexing:** Runs the Macro-Chunking algorithm, embeds the text using `NV-Embed-v2`, and writes the vectors to `LanceDB`.
   ```bash
   python src/chunk_and_embed.py data/
   ```

### Phase 2: Querying & Chat
The interactive terminal application embeds your query, fetches candidates from LanceDB, reranks them via Qwen, and streams a factually grounded answer.
```bash
python src/chat.py
```

## 🎯 Performance Demonstration

The pipeline has been extensively tested against the *N.S. SAVANNAH Safety Analysis Report*. By combining Macro-Chunking with the 32B Cross-Encoder reranker, the system successfully extracts and synthesizes correct answers from highly complex, tabular engineering data where standard RAG systems fail.

![Pipeline extracting complex tabular metrics](assets/performance_1.png)
![Pipeline querying component symbols](assets/performance_2.png)

## 🛠 Engineering "War Stories"
Building this pipeline involved solving several critical limitations of modern LLMs:
* **Token Truncation:** We discovered embedding models natively amputated the bottom 200 tokens of 1500-word chunks. We mathematically fixed this by reducing chunk thresholds to 1000 words.
* **GPU OOM Crashes:** We hardcoded batch processing sizes (e.g., `batch_size=1` for reranking) to permanently stabilize VRAM usage, preventing out-of-memory crashes during deep multi-chunk reranking.
* **Ground Truth Discrepancies:** When the LLM supposedly "failed" test queries, we wrote programmatic PDF-extraction scripts that proved the LLM was actually 100% correct—the expected test answers were factually missing from the original source documents!

---
*Developed for the IMPACT Program — UMass Boston Venture Development Center*
