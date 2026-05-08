# conda activate x1025
# python retrieve.py <lancedb/table_dir> "your query here"
#
# Step 1 — Retrieval:  nvidia/NV-Embed-v2 query embedding + hybrid (cosine + BM25/RRF), top-k candidates
# Step 2 — Rerank:     Qwen/Qwen3-Reranker-8B cross-encoder, top-N from candidates
#
# Importable:
#   from retrieve import retrieve
#   results = retrieve(Path("lancedb/my_table"), "your query")            # defaults: k=100, top_n=15
#   results = retrieve(Path("lancedb/my_table"), "your query", k=200)     # widen candidate pool
#   results = retrieve(Path("lancedb/my_table"), "your query", top_n=5)   # tighter rerank output

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

# MUST be set before importing transformers or lancedb
load_dotenv(Path(__file__).parent / ".env")
os.environ.setdefault("LANCE_LOG", "ERROR")

import lancedb
import torch
import torch.nn.functional as F
from lancedb.rerankers import RRFReranker
from transformers import AutoModelForCausalLM, AutoTokenizer

from ingest import load_embed_model

__all__ = ["retrieve", "open_retriever", "retrieve_with", "switch_table"]

_RERANKER_ID = "Qwen/Qwen3-Reranker-0.6B"
_QUERY_INSTRUCTION = "Instruct: Given a technical question about a technical manual or report, retrieve the most relevant passages that answer the question.\nQuery: "
_RERANK_PROMPT = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    '<|im_start|>user\n'
    '<Instruct>: Given a technical question about a technical manual or report, assess whether the document contains '
    'relevant information to answer the question.\n'
    '<Query>: {query}\n<Document>: {document}<|im_end|>\n'
    '<|im_start|>assistant\n<think>\n\n</think>\n\n'
)

def _open_table(table_path: Path):
    table_name = table_path.name.removesuffix(".lance")
    return lancedb.connect(str(table_path.parent)).open_table(table_name)

def _embed_query(embedder, query: str) -> list:
    with torch.no_grad():
        vec = embedder.encode([query], instruction=_QUERY_INSTRUCTION, max_length=2048)
    return F.normalize(vec.float(), p=2, dim=1).cpu().tolist()[0]

def _search_hybrid(table, vector: list, query: str, k: int) -> list:
    return (table.search(query_type="hybrid")
                 .vector(vector).text(query).metric("cosine")
                 .rerank(RRFReranker(return_score="all"))
                 .limit(k).to_list())

def _load_reranker():
    tokenizer = AutoTokenizer.from_pretrained(_RERANKER_ID, trust_remote_code=True, padding_side="left")
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(_RERANKER_ID, dtype=torch.float16, device_map=device).eval()
    return model, tokenizer, tokenizer.convert_tokens_to_ids("yes"), tokenizer.convert_tokens_to_ids("no")

def _rerank(reranker, query: str, candidates: list, top_n: int, batch_size: int = 16) -> list:
    model, tokenizer, yes_id, no_id = reranker
    prompts = [_RERANK_PROMPT.format(query=query, document=c["text"]) for c in candidates]
    scores = []
    for i in range(0, len(prompts), batch_size):
        inputs = tokenizer(prompts[i:i + batch_size], padding=True, truncation=True, max_length=2048,
                           return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            last_logits = model(**inputs).logits[:, -1]
            scores.extend(torch.softmax(last_logits[:, [no_id, yes_id]], dim=1)[:, 1].cpu().tolist())
    for c, s in zip(candidates, scores):
        c["_rerank_score"] = s
    return sorted(candidates, key=lambda x: x["_rerank_score"], reverse=True)[:top_n]

def _print_results(results: list):
    for i, r in enumerate(results, 1):
        rrf = r.get("_relevance_score") or 0.0
        print(f"\n#{i} Rerank:{r.get('_rerank_score', 0):.3f} | RRF:{rrf:.3f}\n"
              f"Type: {r.get('chunk_type', '')} | Section: {r.get('section', '')}\n"
              f"Text: {r['text'][:400].replace(chr(10), ' ')}...")

def open_retriever(table_path: Path) -> tuple:
    return (load_embed_model(), _load_reranker(), _open_table(table_path))

def retrieve_with(session: tuple, query: str, k: int = 100, top_n: int = 15) -> list:
    embedder, reranker, table = session
    vector = _embed_query(embedder, query)
    candidates = _search_hybrid(table, vector, query, k)
    return _rerank(reranker, query, candidates, top_n)

def switch_table(session: tuple, table_path: Path) -> tuple:
    embedder, reranker, _ = session
    return (embedder, reranker, _open_table(table_path))

def retrieve(table_path: Path, query: str, k: int = 100, top_n: int = 15) -> list:
    results = retrieve_with(open_retriever(table_path), query, k, top_n)
    _print_results(results)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("table_path", type=Path, help="e.g. lancedb/my_table")
    parser.add_argument("query", nargs="+", help="Query string")
    args = parser.parse_args()
    retrieve(args.table_path, " ".join(args.query))
