# conda activate x1025
# python answer.py <lancedb/table_dir> "your question here"
#
# Self-contained one-shot test script for validating end-to-end RAG output.
# (For repeated/interactive queries with a warm LLM, use chat.py instead.)
#
# Pipeline:
#   1. retrieve()  → NV-Embed hybrid search + Qwen3-Reranker → top chunks
#                    (runs in this process; uses cuda:0 + cuda:1)
#   2. _generate() → spawns a child process with CUDA_VISIBLE_DEVICES pinned to
#                    one MIG slice; child loads Qwen3.6-35B-A3B Q6_K via
#                    llama-cpp-python and generates. Subprocess isolation is
#                    required because llama.cpp dedupes devices by PCI BDF — all
#                    MIG slices share one BDF, so single-slice pinning only works
#                    when the slice is the only one visible to the process.
#
# Slice layout (each H200 MIG slice ~34.9 GB):
#   cuda:0  — NV-Embed-v2           (~15.7 GB) [parent]
#   cuda:1  — Qwen3-Reranker-8B     (~16.4 GB) [parent]
#   slice 2 — Qwen3.6-35B-A3B Q6_K  (~28-30 GB) [child, single-slice]
#
# First run downloads the Q6_K GGUF (~29 GB) to HF_HOME (~5 min). Each subsequent
# run reloads the model in the child (~2 min) — by design for a test script.
#
# Requirements:
#   - 3+ MIG slices visible (--gres=gpu:3 minimum on Slurm)
#   - llama-cpp-python built with CUDA support:
#       CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir \
#           --force-reinstall --no-binary=llama-cpp-python
#
# Override which MIG slice the LLM uses by setting LLM_MIG_UUID
# (defaults to slice index 2 as listed by `nvidia-smi -L`).
#
# Importable:
#   from answer import answer
#   response = answer(Path("lancedb/my_table"), "your question")            # defaults: k=100, top_n=15
#   response = answer(Path("lancedb/my_table"), "your question", k=200)     # widen candidate pool
#   response = answer(Path("lancedb/my_table"), "your question", top_n=5)   # fewer chunks to LLM

import argparse
import multiprocessing as mp
import os
import re
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
os.environ.setdefault("LANCE_LOG", "ERROR")

from retrieve import retrieve

__all__ = ["answer", "open_llm", "generate_with", "close_llm"]

_LLM_REPO = "unsloth/Qwen3.6-35B-A3B-GGUF"
_LLM_FILE = "Qwen3.6-35B-A3B-UD-Q6_K.gguf"
_LLM_CTX = 8192
_LLM_MAX_NEW_TOKENS = 1024
_LLM_SLICE_INDEX = 2
_SYSTEM_PROMPT = (
    "You are a highly precise technical assistant.\n\n"
    "Rules:\n"
    "1. Use ONLY information explicitly stated in the provided context. Do not draw on outside knowledge.\n"
    "2. Report exact values, labels, tag numbers, and steps exactly as they appear.\n"
    "3. Never infer, extrapolate, or extend beyond what is written.\n"
    "4. If the context is insufficient to answer fully, state what is and is not available.\n"
    "5. Organize your final response in a clean, highly readable manner using ONLY plain text formatting (newlines and indentation). ABSOLUTELY DO NOT use Markdown formatting such as **, *, or #."
)
# Trailing <think>\n\n</think>\n\n disables Qwen3 thinking mode (same effect as enable_thinking=False)
_PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

def _llm_mig_uuid() -> str:
    if uuid := os.environ.get("LLM_MIG_UUID"):
        return uuid
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
    uuids = re.findall(r"UUID: (MIG-[a-f0-9-]+)", out)
    if len(uuids) <= _LLM_SLICE_INDEX:
        raise RuntimeError(f"Need at least {_LLM_SLICE_INDEX + 1} MIG slices, found {len(uuids)}")
    return uuids[_LLM_SLICE_INDEX]

def _llm_worker(conn):
    from llama_cpp import Llama
    llm = Llama.from_pretrained(
        repo_id=_LLM_REPO, filename=_LLM_FILE,
        n_gpu_layers=-1, n_ctx=_LLM_CTX, verbose=False,
    )
    while True:
        try:
            prompt = conn.recv()
        except EOFError:
            break
        if prompt is None:
            break
        output = llm(
            prompt,
            max_tokens=_LLM_MAX_NEW_TOKENS,
            temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5,
            stop=["<|im_end|>"],
        )
        conn.send(output["choices"][0]["text"])

def _build_prompt(query: str, chunks: list) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        header = f"[{i}] Section: {c['section']}"
        if c["chunk_type"] == "image":
            header += f"\n[Figure: {c['image_src']}] Description:"
        parts.append(f"{header}\n{c['text']}")
    context = "\n\n---\n\n".join(parts)
    user_content = (
        "Use the following passages from the technical document or report to answer the question.\n\n"
        f"{context}\n\nQuestion: {query}"
    )
    return _PROMPT_TEMPLATE.format(system=_SYSTEM_PROMPT, user=user_content)

def open_llm() -> tuple:
    parent, child = mp.Pipe()
    saved = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = _llm_mig_uuid()
    try:
        p = mp.get_context("spawn").Process(target=_llm_worker, args=(child,))
        p.start()
    finally:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved
    return (p, parent)

def generate_with(session: tuple, query: str, chunks: list) -> str:
    p, conn = session
    conn.send(_build_prompt(query, chunks))
    try:
        text = conn.recv().strip()
    except EOFError:
        raise RuntimeError(f"LLM subprocess crashed (exit code {p.exitcode})")
    if "<think>" in text:
        end = text.find("</think>")
        if end != -1:
            text = text[end + len("</think>"):].strip()
    return text

def close_llm(session: tuple):
    p, conn = session
    try:
        conn.send(None)
    except (BrokenPipeError, OSError):
        pass
    p.join(timeout=5)
    if p.is_alive():
        p.terminate()

def _print_answer(query: str, response: str):
    print(f"\nQuestion: {query}\n")
    print("=" * 60)
    print(response)
    print("=" * 60)

def answer(table_path: Path, query: str, k: int = 100, top_n: int = 15) -> str:
    chunks = retrieve(table_path, query, k=k, top_n=top_n)
    session = open_llm()
    try:
        response = generate_with(session, query, chunks)
    finally:
        close_llm(session)
    _print_answer(query, response)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("table_path", type=Path, help="e.g. lancedb/my_table")
    parser.add_argument("query", nargs="+", help="Query string")
    args = parser.parse_args()
    answer(args.table_path, " ".join(args.query))
