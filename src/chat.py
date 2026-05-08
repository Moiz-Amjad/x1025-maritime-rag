# conda activate x1025
# python chat.py
#
# Interactive Q&A loop. Loads NV-Embed + Reranker (parent) and Qwen3.6-35B-A3B
# Q6_K (child subprocess, single MIG slice) ONCE, then runs an input loop with
# warm models. Same backend as answer.py — only the lifecycle differs.
#
# Commands inside the chat:
#   switch  — pick a different manual (models stay loaded)
#   quit    — exit (also: q, exit, Ctrl+D, Ctrl+C)

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
os.environ.setdefault("LANCE_LOG", "ERROR")

import sys
from retrieve import open_retriever, retrieve_with, switch_table
from answer import open_llm, generate_with, close_llm

__all__ = ["chat"]

_DB_DIR = Path("data/lancedb")

def _list_manuals() -> list:
    if not _DB_DIR.is_dir():
        sys.exit(f"Error: {_DB_DIR} not found.")
    manuals = sorted(t.name[:-6] for t in _DB_DIR.iterdir() if t.is_dir() and t.name.endswith(".lance"))
    if not manuals:
        sys.exit(f"Error: No tables in {_DB_DIR}.")
    return manuals

def _pick_manual(manuals: list) -> str:
    print("\nAvailable manuals:")
    for i, name in enumerate(manuals, 1):
        print(f"  {i}. {name}")
    while True:
        try:
            choice = input(f"\nPick a manual (1-{len(manuals)}, or q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if choice.lower() in {"q", "quit", "exit"}:
            sys.exit(0)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(manuals):
                return manuals[idx]
        except ValueError:
            pass
        print(f"Invalid. Pick a number between 1 and {len(manuals)}.")

def chat():
    manuals = _list_manuals()
    table_name = _pick_manual(manuals)

    print(f"\nLoading models for '{table_name}' (~3 min on first run)...")
    retriever = open_retriever(_DB_DIR / f"{table_name}.lance")
    llm = open_llm()
    print("\nReady. Type a question. Commands: 'switch' / 'quit'.")

    try:
        while True:
            try:
                query = input(f"\n[{table_name}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                continue
            if query.lower() in {"q", "quit", "exit"}:
                break
            if query.lower() == "switch":
                table_name = _pick_manual(manuals)
                retriever = switch_table(retriever, _DB_DIR / f"{table_name}.lance")
                continue

            print("  Retrieving + reranking...", flush=True)
            chunks = retrieve_with(retriever, query, 50, 5)
            print("  Generating...\n", flush=True)
            print(generate_with(llm, query, chunks))
    finally:
        close_llm(llm)
        print("\nGoodbye.")

if __name__ == "__main__":
    chat()
