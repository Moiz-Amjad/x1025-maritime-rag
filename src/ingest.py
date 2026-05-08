# conda activate x1025
# python ingest.py <folder_with_md_and_images>
#
# Reads:  <folder>/manual.md  +  <folder>/image_manifest.json
# Writes: lancedb/<folder>_lancedb  (vector table with FTS index on "text")
#
# Importable:
#   from ingest import ingest
#   ingest(Path("path/to/folder"))

import argparse
import glob
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

# MUST be set before importing transformers or lancedb
load_dotenv(Path(__file__).parent.parent / ".env")
os.environ.setdefault("LANCE_LOG", "ERROR")

import lancedb
import pyarrow as pa
import torch
import torch.nn.functional as F
from transformers import AutoModel

__all__ = ["ingest", "load_embed_model", "patch_nvembed"]

_MODEL_ID = "nvidia/NV-Embed-v2"
_EMBED_DIM = 4096
_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), _EMBED_DIM)),
    pa.field("chunk_type", pa.string()),
    pa.field("section", pa.string()),
    pa.field("image_index", pa.int32()),
    pa.field("image_src", pa.string()),
])

def patch_nvembed():
    for path in glob.glob(os.path.join(os.environ["HF_HOME"], "**/modeling_nvembed.py"), recursive=True):
        with open(path, "r+") as f:
            code = f.read()
            if "position_embeddings = self.rotary_emb" in code and "clone().detach()" in code and "isinstance(layer_outputs" in code:
                continue
            code = code.replace("'input_ids': torch.tensor(batch_dict.get('input_ids').to(batch_dict.get('input_ids')).long()),", "'input_ids': batch_dict.get('input_ids').clone().detach().long(),")
            code = code.replace("        use_cache = use_cache if use_cache is not None else self.config.use_cache", "        use_cache = False")
            code = code.replace("        hidden_states = inputs_embeds\n\n        # decoder layers", "        hidden_states = inputs_embeds\n        position_embeddings = self.rotary_emb(hidden_states, position_ids)\n\n        # decoder layers")
            code = code.replace("                    output_attentions,\n                    use_cache,\n                )\n            else:", "                    output_attentions,\n                    use_cache,\n                    None,\n                    position_embeddings,\n                )\n            else:")
            code = code.replace("                    output_attentions=output_attentions,\n                    use_cache=use_cache,\n                )", "                    output_attentions=output_attentions,\n                    use_cache=use_cache,\n                    position_embeddings=position_embeddings,\n                )")
            code = code.replace("            hidden_states = layer_outputs[0]", "            hidden_states = layer_outputs if isinstance(layer_outputs, torch.Tensor) else layer_outputs[0]")
            f.seek(0); f.write(code); f.truncate()

def load_embed_model():
    patch_nvembed()
    return AutoModel.from_pretrained(_MODEL_ID, trust_remote_code=True).half().cuda().eval()

def _read_source(folder: Path) -> tuple[str, dict]:
    md = (folder / "manual.md").read_text("utf-8")
    manifest = {e["index"]: e for e in json.loads((folder / "image_manifest.json").read_text("utf-8"))}
    return md, manifest

def _parse_image_chunks(md: str, manifest: dict, src_dir: Path) -> list:
    chunks, section = [], "Introduction"
    for block in re.split(r'(<!-- IMAGE_PLACEHOLDER.*?-->)', md, flags=re.S):
        if block.startswith("<!-- IMAGE_PLACEHOLDER"):
            m = re.search(r'^index:\s*(\d+)', block, re.M)
            if not m: continue
            idx = int(m.group(1))
            entry = manifest.get(idx, {})
            desc, src = entry.get("description", "").strip(), entry.get("source", "")
            img = src_dir / src
            if desc and not desc.startswith("[") and img.exists() and img.stat().st_size >= 1000:
                chunks.append({"id": f"img_{idx:04d}", "text": desc, "chunk_type": "image",
                               "section": section, "image_index": idx, "image_src": src})
            continue
        for line in block.splitlines():
            m = re.match(r'^#{1,6}\s+(.+)$', line.strip())
            if m: section = m.group(1)
    return chunks

def _parse_text_chunks(md: str) -> list:
    text = re.sub(r'<!-- IMAGE_PLACEHOLDER.*?-->', '', md, flags=re.S)
    chunks, section, buf, buf_w, txt_id = [], "Introduction", [], 0, 0

    def flush():
        nonlocal txt_id
        body = "\n".join(buf).strip()
        if body:
            chunks.append({"id": f"txt_{txt_id:04d}", "text": f"Section: {section}\n{body}",
                           "chunk_type": "text", "section": section, "image_index": -1, "image_src": ""})
            txt_id += 1

    for line in text.splitlines():
        line = line.strip()
        if not line: continue
        m = re.match(r'^#{1,6}\s+(.+)$', line)
        if m:
            flush()
            buf, buf_w = [], 0
            section = m.group(1)
            continue
        words = len(line.split())
        if buf_w + words > 1000 and buf:
            flush()
            buf = buf[-5:]
            buf_w = sum(len(x.split()) for x in buf)
        buf.append(line)
        buf_w += words
    flush()
    return chunks

def _embed_chunks(model, chunks: list) -> list:
    with torch.no_grad():
        vecs = model._do_encode([c["text"] for c in chunks], batch_size=8, instruction="", max_length=2048, num_workers=2)
    for c, v in zip(chunks, F.normalize(vecs.float(), p=2, dim=1).cpu().tolist()):
        c["vector"] = v
    return chunks

def _store_chunks(chunks: list, db_path: Path, table_name: str):
    db_path.mkdir(parents=True, exist_ok=True)
    table = lancedb.connect(str(db_path)).create_table(table_name, data=chunks, schema=_SCHEMA, mode="overwrite")
    table.create_fts_index("text", replace=True, use_tantivy=False)
    print(f"Indexed {len(chunks)} chunks to {db_path.name}/{table_name}.")

def ingest(folder: Path):
    folder = folder.resolve()
    md, manifest = _read_source(folder)
    chunks = _parse_image_chunks(md, manifest, folder) + _parse_text_chunks(md)
    chunks = _embed_chunks(load_embed_model(), chunks)
    _store_chunks(chunks, folder.parent / "lancedb", f"{folder.name}_lancedb")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    ingest(parser.parse_args().folder)
