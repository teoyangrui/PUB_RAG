# Ensure modern SQLite (>=3.35) on Streamlit Cloud
try:
    import pysqlite3  # provides a newer SQLite
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
import os, io, shutil, tempfile
from typing import List, Dict, Tuple

import streamlit as st
from . import LLM as llm 
from pypdf import PdfReader
import docx2txt
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer 

#create temparory vectordb from uploaded document
#read uploaded pdf
#Return (text, metadata) for the whole file 
def read_pdf_pages_from_bytes(data: bytes, name: str) -> List[Tuple[str, Dict]]:
    reader = PdfReader(io.BytesIO(data))
    docs: List[Tuple[str, Dict]] = [] #to upload page by page with metadata
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            docs.append((text, {"source": name, "page": i + 1, "type": "pdf"}))
    return docs
#read uploaded docx
#Return (text, metadata) for the whole file 
def read_docx_whole_from_bytes(data: bytes, name: str) -> List[Tuple[str, Dict]]:
    # docx2txt requires a file path; use a secure temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"__{name}") as tf:
        tf.write(data)
        temp_path = tf.name
    try:
        text = docx2txt.process(temp_path) or ""
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass
    return [(text, {"source": name, "type": "docx"})] if text.strip() else []
#read txt file
#Return (text, metadata) for the whole file 
def read_txt_whole_from_bytes(data: bytes, name: str) -> List[Tuple[str, Dict]]:
    try:
        text = data.decode("utf-8")
    except Exception:
        text = data.decode("latin-1", errors="ignore")
    return [(text, {"source": name, "type": "text"})] if text.strip() else []

#function to chunk text
def chunk_text(text: str, chunk_size_words: int = 220, overlap_words: int = 40) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks, step = [], max(1, chunk_size_words - overlap_words)
    for start in range(0, len(words), step):
        end = start + chunk_size_words
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks

# bringing the above functions together
# Turn uploaded files into segments: [{"id": str, "text": str, "metadata": {"source":..., "page":...}}, ...]
def build_segments_from_uploads(files) -> List[Dict]:
    #if multiple files are uploaded
    segments: List[Dict] = []
    for f in files:
        name = getattr(f, "name", "uploaded")
        data = f.getvalue()  # read bytes once
        ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
        if ext == "pdf":
            pairs = read_pdf_pages_from_bytes(data, name)
        elif ext in ("docx",):
            pairs = read_docx_whole_from_bytes(data, name)
        elif ext in ("txt", "md"):
            pairs = read_txt_whole_from_bytes(data, name)
        else:
            pairs = read_txt_whole_from_bytes(data, name)

        # chunk + make unique IDs
        for (page_text, meta) in pairs:
            is_pdf = meta.get("type") == "pdf"
            base_page = meta.get("page", 1)
            chunks = chunk_text(page_text, 220, 40)
            for ci, c in enumerate(chunks):
                page = base_page if is_pdf else (ci + 1)
                seg_id = f"{name}::p{page}::c{ci}"
                segments.append({"id": seg_id, "text": c, "metadata": {"source": name, "page": page}})
    return segments

def get_temp_chroma():
    if "tmp_chroma_dir" not in st.session_state:
        st.session_state.tmp_chroma_dir = tempfile.mkdtemp(prefix="chroma_session_")
        st.session_state.tmp_chroma_client = chromadb.PersistentClient(
            path=st.session_state.tmp_chroma_dir,
            settings=Settings(allow_reset=True)
        )
        st.session_state.tmp_chroma_coll = st.session_state.tmp_chroma_client.get_or_create_collection(
            "session_docs"
        )
    return (st.session_state.tmp_chroma_client,
            st.session_state.tmp_chroma_coll,
            st.session_state.tmp_chroma_dir)
#Deletes the session’s temp chroma directory
def clear_temp_chroma():
    tmp = st.session_state.pop("tmp_chroma_dir", None)
    st.session_state.pop("tmp_chroma_client", None)
    st.session_state.pop("tmp_chroma_coll", None)
    if tmp and os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)

def get_embedder():
    model = st.session_state.get("_embedder")
    if model is None:
        #use minilm instead of GPT for temp documents to save cost
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        st.session_state._embedder = model
    return model

#embed the text 
def embed_chunks(texts: List[str]) -> List[List[float]]:
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True).tolist()

def chroma_add_segments(segments: List[Dict]):
    """
    segments: list of dicts like {"id": str, "text": str, "metadata": {...}}
    """
    _, coll, _ = get_temp_chroma()
    texts = [s["text"] for s in segments]
    ids = [s["id"] for s in segments]
    metas = [s.get("metadata", {}) for s in segments]
    embs = embed_chunks(texts)
    coll.add(documents=texts, embeddings=embs, metadatas=metas, ids=ids)

def chroma_query(query: str, top_k: int = 8) -> List[Dict]:
    _, coll, _ = get_temp_chroma()
    # Supply query_embeddings (not query_texts) since we didn't attach an embedding fn to the collection
    
    q_emb = get_embedder().encode([query], normalize_embeddings=True).tolist()
    results = coll.query(query_embeddings=q_emb, n_results=top_k)
    outs: List[Dict] = []
    if results and results.get("documents"):
        for i, doc in enumerate(results["documents"][0]):
            outs.append({
                "text": doc,
                "metadata": (results["metadatas"][0][i] if results.get("metadatas") else {}),
                "distance": (results["distances"][0][i] if results.get("distances") else None),
            })
    return outs

#function for llm to answer questions from uploaded document
#wraps around ask function from llm.py with temp_context=True where persistent vector store will not be
#used
def ask_with_temp_context(question: str, excerpts: List[Dict], strict: bool = True) -> str:
    context_lines = []
    for e in excerpts:
        src = e.get("metadata", {}).get("source", "uploaded")
        page = e.get("metadata", {}).get("page", "?")
        context_lines.append(f"[{src} p.{page}]\n{e['text']}\n")
    context = "\n".join(context_lines)
#optinality to only answer from document or not to
    guard = ("If the answer is not present in the context, reply exactly: "
             "'Not found in the uploaded documents.'") if strict else \
            "Prefer answers grounded in the context. If something isn't covered, you may add concise general knowledge (flag clearly)."

    prompt = (
        f'''
        Use the context below to answer the question.
        {guard}
        Context excerpts:
        {context}
        User question: {question}
        Answer concisely with brief citations like [source p.page] where applicable.
        '''
    )
    # Call into LLM.ask, indicating temp_context=True
    return llm.ask(prompt, temp_context=True) 