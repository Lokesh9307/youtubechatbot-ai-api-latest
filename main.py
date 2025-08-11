# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
import uuid
import math
from typing import List, Dict, Tuple
from fastapi.middleware.cors import CORSMiddleware
from youtube_utils import get_youtube_video_id, get_youtube_transcript

# Vector/embedding libs
from fastembed import TextEmbedding
import faiss
import numpy as np

import dotenv
dotenv.load_dotenv()
from utils.formatResponse import format_response_as_points

# App + keys
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# Embedding model (local)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = TextEmbedding(EMBED_MODEL_NAME)

# In-memory session store:
# sessions: { session_id: { "index": faiss.IndexFlatIP, "id2chunk": {int: str}, "meta": {...}, "history": [(q,a), ...] } }
sessions: Dict[str, Dict] = {}

# PARAMETERS
CHUNK_SIZE = 2000      # characters per chunk (approx)
CHUNK_OVERLAP = 500   # overlap chars
TOP_K = 50             # number of chunks to retrieve


class VideoRequest(BaseModel):
    url: str


class ChatRequest(BaseModel):
    session_id: str
    query: str


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Chunk text into overlapping pieces roughly chunk_size long (by characters).
    """
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= text_len:
            break
        start = end - overlap
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Return L2-normalized embeddings (rows = vectors).
    We'll normalize to use inner-product for cosine similarity in FAISS.
    """
    embs_generator = embedder.embed(texts)  # Returns a generator
    embs = np.array(list(embs_generator))   # Convert to NumPy array
    
    # normalize
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embs = embs / norms
    return embs.astype('float32')


@app.post("/load_video")
def load_video(request: VideoRequest):
    vid = get_youtube_video_id(request.url)
    if not vid:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    transcript = get_youtube_transcript(vid)
    if not transcript:
        # return structured JSON so frontend can handle gracefully
        return {
            "status": "transcript_unavailable",
            "reason": "No captions and Whisper fallback failed or file too large/unsupported",
            "video_id": vid
        }

    # chunk
    chunks = chunk_text(transcript)
    if not chunks:
        raise HTTPException(status_code=500, detail="Failed to chunk transcript")

    # embed
    embs = embed_texts(chunks)

    # create FAISS index (inner product on normalized vectors => cosine)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    # map internal ids
    id2chunk = {i: chunks[i] for i in range(len(chunks))}

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "index": index,
        "id2chunk": id2chunk,
        "history": [],     # (q,a) tuples
        "meta": {
            "video_id": vid,
            "n_chunks": len(chunks)
        }
    }
    print(f"[Session Created] {session_id} video={vid} chunks={len(chunks)}")
    return {"session_id": session_id, "n_chunks": len(chunks)}


@app.post("/chat")
def chat(request: ChatRequest):
    session = sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    # embed query
    q_emb = embed_texts([query])  # shape (1,dim)
    D, I = session["index"].search(q_emb, TOP_K)
    retrieved_chunks = []
    for idx in I[0]:
        if idx < 0:
            continue
        chunk_text = session["id2chunk"].get(int(idx))
        if chunk_text:
            retrieved_chunks.append(chunk_text)

    # Build prompt — include retrieved context + conversation history + instruction
    history_text = ""
    if session["history"]:
        formatted = []
        for (q, a) in session["history"][-6:]:  # keep last 6 turns
            formatted.append(f"User: {q}\nAssistant: {a}")
        history_text = "\n".join(formatted)

    context = "\n\n---\n\n".join(retrieved_chunks) if retrieved_chunks else ""
    prompt = (
    "You are a helpful assistant answering questions about a YouTube video transcript.\n"
    "Always answer in clear English, even if the transcript is in another language.\n"
    "If the answer is not explicitly in the provided excerpts, try to answer based on related information in them.\n"
    "Do not simply say you don't know unless there is no relevant clue at all.\n\n"
    f"Transcript excerpts:\n{context}\n\n"
    f"Conversation history:\n{history_text}\n\n"
    f"User: {query}\nAssistant:"
)


    # Call Gemini
    try:
        resp = model.generate_content(prompt)
        answer = resp.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # Save to history
    session["history"].append((query, answer))
    # Keep history bounded
    if len(session["history"]) > 40:
        session["history"] = session["history"][-40:]
        
    formatted_answer = format_response_as_points(answer)

    return {"answer": formatted_answer, "retrieved_count": len(retrieved_chunks)}
