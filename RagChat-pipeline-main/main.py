"""
RAG Pipeline — FastAPI Service
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Integrates:
  ✓ RecursiveCharacterTextSplitter  (500 tokens / 50 overlap)
  ✓ Hybrid Search                   (ChromaDB 0.7 + BM25 0.3)
  ✓ Multi-Query Retrieval           (3 query variations)
  ✓ Reciprocal Rank Fusion          (RRF across all queries)
  ✓ Cohere Reranking                (optional — needs COHERE_API_KEY)
  ✓ History-Aware Generation        (standalone question rewriting)
  ✓ GPT-4o Vision                   (image → searchable description)
  ✓ YouTube Transcript              (video URL → text → chunks)
  ✓ Per-Project ChromaDB Isolation  (collection per projectId)
  ✓ Persistent Storage              (survives restarts)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API contract (called by Spring Boot):
  POST /ingest        { projectId, type, content|filePath|videoUrl }
  POST /query         { projectId, question, chatHistory? }
  DELETE /project/{id}  — cleans up ChromaDB collection
  GET  /health
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import base64
import logging
from collections import defaultdict
from typing import List, Optional, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config (all overridable via environment variables) ─────────────────
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY   = os.getenv("COHERE_API_KEY")
CHROMA_PERSIST   = os.getenv("CHROMA_PERSIST_PATH", "./chroma_data")
PORT             = int(os.getenv("PORT", 8000))

# Retrieval tuning
TOP_K_RETRIEVE   = 15   # docs fetched per query before fusion/reranking
TOP_K_FINAL      = 5    # docs sent to LLM after reranking
QUERY_VARIATIONS = 3    # number of multi-query variations
RRF_K            = 60   # RRF constant (higher = less position penalty)

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to your environment variables.")


# ── Gemini LLM + Embedding Models ──────────────────────────────────────

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY,
    convert_system_message_to_human=True
)

# Reuse Gemini Flash for intermediate tasks
llm_mini = llm

# ── Text Splitter: 500 tokens / 50 overlap (from file 5) ──────────────
# Uses tiktoken encoder so chunk_size is actual token count, not chars
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4o",
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ── FastAPI App ────────────────────────────────────────────────────────
app = FastAPI(title="RAG Pipeline", version="1.0.0", description="Multi-modal RAG service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Request / Response Models ────────────────────────────────
class IngestRequest(BaseModel):
    projectId: str
    type: str                        # "TEXT" | "IMAGE" | "VIDEO"
    content:  Optional[str] = None   # TEXT: raw text content
    filePath: Optional[str] = None   # IMAGE: absolute path on Spring Boot server disk
    videoUrl: Optional[str] = None   # VIDEO: YouTube or direct video URL

class ChatMessage(BaseModel):
    role: str      # "user" | "assistant"
    content: str

class QueryRequest(BaseModel):
    projectId: str
    question: str
    chatHistory: Optional[List[ChatMessage]] = []

class IngestResponse(BaseModel):
    success: bool
    message: str
    chunksIngested: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str

# ═══════════════════════════════════════════════════════════════════════
# SECTION 1: ChromaDB Helpers — Per-Project Isolation
# Each project gets its own collection: "proj_{projectId}"
# ═══════════════════════════════════════════════════════════════════════

def _collection_name(project_id: str) -> str:
    """
    ChromaDB collection name. Must be 3-63 chars, alphanumeric + _ + - only.
    MongoDB IDs are 24 hex chars → "proj_{id}" = 29 chars ✓
    """
    return f"proj_{project_id}"


def _get_vectorstore(project_id: str) -> Chroma:
    """Open (or create) the ChromaDB collection for a project."""
    return Chroma(
        collection_name=_collection_name(project_id),
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _get_all_docs(project_id: str) -> List[Document]:
    """Fetch all stored documents for a project (needed to build BM25 index)."""
    vs  = _get_vectorstore(project_id)
    raw = vs._collection.get(include=["documents", "metadatas"])
    return [
        Document(page_content=txt, metadata=meta or {})
        for txt, meta in zip(raw["documents"], raw["metadatas"])
        if txt  # skip empty strings
    ]


def _has_context(project_id: str) -> bool:
    """Return True if the project has any ingested context."""
    try:
        vs = _get_vectorstore(project_id)
        return vs._collection.count() > 0
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2: Hybrid Retriever — ChromaDB (0.7) + BM25 (0.3)
# Source: 12_hybrid_search.ipynb
# ═══════════════════════════════════════════════════════════════════════

def _build_hybrid_retriever(project_id: str) -> Optional[EnsembleRetriever]:
    """
    Build a hybrid retriever combining:
      - ChromaDB (semantic / dense retrieval)   weight=0.7
      - BM25     (keyword / sparse retrieval)   weight=0.3

    Returns None if no context has been ingested yet.
    """
    all_docs = _get_all_docs(project_id)

    if not all_docs:
        log.warning(f"[{project_id}] No context in vector store — skipping hybrid retriever")
        return None

    vs = _get_vectorstore(project_id)

    vector_retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RETRIEVE}
    )

    bm25_retriever   = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = TOP_K_RETRIEVE

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3],
    )


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3: Multi-Query Retrieval
# Source: 10_multi_query_retrieval.py
# ═══════════════════════════════════════════════════════════════════════

def _generate_query_variations(question: str) -> List[str]:
    """
    Generate QUERY_VARIATIONS alternative phrasings of the question.
    Uses Gemini Flash for query variations.
    """
    prompt = (
        f"Generate {QUERY_VARIATIONS} different search queries for this question.\n"
        f"Each should approach the same topic from a different angle.\n"
        f"Return ONLY the queries, one per line, no numbering, bullets, or extra text.\n\n"
        f"Question: {question}"
    )
    result = llm_mini.invoke([HumanMessage(content=prompt)])
    lines  = [l.strip() for l in result.content.strip().split("\n") if l.strip()]
    return lines[:QUERY_VARIATIONS]


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4: Reciprocal Rank Fusion
# Source: 11_reciprocal_rank_fusion.py
# ═══════════════════════════════════════════════════════════════════════

def _reciprocal_rank_fusion(
    results_list: List[List[Document]],
    k: int = RRF_K
) -> List[Document]:
    """
    Fuse multiple ranked document lists using RRF.
    Score = Σ  1 / (k + position)  for each list the doc appears in.
    Returns deduplicated docs sorted by descending RRF score.
    """
    scores: Dict[str, float]    = defaultdict(float)
    docs:   Dict[str, Document] = {}

    for result in results_list:
        for position, doc in enumerate(result, 1):
            key           = doc.page_content          # unique by content
            scores[key]  += 1.0 / (k + position)
            docs[key]     = doc

    sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
    return [docs[key] for key in sorted_keys]


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5: Cohere Reranker (optional)
# Source: 13_reranker.ipynb
# Falls back to simple top-k slice if COHERE_API_KEY is not set.
# ═══════════════════════════════════════════════════════════════════════

def _rerank(docs: List[Document], query: str) -> List[Document]:
    """
    Rerank documents with Cohere if COHERE_API_KEY is available.
    Otherwise returns the top TOP_K_FINAL docs from the RRF-ordered list.
    """
    if not docs:
        return []

    if COHERE_API_KEY:
        try:
            from langchain_cohere import CohereRerank
            reranker = CohereRerank(
                cohere_api_key=COHERE_API_KEY,
                model="rerank-english-v3.0",
                top_n=TOP_K_FINAL,
            )
            reranked = reranker.compress_documents(docs, query)
            log.info(f"Cohere reranking applied — {len(reranked)} docs returned")
            return reranked
        except Exception as e:
            log.warning(f"Cohere reranking failed, falling back to top-k: {e}")

    return docs[:TOP_K_FINAL]


# ═══════════════════════════════════════════════════════════════════════
# SECTION 6: History-Aware Question Rewriting
# Source: 4_history_aware_generation.py
# ═══════════════════════════════════════════════════════════════════════

def _rewrite_question(question: str, history: List[ChatMessage]) -> str:
    """
    If there's prior conversation, rewrite the question to be standalone
    so it retrieves correctly without relying on implicit context.
    """
    if not history:
        return question

    # Use last 6 messages (3 turns) to avoid context overflow
    recent = history[-6:]
    history_text = "\n".join([f"{m.role}: {m.content}" for m in recent])

    prompt = (
        f"Given this conversation history, rewrite the new question to be completely "
        f"standalone and searchable without needing the history.\n"
        f"Return ONLY the rewritten question — no explanation, no quotes.\n\n"
        f"Conversation History:\n{history_text}\n\n"
        f"New Question: {question}"
    )

    result = llm_mini.invoke([HumanMessage(content=prompt)])
    rewritten = result.content.strip()
    log.info(f"Question rewritten: '{question}' → '{rewritten}'")
    return rewritten


# ═══════════════════════════════════════════════════════════════════════
# SECTION 7: Ingestion Handlers
# ═══════════════════════════════════════════════════════════════════════

def _ingest_text(project_id: str, text: str) -> int:
    """
    Chunk text with RecursiveCharacterTextSplitter (500 tok / 50 overlap)
    and store in this project's ChromaDB collection.
    Source: 1_ingestion_pipeline.py + 5_recursive_character_text_splitter.py
    """
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"project_id": project_id, "type": "text"}],
    )
    vs = _get_vectorstore(project_id)
    vs.add_documents(chunks)
    log.info(f"[{project_id}] TEXT: ingested {len(chunks)} chunks")
    return len(chunks)


def _ingest_image(project_id: str, file_path: str) -> int:
    """
   Read image from disk → Gemini Vision creates a rich searchable description
    → embed and store the description in ChromaDB.
    Source: 8_multi_modal_rag.ipynb (create_ai_enhanced_summary)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image not found at path: {file_path}")

    # Read and encode image
    with open(file_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    ext  = file_path.lower().rsplit(".", 1)[-1] if "." in file_path else "jpg"
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp",
            "gif": "image/gif"}.get(ext, "image/jpeg")

    # Ask Gemini Vision to produce a thorough searchable description
    vision_prompt = """You are creating a detailed, searchable description for a document retrieval system.

Analyze this image and produce a comprehensive description covering:
1. All visible text (transcribe it exactly)
2. Charts or graphs — describe the data, axes, trends, and key values
3. Diagrams or flowcharts — describe each element and how they connect
4. Tables — reproduce the data in readable form
5. Main visual objects and their relationships
6. Key concepts, topics, and facts depicted
7. Alternative search terms a user might use to find this image

Be thorough and specific — prioritize findability over brevity."""

    message = HumanMessage(content=[
        {"type": "text", "text": vision_prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
    ])

    result      = llm.invoke([message])
    description = result.content

    doc = Document(
        page_content=description,
        metadata={"project_id": project_id, "type": "image", "source": file_path},
    )

    vs = _get_vectorstore(project_id)
    vs.add_documents([doc])
    log.info(f"[{project_id}] IMAGE: ingested vision description ({len(description)} chars)")
    return 1


def _extract_youtube_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    if "youtube.com/watch?v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    if "youtube.com/embed/" in url:
        return url.split("embed/")[1].split("?")[0]
    return None


def _ingest_video(project_id: str, video_url: str) -> int:
    """
    Extract transcript from YouTube URL using youtube-transcript-api,
    chunk it with the same splitter, and store in ChromaDB.
    For non-YouTube URLs, stores a placeholder.
    """
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

    video_id = _extract_youtube_id(video_url)

    if not video_id:
        # Non-YouTube URL: store as a placeholder note
        doc = Document(
            page_content=(
                f"Video content from: {video_url}\n"
                "Transcript extraction is currently only supported for YouTube URLs."
            ),
            metadata={"project_id": project_id, "type": "video", "source": video_url},
        )
        vs = _get_vectorstore(project_id)
        vs.add_documents([doc])
        log.warning(f"[{project_id}] VIDEO: Non-YouTube URL, stored as placeholder")
        return 1

    # Fetch transcript — try English first, fall back to any available language
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    except NoTranscriptFound:
        try:
            transcript_data  = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript       = transcript_data.find_generated_transcript(["en"])
            transcript_list  = transcript.fetch()
        except Exception as e:
            raise ValueError(f"No transcript available for video {video_id}: {e}")
    except TranscriptsDisabled:
        raise ValueError(f"Transcripts are disabled for video {video_id}")

    # Join transcript segments into full text
    full_text = " ".join([seg["text"] for seg in transcript_list])

    # Add title metadata if possible
    chunks = splitter.create_documents(
        texts=[full_text],
        metadatas=[{"project_id": project_id, "type": "video", "source": video_url}],
    )

    vs = _get_vectorstore(project_id)
    vs.add_documents(chunks)
    log.info(f"[{project_id}] VIDEO: ingested {len(chunks)} transcript chunks from {video_url}")
    return len(chunks)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 8: Answer Generation
# Source: 3_answer_generation.py + 4_history_aware_generation.py
# ═══════════════════════════════════════════════════════════════════════

def _generate_answer(
    question: str,
    context_docs: List[Document],
    history: List[ChatMessage],
) -> str:
    """
    Generate a grounded answer using retrieved context and conversation history.
    Gemini is instructed to stay within the provided context.
    """
    # Build numbered context block (from 3_answer_generation.py pattern)
    context_block = "\n\n".join([
        f"[Document {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(context_docs)
    ])

    answer_prompt = (
        f"Based on the following documents, answer this question:\n"
        f"Question: {question}\n\n"
        f"Documents:\n{context_block}\n\n"
        f"Instructions:\n"
        f"- Answer using ONLY the information from these documents.\n"
        f"- Be clear, specific, and thorough.\n"
        f"- If the documents don't contain enough information, say: "
        f"'I don't have enough information to answer that question based on the provided context.'\n"
        f"- Do not make up facts or use external knowledge."
    )

    # Build message list with history (from 4_history_aware_generation.py pattern)
    messages = [
        SystemMessage(content=(
            "You are a helpful AI assistant that answers questions strictly based on "
            "provided context documents. You have access to conversation history for context."
        ))
    ]

    # Include last 6 messages (3 turns) of history
    for msg in history[-6:]:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))

    messages.append(HumanMessage(content=answer_prompt))

    result = llm.invoke(messages)
    return result.content


# ═══════════════════════════════════════════════════════════════════════
# SECTION 9: Full RAG Pipeline (retrieval → fusion → rerank → generate)
# ═══════════════════════════════════════════════════════════════════════

def _run_rag_pipeline(
    project_id: str,
    question: str,
    history: List[ChatMessage],
) -> str:
    """
    Full pipeline:
      1. Rewrite question if history exists
      2. Generate multi-query variations
      3. Retrieve with hybrid search for each variation
      4. Fuse with RRF
      5. Rerank with Cohere (optional)
      6. Generate grounded answer with history context
    """

    # Step 1: History-aware question rewriting
    search_question = _rewrite_question(question, history)

    # Step 2: Build hybrid retriever
    retriever = _build_hybrid_retriever(project_id)

    if retriever is None:
        log.info(f"[{project_id}] No context — falling back to LLM-only response")
        result = llm.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=question),
        ])
        return result.content

    # Step 3: Multi-query variations
    variations   = _generate_query_variations(search_question)
    all_queries  = [search_question] + variations
    log.info(f"[{project_id}] Multi-query: {all_queries}")

    # Step 4: Retrieve for each query
    all_results: List[List[Document]] = []
    for q in all_queries:
        try:
            docs = retriever.invoke(q)
            all_results.append(docs)
            log.info(f"[{project_id}]   '{q[:60]}' → {len(docs)} docs")
        except Exception as e:
            log.warning(f"[{project_id}] Retrieval failed for variation '{q}': {e}")

    if not all_results:
        raise RuntimeError("All retrieval attempts failed")

    # Step 5: RRF fusion
    fused_docs = _reciprocal_rank_fusion(all_results)
    log.info(f"[{project_id}] RRF fused {sum(len(r) for r in all_results)} → {len(fused_docs)} unique docs")

    # Step 6: Rerank (Cohere or top-k slice)
    final_docs = _rerank(fused_docs, search_question)
    log.info(f"[{project_id}] Final context: {len(final_docs)} docs for generation")

    # Step 7: Generate answer
    answer = _generate_answer(question, final_docs, history)
    return answer


# ═══════════════════════════════════════════════════════════════════════
# SECTION 10: API Routes
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
def health_check():
    """Health check — Spring Boot calls this to verify the service is up."""
    return {
        "status":        "ok",
        "service":       "RAG Pipeline",
        "cohere":        bool(COHERE_API_KEY),
        "chroma_path":   CHROMA_PERSIST,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """
    Receive context from Spring Boot and ingest it into the project's ChromaDB collection.
    Called by RagPipelineClient.ingestContext() after a user uploads context.
    """
    log.info(f"[{req.projectId}] Ingest request — type={req.type}")

    try:
        chunks_count = 0

        if req.type == "TEXT":
            if not req.content or not req.content.strip():
                raise HTTPException(status_code=400, detail="content is required for TEXT type")
            chunks_count = _ingest_text(req.projectId, req.content)

        elif req.type == "IMAGE":
            if not req.filePath:
                raise HTTPException(status_code=400, detail="filePath is required for IMAGE type")
            chunks_count = _ingest_image(req.projectId, req.filePath)

        elif req.type == "VIDEO":
            if not req.videoUrl or not req.videoUrl.strip():
                raise HTTPException(status_code=400, detail="videoUrl is required for VIDEO type")
            chunks_count = _ingest_video(req.projectId, req.videoUrl)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown type '{req.type}'. Supported: TEXT, IMAGE, VIDEO"
            )

        return IngestResponse(
            success=True,
            message=f"{req.type} context ingested successfully for project {req.projectId}",
            chunksIngested=chunks_count,
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        log.error(f"[{req.projectId}] File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        log.error(f"[{req.projectId}] Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"[{req.projectId}] Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Receive a question from Spring Boot and return an AI-generated answer.
    Runs the full RAG pipeline: hybrid search + RRF + reranking + Gemini.
    Called by RagPipelineClient.query().
    """
    log.info(f"[{req.projectId}] Query: '{req.question[:80]}' | history={len(req.chatHistory)} msgs")

    try:
        answer = _run_rag_pipeline(
            project_id=req.projectId,
            question=req.question,
            history=req.chatHistory,
        )
        return QueryResponse(answer=answer)

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[{req.projectId}] Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.delete("/project/{project_id}")
def delete_project(project_id: str):
    """
    Delete all vectors for a project from ChromaDB.
    Should be called when a project is deleted in the Spring Boot backend.
    """
    log.info(f"[{project_id}] Deleting ChromaDB collection")
    try:
        vs = _get_vectorstore(project_id)
        vs._client.delete_collection(_collection_name(project_id))
        return {"success": True, "message": f"Collection deleted for project {project_id}"}
    except Exception as e:
        log.warning(f"[{project_id}] Could not delete collection: {e}")
        return {"success": False, "message": str(e)}


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting RAG Pipeline on port {PORT}")
    log.info(f"ChromaDB persist path: {CHROMA_PERSIST}")
    log.info(f"Cohere reranking: {'enabled' if COHERE_API_KEY else 'disabled (set COHERE_API_KEY to enable)'}")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
