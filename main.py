import logging
import os
import uuid
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
from groq import Groq

# Correct imports
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStroage
from custom_types import (
    RAGChunkAndSrc,
)

load_dotenv()

# --- LIFESPAN MANAGER (Prevents Timeout) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Application startup: FastAPI is ready!")
    yield
    logging.info("Application shutdown")

app = FastAPI(lifespan=lifespan)

# Setup Inngest
inngest_client = inngest.Inngest(
    app_id="rag_app",
    is_production=os.getenv("RENDER") == "true", # Auto-detect production
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):

    def _load():
        pdf_path = ctx.event.data["pdf_path"]
        src_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=src_id).model_dump()

    def _upsert(data_dict):
        data = RAGChunkAndSrc(**data_dict)
        chunks = data.chunks
        src_id = data.source_id
        
        # This will trigger lazy load of model safely
        vectors = embed_texts(chunks)

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{src_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {"source": src_id, "text": chunks[i]}
            for i in range(len(chunks))
        ]

        QdrantStroage().upsert(ids, vectors, payloads)
        return {"ingested": len(chunks)}

    chunks_and_src = await ctx.step.run("load-and-chunk", _load)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src))
    return ingested

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):

    def _search(q, k):
        query_vector = embed_texts([q])[0]
        store = QdrantStroage()
        return store.search(query_vector, top_k=k)

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k))

    context_block = "\n\n".join(f"- {c}" for c in found["contexts"])

    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using only the context above."
    )

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = await asyncio.to_thread(
        client.chat.completions.create,
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You answer questions using only the provided context."},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": found["sources"],
        "num_contexts": len(found["contexts"]),
    }

inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_ingest_pdf, rag_query_pdf_ai],
)