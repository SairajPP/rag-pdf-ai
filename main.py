import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import inngest
import inngest.fast_api
from dotenv import load_dotenv
from groq import Groq

# Reuse your existing loader and db logic
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStroage
from custom_types import RAGChunkAndSrc

load_dotenv()

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Application startup: FastAPI is ready!")
    yield
    logging.info("Application shutdown")

app = FastAPI(lifespan=lifespan)

# --- INNGEST SETUP ---
inngest_client = inngest.Inngest(
    app_id="rag_app",
    is_production=os.getenv("RENDER") == "true",
)

# --- 1. BACKGROUND JOB: EMBEDDING ---
# Note: We pass 'chunks' (text), not 'pdf_path', so this works on any server.
@inngest_client.create_function(
    fn_id="RAG: Embed and Upsert",
    trigger=inngest.TriggerEvent(event="rag/embed_chunks"),
)
async def rag_embed_chunks(ctx: inngest.Context):
    data = ctx.event.data
    chunks = data["chunks"]
    src_id = data["source_id"]

    # Embed and Upsert
    vectors = embed_texts(chunks)
    
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{src_id}:{i}")) for i in range(len(chunks))]
    payloads = [{"source": src_id, "text": chunk} for chunk in chunks]
    
    QdrantStroage().upsert(ids, vectors, payloads)
    
    return {"status": "completed", "count": len(chunks)}

# --- 2. API ENDPOINT: UPLOAD PDF ---
# Streamlit calls this endpoint to send the file to Render.
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4()}.pdf"
    
    try:
        # Save uploaded file temporarily on Render
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract text immediately
        chunks = load_and_chunk_pdf(temp_filename)
        
        # Send chunks to Inngest for background embedding
        await inngest_client.send(
            inngest.Event(
                name="rag/embed_chunks",
                data={"chunks": chunks, "source_id": file.filename}
            )
        )
        
        return {
            "message": "File received. Processing started.", 
            "filename": file.filename,
            "chunks_count": len(chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- 3. API ENDPOINT: CHAT ---
class ChatRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # Search Vector DB
        query_vector = embed_texts([req.question])[0]
        results = QdrantStroage().search(query_vector, top_k=req.top_k)
        
        if not results["contexts"]:
            return {"answer": "No relevant context found.", "sources": []}
        
        # Generate Answer
        context_str = "\n\n".join([f"- {c}" for c in results["contexts"]])
        
        system_prompt = "You are a helpful assistant. Answer using ONLY the provided context."
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {req.question}\nAnswer:"
        
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2
        )
        
        return {
            "answer": response.choices[0].message.content,
            "sources": results["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- SERVE INNGEST ---
#inngest.fast_api.serve(app, inngest_client, functions=[rag_embed_chunks])
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
