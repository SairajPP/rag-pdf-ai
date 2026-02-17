RAG PDF AI — Groq + Qdrant + FastAPI

A Retrieval-Augmented Generation (RAG) system that:

Ingests PDFs

Chunks and embeds text using SentenceTransformers

Stores vectors in Qdrant

Retrieves relevant context

Generates answers using Groq LLM (Llama 3.1)

Provides a Streamlit UI

Architecture
PDF → Chunk → Embed → Qdrant
                        ↓
Question → Embed → Search → Context → Groq LLM → Answer

Tech Stack

FastAPI

Inngest (event-driven orchestration)

Qdrant (vector database)

SentenceTransformers (embeddings)

Groq LLM (Llama 3.1-8B-Instant)

Streamlit (frontend)

Setup Instructions
1. Clone Repository
git clone https://github.com/SairajPP/rag-pdf-ai.git
cd rag-pdf-ai

2. Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Add Environment Variables

Create a .env file:

GROQ_API_KEY=your_groq_key_here


Do not commit .env to GitHub.

5. Run Qdrant (Docker)
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  qdrant/qdrant


Verify:

http://localhost:6333

6. Start Backend
uvicorn main:app --reload

7. Start Streamlit UI
streamlit run streamlit_app.py

Features

Event-driven ingestion via Inngest

Async-safe Groq LLM calls

Custom Pydantic models

Local embedding model (no OpenAI dependency)

Fully self-hostable

Example Query

Upload a PDF and ask:

What are the key skills mentioned in this resume?


The system:

Retrieves top-k relevant chunks

Sends context to Groq

Returns grounded answer with sources

Security Notes

API keys must be stored in .env

Never hardcode secrets

Ensure .env is included in .gitignore

Deployment Options

You can deploy this on:

Render (backend)

Railway

Fly.io

AWS EC2

Docker container

Streamlit Cloud (frontend only)

License

MIT License
