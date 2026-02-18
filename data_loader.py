from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

print("Preloading SentenceTransformer model...")
_model = SentenceTransformer(
    EMBED_MODEL_NAME,
    device="cpu"
)
print("Model loaded successfully.")

splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

def load_and_chunk_pdf(path: str) -> list[str]:
    docs = PDFReader().load_data(file=path)

    text_blocks = [
        d.text for d in docs
        if getattr(d, "text", None)
    ]

    chunks = []
    for text in text_blocks:
        chunks.extend(splitter.split_text(text))

    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    vectors = _model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    return vectors.tolist()
