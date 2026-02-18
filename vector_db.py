import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStroage:
    def __init__(self):
        # Read from Environment Variables (set these in Render Dashboard)
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = os.getenv("QDRANT_API_KEY", None)
        self.collection = "docs"
        self.dim = 384

        self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=30)

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        res = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        contexts = []
        sources = []

        for point in res.points:
            payload = point.payload or {}
            text = payload.get("text")
            source = payload.get("source")

            if text:
                contexts.append(str(text))

            if source and source not in sources:
                sources.append(str(source))

        return {
            "contexts": contexts,
            "sources": sources,
        }