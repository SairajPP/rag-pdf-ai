import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStroage:
    def __init__(self, collection="docs", dim=384):
        self.collection = collection
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_api_key:
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=30,
            )
        else:
            self.client = QdrantClient(
                url=qdrant_url,
                timeout=30,
            )

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

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
