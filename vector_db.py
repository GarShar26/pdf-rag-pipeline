from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    
    def __init__(self, url="http://localhost:6333", collection="docs_ollama", dim=768):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

     
        if self.client.collection_exists(self.collection):
            coll_info = self.client.get_collection(self.collection)
            current_dim = coll_info.config.params.vectors.size
            if current_dim != dim:
                print(f"⚠️ Migration: Collection '{self.collection}' has dimension {current_dim} (Old).")
                self.client.delete_collection(self.collection)

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )
        points = result.points
        contexts = []
        sources = set()

        for r in points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                if source:
                    sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}