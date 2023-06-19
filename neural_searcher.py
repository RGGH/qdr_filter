from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter
from qdrant_client.http import models


class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # initialize Qdrant client
        self.qdrant_client = QdrantClient("http://localhost:6333")

    def search(self, text: str, city: str):
        # Convert text query into vector
        vector = self.model.encode(text).tolist()

        city_of_interest = city

        # Define a filter for cities
        city_filter = Filter(
            must=[
                models.FieldCondition(
                    key="city",
                    match=models.MatchValue(value=city_of_interest),
                )
            ]
        )

        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=city_filter,  # If you don't want any filters for now
            top=2,  # 5 the most closest results is enough
        )
        # `search_result` contains found vector ids with similarity scores along with the stored payload
        # In this function you are interested in payload only
        payloads = [hit.payload for hit in search_result]
        return payloads[:2]
