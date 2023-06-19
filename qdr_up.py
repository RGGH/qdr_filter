# pip install sentence-transformers numpy pandas tqdm
# https://storage.googleapis.com/generall-shared-data/startups_demo.json

from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pandas as pd
from tqdm.notebook import tqdm

# Import client library
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

qdrant_client = QdrantClient("http://localhost:6333")


model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

df = pd.read_json("./startups_demo.json", lines=True)
print(df[:4])

# Vectors

vectors = model.encode(
    [row.alt + ". " + row.description for row in df.itertuples()], show_progress_bar=True
)

print(vectors.shape)
np.save("startup_vectors.npy", vectors, allow_pickle=False)

try:
    qdrant_client.recreate_collection(
        collection_name="startupsX",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
except:
    pass

collectionz = qdrant_client.get_collections()
print(collectionz)

vectors = np.load("./startup_vectors.npy")

# Payload
fd = open("./startups_demo.json")
payload = map(json.loads, fd)


qdrant_client.upload_collection(
    collection_name="startupsX",
    vectors=vectors,
    payload=payload,
    ids=None,  # Vector ids will be assigned automatically
    batch_size=256,  # How many vectors will be uploaded in a single request?
)
