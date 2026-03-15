import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import boto3
import os


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BUCKET_NAME = "amzn-s3-faiss"
#DATA_PATH = "data/taylor_swift_with_summaries-2.xlsx"
#INDEX_PATH = "data/songs.index"

#df = pd.read_excel(DATA_PATH)
#model = SentenceTransformer(MODEL_NAME)
#index = faiss.read_index(INDEX_PATH)

def download_from_s3():
    s3 = boto3.client('s3')
    if not os.path.exists("songs.index"):
        print("Downloading FAISS index from S3...")
        s3.download_file(BUCKET_NAME, 'songs.index', 'songs.index')
    if not os.path.exists("songs_metadata.csv"):
        print("Downloading metadata from S3...")
        s3.download_file(BUCKET_NAME, 'taylor_swift_with_summaries-2.xlsx', 'taylor_swift_with_summaries-2.xlsx')

download_from_s3()

df = pd.read_excel("taylor_swift_with_summaries-2.xlsx")
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index("songs.index")

def recommend(user_input: str, top_k: int = 3):
    user_embedding = model.encode(user_input).astype('float32').reshape(1, -1)
    faiss.normalize_L2(user_embedding)
    
    scores, indices = index.search(user_embedding, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "song": df['Song Name'][idx],
            "album": df['Album'][idx],
            "score": round(float(score), 3)
        })
    return results