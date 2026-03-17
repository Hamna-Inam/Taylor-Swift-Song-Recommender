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

import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util


def download_from_s3():
    s3 = boto3.client('s3')
    if not os.path.exists("swift_data.json"):
        print("Downloading data from S3...")
        s3.download_file(BUCKET_NAME, 'swift_data1.json', 'swift_data.json')

download_from_s3()

with open('swift_data.json', 'r', encoding='utf-8') as f:
    swift_data = json.load(f)

bi_model = SentenceTransformer('all-MiniLM-L6-v2')
cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

thematic_key = 'Thematic Logic (The "Why")'
song_moods = [
    f"{s[thematic_key]} {s['User Mood (First-Person Input)']}"
    for s in swift_data
]
song_embeddings = bi_model.encode(song_moods, convert_to_tensor=True)

def recommend(user_input: str, top_k: int = 1):
    user_vec = bi_model.encode(user_input, convert_to_tensor=True)
    hits = util.semantic_search(user_vec, song_embeddings, top_k=10)[0]
    
    cross_inp = [[user_input, song_moods[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_model.predict(cross_inp)
    
    for idx in range(len(hits)):
        hits[idx]['cross_score'] = float(cross_scores[idx])
    
    hits = sorted(hits, key=lambda x: x['cross_score'], reverse=True)
    
    results = []
    for hit in hits[:top_k]:
        song = swift_data[hit['corpus_id']]
        results.append({
            "song": song['Song Name'],
            "album": song.get('Album', ''),
            "score": round(hit['cross_score'], 1)
        })
    return results