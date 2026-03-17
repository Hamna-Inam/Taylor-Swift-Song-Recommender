# 🎵 Taylor Swift Song Recommender

A mood-based song recommender that takes how you're feeling and finds the Taylor Swift song that matches. Built as a learning project covering NLP, API development, and cloud deployment.

<img width="1203" height="672" alt="image" src="https://github.com/user-attachments/assets/3fe860ff-f68d-4743-92d2-d8a9a290d3c0" />



---

## How It Works

You describe your mood in plain text. The app finds the most emotionally similar Taylor Swift song from a dataset of 147 songs across all her albums.

```
"I feel heartbroken and miss someone I lost"
        ↓
   Qwen3-Embedding
        ↓
   FAISS vector search
        ↓
   my tears ricochet — folklore
```

---

## The NLP Pipeline

The interesting challenge here is the **domain gap** — Taylor Swift's lyrics are poetic, metaphorical, and narrative. Standard sentence embedding models struggle to match a plain mood description like *"I'm angry and betrayed"* against lyrics like *"Karma is the thunder rattling your cage."*

The solution was a two-stage pipeline:

**Stage 1 — Summarization**
Each song's lyrics are passed through `microsoft/Phi-3-mini-4k-instruct`, a small LLM that generates a 2-3 sentence description of the song's situation and emotional tone. For example:

> *"Back to December is a breakup song where the narrator regrets ending a relationship, missing specific details like his tan skin and kind eyes. The emotions expressed include sadness, regret, longing, and heartbreak."*

This bridges the domain gap — the summary is in plain descriptive language, just like a user's mood input.

**Stage 2 — Semantic Search**
Song summaries are embedded using `Qwen/Qwen3-Embedding` and stored in a FAISS index. At query time, the user's mood is embedded with the same model and cosine similarity retrieves the closest songs.

### Model Experiments

Several embedding models were tested before settling on Qwen3:

| Model | Notes |
|---|---|
| `all-MiniLM-L6-v2` | Too generic, poor on emotional queries |
| `multi-qa-mpnet-base-dot-v1` | Better asymmetric search, decent results |
| `all-roberta-large-v1` | Larger but no improvement |
| `sentence-t5-large` | High scores but poor differentiation |
| `Qwen/Qwen3-Embedding` | Best results across all test queries |

Without the Phi-3 summarization step, even the best embedding model struggled with lyrical language. With summaries, results improved significantly.

---

## Tech Stack

| Component | Tool |
|---|---|
| Summarization | Phi-3-mini-4k-instruct (Kaggle GPU) |
| Embeddings | Qwen3-Embedding |
| Vector Search | FAISS |
| API | FastAPI |
| Frontend | Vanilla HTML/CSS/JS |
| Storage | AWS S3 (FAISS index + dataset) |
| Deployment | AWS EC2 (Docker container) |
| Container Registry | GitHub Container Registry (GHCR) |

---

## Architecture

```
User (Browser)
    ↓
FastAPI (EC2, Docker)
    ↓ on startup
    ├── downloads FAISS index from S3
    └── loads Qwen3-Embedding from HuggingFace

At query time:
    User mood → embed → FAISS search → top 3 songs
```

The app is fully merged — embedding model, vector search, and API all run in one process on a single EC2 instance. This is appropriate at this scale since Qwen3-Embedding runs efficiently on CPU.

---

## Running Locally

```bash
git clone https://github.com/Hamna-Inam/taylor-recommender.git
cd taylor-recommender
pip install -r requirements.txt
```

Set your AWS credentials (needed to download the FAISS index from S3):
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=your_region
```

Run:
```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000`

---

## Running with Docker

```bash
docker build --platform linux/amd64 -t song-recommender .
docker run -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e AWS_DEFAULT_REGION=your_region \
  song-recommender
```

---

## API

**POST /recommend**
```json
{
  "mood": "I feel nostalgic and happy",
  "top_k": 3
}
```

Response:
```json
{
  "mood": "I feel nostalgic and happy",
  "recommendations": [
    { "song": "Holy Ground", "album": "Red", "score": 0.522 },
    { "song": "Fifteen", "album": "Fearless", "score": 0.498 },
    { "song": "The Best Day", "album": "Taylor Swift", "score": 0.444 }
  ]
}
```

---

## Limitations

- Dataset is 147 songs — small corpus means the 3rd recommendation is sometimes weak
- Phi-3 summaries occasionally hallucinate details not in the lyrics, especially for shorter songs
- Model was not fine-tuned on lyrics data — performance ceiling is set by the pretrained Qwen3 model

---

## What I Learned

- Why general-purpose embedding models struggle with poetic/lyrical text (domain gap)
- How summarization-augmented retrieval bridges that gap
- FastAPI for building ML inference APIs
- Docker multi-platform builds (ARM vs AMD64)
- AWS EC2, S3, and GHCR for deployment
