from fastapi import FastAPI
from pydantic import BaseModel
from app.recommender import recommend

app = FastAPI()

class MoodRequest(BaseModel):
    mood: str
    top_k: int = 3

@app.get("/")
def root():
    return {"message": "Taylor Swift Song Recommender"}

@app.post("/recommend")
def get_recommendation(request: MoodRequest):
    results = recommend(request.mood, request.top_k)
    return {"mood": request.mood, "recommendations": results}