from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.recommender import recommend as get_recommendations



app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")


class MoodRequest(BaseModel):
    mood: str
    top_k: int = 1

@app.get("/")
def root():
    return FileResponse("app/static/index.html")

@app.post("/recommend")
def get_recommendation(request: MoodRequest):
    results = get_recommendations(request.mood, request.top_k)
    return {"mood": request.mood, "recommendations": results}