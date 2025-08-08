# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os

from youtube_utils import get_youtube_video_id, get_youtube_transcript

# FastAPI app
app = FastAPI()

# Load API key for Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in environment variables.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


class VideoRequest(BaseModel):
    url: str


@app.post("/summarize")
def summarize_video(request: VideoRequest):
    video_id = get_youtube_video_id(request.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    transcript = get_youtube_transcript(video_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")

    try:
        prompt = f"Summarize this YouTube video:\n\n{transcript}"
        response = model.generate_content(prompt)
        return {"summary": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}