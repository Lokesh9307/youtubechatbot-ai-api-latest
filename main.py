import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import google.generativeai as genai
from youtube_utils import get_youtube_video_id, get_youtube_transcript
import dotenv
dotenv.load_dotenv()
# Load Gemini API Key
API_KEY = "AIzaSyATeungl9_J2SZ6l0aegBS0N87AxLZrnY4"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI(title="YouTube Summarizer API")

class VideoURLRequest(BaseModel):
    url: str

@app.post("/summarize")
async def summarize_video(data: VideoURLRequest):
    video_id = get_youtube_video_id(data.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL.")

    transcript = get_youtube_transcript(video_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not available for this video.")

    prompt = f"Summarize the following YouTube video transcript:\n\n{transcript}"
    try:
        response = model.generate_content(prompt)
        return {"summary": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    print("This script is intended to be run as a FastAPI application.")