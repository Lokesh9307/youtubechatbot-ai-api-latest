import google.generativeai as genai
import os
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import re

API_KEY = "AIzaSyATeungl9_J2SZ6l0aegBS0N87AxLZrnY4"

genai.configure(api_key=API_KEY)

def get_youtube_video_id(url):
    """
    Extracts the YouTube video ID from a given URL, supporting various formats.
    """
    if "youtu.be/" in url:
        # Handle short URLs like https://youtu.be/VIDEO_ID
        return url.split("youtu.be/")[1].split("?")[0].split("&")[0]
    elif "youtube.com/watch?v=" in url:
        # Handle standard URLs like https://www.youtube.com/watch?v=VIDEO_ID
        query = urlparse(url).query
        params = parse_qs(query)
        return params.get('v', [None])[0]
    elif "youtube.com/embed/" in url:
        # Handle embed URLs like https://www.youtube.com/embed/VIDEO_ID
        return url.split("youtube.com/embed/")[1].split("?")[0].split("&")[0]
    elif "youtube.com/v/" in url:
        # Handle old style URLs like https://www.youtube.com/v/VIDEO_ID
        return url.split("youtube.com/v/")[1].split("?")[0].split("&")[0]
    elif "youtube.com/shorts/" in url:
        # Handle YouTube Shorts URLs
        return url.split("youtube.com/shorts/")[1].split("?")[0].split("&")[0]
    else:
        # Attempt to find a video ID using a regex for more complex cases
        match = re.search(r'(?:v=|/videos/|embed/|youtu.be/|\/v\/|\/e\/|watch\?v=|&v=|user\/\S+\/|user\/[^#]+\/#p\/a\/u\/1\/|user\/[^#]+\/#p\/u\/\d\/|user\/[^#]+\/#p\/a\/u\/\d\/|user\/[^#]+\/#p\/f\/|\.be\/)([a-zA-Z0-9_-]{11})', url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_id):
    """Fetches the transcript for a given YouTube video ID."""
    try:
        ytt = YouTubeTranscriptApi()
        # List all available transcripts for the video
        transcript_list = ytt.list(video_id)
        
        # Try to find an English transcript, or the first available if English is not found
        transcript = None
        try:
            transcript = transcript_list.find_transcript(['en','hi'])
        except Exception:
            # If English is not found, try to get the first available transcript
            if transcript_list:
                transcript = next(iter(transcript_list)) # Get the first transcript object

        if transcript:
            # Fetch the actual transcript content
            fetched_transcript = transcript.fetch()
            transcript_text = " ".join([item.text for item in fetched_transcript])
            return transcript_text
        else:
            print(f"No suitable transcript found for video ID: {video_id}.")
            return None
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

try:
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")

    youtube_url = input("Enter the YouTube video URL: ")
    video_id = get_youtube_video_id(youtube_url)
    
    if video_id:
        transcript = get_youtube_transcript(video_id)

        if transcript:
            prompt = f"Summarize the following YouTube video transcript:\n\n{transcript}"
            response = model.generate_content(prompt)
            print(response.text)
        else:
            print(f"Could not retrieve transcript for the video ID: {video_id}.")
    else:
        print(f"Could not extract video ID from the URL: {youtube_url}.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure you have a valid API key and internet connection.")
    print("Also, check if the model name is correct and accessible.")
