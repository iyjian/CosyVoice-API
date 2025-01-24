from pathlib import Path
import openai
import os

# export OPENAI_BASE_URL="http://192.168.1.53:4000"
# export OPENAI_API_KEY="XXXXXXXXXXXXXXXX"

openai.base_url = os.getenv("OPENAI_BASE_URL")
openai.api_key = os.getenv("OPENAI_API_KEY")

speech_file_path = Path(__file__).parent / "speech.mp3"

client = openai.OpenAI()

response = client.audio.speech.create(
    model="cosyvoice2",
    voice="中文男",
    input="Hello world! This is a streaming test.",
    speed=1,
    response_format='mp3',
)

response.stream_to_file(speech_file_path)