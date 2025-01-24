from pathlib import Path
import openai
import os

openai.base_url = os.getenv("OPENAI_BASE_URL")
openai.api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI()

# 上传文件到服务器，上传后可以用文件名引用已上传的文件
with open('./liujie16k.wav', 'rb') as file:
    response = client.files.create(
        file=file,
        purpose='cosyvoice-prompt_speech_16k'  # 文件用途
    )
    print(response)


# 声音克隆
speech_file_path = Path(__file__).parent / "speech.mp3"

response = client.audio.speech.create(
    model="cosyvoice2",
    input="好好学习，天天向上！上学校！",
    voice="",
    # speed=1, # optional
    # response_format='mp3', # optional same as openai API
    extra_body = {
        "prompt_speech_16k": 'liujie16k.wav',
        "sample_text": '人工智能已成为上海引领未来发展的关键先导产业，其发展势头迅猛，国家产业规模预计在2020年到2028年间实现跨越式增长。'
    }
)

response.stream_to_file(speech_file_path)

# 指令
speech_file_path = Path(__file__).parent / "speech_instruct.mp3"

response = client.audio.speech.create(
    model="cosyvoice2",
    input="好好学习，天天向上！上学校！",
    voice="",
    # speed=1, # optional
    # response_format='mp3', # optional same as openai API
    extra_body = {
        "prompt_speech_16k": 'liujie16k.wav',
        "prompt": '请用四川话说'
    }
)

response.stream_to_file(speech_file_path)