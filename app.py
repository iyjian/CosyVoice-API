from flask import Flask, Response, request, jsonify, stream_with_context
# from celery import Celery
import io
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import torch
from typing import TypedDict, Literal
from werkzeug.utils import secure_filename
import boto3
import os
from datetime import datetime

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
    region_name='us-east-1',
    endpoint_url=os.getenv("S3_ENDPOINT")
)

class CreateSpeechPayLoad(TypedDict):
    input: str
    voice: str
    response_format: Literal['mp3', 'wav', 'opus', 'aac', 'flac', 'pcm']
    speed: float
    model: Literal['CosyVoice2-0.5B', 'CosyVoice-300M-SFT']
    sample_text: str
    prompt_speech_16k: str
    prompt: str    


app = Flask(__name__)

# TODO: set model param in env
model = 'CosyVoice2-0.5B'

cosyvoice = None
if model.startswith('CosyVoice2'):
  cosyvoice = CosyVoice2(f'pretrained_models/{model}', load_jit=False, load_trt=False, fp16=False)
else:
  cosyvoice = CosyVoice(f'pretrained_models/{model}', load_jit=False, load_trt=False, fp16=False)

@app.route('/v1/files', methods=['POST'])
def upload_file():
    # 获取token
    auth_header = request.headers.get('Authorization')
    token = auth_header.split(' ')[1]
    print(token)

    # 检查文件是否存在
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    purpose = request.form.get('purpose')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 安全地处理文件名
    filename = secure_filename(file.filename)
    unique_filename = token + '_' + secure_filename(file.filename)

    try:
        # 上传文件到 S3
        s3_client.upload_fileobj(
            file,
            os.getenv('S3_BUCKET'),
            unique_filename
        )
        
        object_head_info = s3_client.head_object(
            Bucket=os.getenv('S3_BUCKET'),
            Key=unique_filename
        )

        # 返回成功响应
        return jsonify({
            "id": object_head_info['ETag'],
            "bytes": object_head_info['ContentLength'],
            "purpose": purpose,
            "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "filename": filename,
            "object": "file"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_audio_stream(payload: CreateSpeechPayLoad):
    print(payload)
    # TODO: stream=True时生成的声音很不稳定
    if 'prompt' in payload:
        print('type1', payload['input'], payload['prompt'], payload['prompt_speech_16k'])
        for i, j in enumerate(cosyvoice.inference_instruct2(payload['input'], payload['prompt'], load_wav(payload['prompt_speech_16k'], 16000), stream=False, speed=payload['speed'], text_frontend=True)):
            audio_chunk = j['tts_speech']
                    
            # Save the audio chunk to a buffer
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio_chunk, cosyvoice.sample_rate, format=payload['response_format'])
            buffer.seek(0)

            # Yield the audio chunk to the client
            yield buffer.read()   
    elif 'sample_text' in payload:
        print('type2', payload['input'], payload['sample_text'], payload['prompt_speech_16k'])
        for i, j in enumerate(cosyvoice.inference_zero_shot(payload['input'], payload['sample_text'], load_wav(payload['prompt_speech_16k'], 16000), stream=False, speed=payload['speed'], text_frontend=True)):
            audio_chunk = j['tts_speech']
                    
            # Save the audio chunk to a buffer
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio_chunk, cosyvoice.sample_rate, format=payload['response_format'])
            buffer.seek(0)

            # Yield the audio chunk to the client
            yield buffer.read()   
    else:
        print('type3', payload['input'], payload['voice'])
        for i, j in enumerate(cosyvoice.inference_sft(payload['input'], payload['voice'], stream=False, speed=payload['speed'], text_frontend=True)):
            audio_chunk = j['tts_speech']
                    
            # Save the audio chunk to a buffer
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio_chunk, cosyvoice.sample_rate, format=payload['response_format'])
            buffer.seek(0)

            # Yield the audio chunk to the client
            yield buffer.read()        

@app.route('/v1/audio/speech', methods=['POST'])
def generate_speech():
    auth_header = request.headers.get('Authorization')
    token = auth_header.split(' ')[1]
    print(token)

    payload: CreateSpeechPayLoad = request.json

    # data validation
    if not payload:
        return jsonify({"error": {"message": "You must provide a 'input' parameter"}}), 400
    if 'input' not in payload:
       return jsonify({"error": {"message": "You must provide a 'input' parameter"}}), 400
    if 'speed' not in payload:
       payload['speed'] = 1
    if 'response_format' not in payload:
       payload['response_format'] = 'mp3'
    if 'prompt_speech_16k' in payload:
        response = s3_client.get_object(
            Bucket=os.getenv('S3_BUCKET'),
            Key=f"{token}_{payload['prompt_speech_16k']}"
        )
        payload['prompt_speech_16k'] = io.BytesIO(response['Body'].read())

    # response to client
    return Response(stream_with_context(generate_audio_stream(payload)), mimetype=f"audio/{payload['response_format']}")

if __name__ == '__main__':
    app.run(debug=True, threaded=False, host='0.0.0.0')
