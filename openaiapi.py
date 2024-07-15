# Set inference model
# export MODEL_DIR=pretrained_models/CosyVoice-300M-Instruct
# For development
# fastapi dev --port 6006 fastapi_server.py
# For production deployment
# fastapi run --port 6006 fastapi_server.py

import os
import sys
import io,time
from fastapi import FastAPI, Response, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware  #引入 CORS中间件模块
from contextlib import asynccontextmanager
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import numpy as np
import torch
import torchaudio
import logging
from pydantic import BaseModel
from typing import Optional
from pydub import AudioSegment

logging.getLogger('matplotlib').setLevel(logging.WARNING)

class LaunchFailed(Exception):
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = os.getenv("MODEL_DIR", "pretrained_models/CosyVoice-300M-SFT")
    if model_dir:
        logging.info("MODEL_DIR is {}", model_dir)
        app.cosyvoice = CosyVoice(model_dir)
        # sft usage
        logging.info("Avaliable speakers {}", app.cosyvoice.list_avaliable_spks())
    else:
        raise LaunchFailed("MODEL_DIR environment must set")
    yield

app = FastAPI(lifespan=lifespan)

#设置允许访问的域名
origins = ["*"]  #"*"，即为所有,也可以改为允许的特定ip。
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,  #设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  #允许跨域的headers，可以用来鉴别来源等作用。

def buildResponse1(output):
    buffer = io.BytesIO()
    torchaudio.save(buffer, output, 22050, format="wav")
    buffer.seek(0)
    return Response(content=buffer.read(-1), media_type="audio/wav")

def buildResponse(output):
    # Save the output tensor as a WAV file to a buffer
    wav_buffer = io.BytesIO()
    torchaudio.save(wav_buffer, output, 22050, format="wav")
    wav_buffer.seek(0)
    
    # Read the WAV buffer using pydub
    audio = AudioSegment.from_wav(wav_buffer)
    
    # Convert and save the audio as MP3 to another buffer
    mp3_buffer = io.BytesIO()
    audio.export(mp3_buffer, format="mp3", bitrate="16k")  # Higher bitrate for better quality
    mp3_buffer.seek(0)

    return Response(content=mp3_buffer.read(), media_type="audio/mpeg")

voice_mappings = {
    'alloy': '中文男',
    'echo': '中文女',
    'fable': '日语男',
    'onyx': '英文女',
    'adam': '英文男',
    'nova': '粤语女',
    'shimmer': '韩语女',
}

class SpeechRequest(BaseModel):
    input: str
    voice: str = 'alloy'
    model: Optional[str] = 'tts-1'
    response_format: Optional[str] = 'mp3'
    speed: Optional[float] = 1.0
    volume: Optional[int] = 10


@app.post("/v1/audio/speech")
def text_to_speech(speechRequest: SpeechRequest):
    text = speechRequest.input
    voice = voice_mappings.get(speechRequest.voice, '中文女')
    output = app.cosyvoice.inference_sft(text, voice)
    return buildResponse(output['tts_speech'])

@app.post("/api/inference/sft")
@app.get("/api/inference/sft")
async def sft(tts: str = Form(), role: str = Form()):
    start = time.process_time()
    output = app.cosyvoice.inference_sft(tts, role)
    end = time.process_time()
    logging.info("infer time is {} seconds", end-start)
    return buildResponse(output['tts_speech'])

@app.post("/api/inference/zero-shot")
async def zeroShot(tts: str = Form(), prompt: str = Form(), audio: UploadFile = File()):
    start = time.process_time()
    prompt_speech = load_wav(audio.file, 16000)
    prompt_audio = (prompt_speech.numpy() * (2**15)).astype(np.int16).tobytes()
    prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
    prompt_speech_16k = prompt_speech_16k.float() / (2**15)

    output = app.cosyvoice.inference_zero_shot(tts, prompt, prompt_speech_16k)
    end = time.process_time()
    logging.info("infer time is {} seconds", end-start)
    return buildResponse(output['tts_speech'])

@app.post("/api/inference/cross-lingual")
async def crossLingual(tts: str = Form(), audio: UploadFile = File()):
    start = time.process_time()
    prompt_speech = load_wav(audio.file, 16000)
    prompt_audio = (prompt_speech.numpy() * (2**15)).astype(np.int16).tobytes()
    prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
    prompt_speech_16k = prompt_speech_16k.float() / (2**15)

    output = app.cosyvoice.inference_cross_lingual(tts, prompt_speech_16k)
    end = time.process_time()
    logging.info("infer time is {} seconds", end-start)
    return buildResponse(output['tts_speech'])

@app.post("/api/inference/instruct")
@app.get("/api/inference/instruct")
async def instruct(tts: str = Form(), role: str = Form(), instruct: str = Form()):
    start = time.process_time()
    output = app.cosyvoice.inference_instruct(tts, role, instruct)
    end = time.process_time()
    logging.info("infer time is {} seconds", end-start)
    return buildResponse(output['tts_speech'])

@app.get("/api/roles")
async def roles():
    return {"roles": app.cosyvoice.list_avaliable_spks()}

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang=zh-cn>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            Get the supported tones from the Roles API first, then enter the tones and textual content in the TTS API for synthesis. <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """
