"""
Style-Bert-VITS2-Editor用のサーバー。
次のリポジトリ
https://github.com/litagin02/Style-Bert-VITS2-Editor
をビルドしてできあがったファイルをWebフォルダに入れて実行する。

TODO: リファクタリングやドキュメンテーションやAPI整理、辞書周りの改善などが必要。
"""

import argparse
import io
import shutil
import sys
import webbrowser
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.io import wavfile

from config import get_path_config
from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    VERSION,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.g2p_utils import g2kata_tone, kata_tone2phone_tone
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.nlp.japanese.user_dict import (
    apply_word,
    delete_word,
    read_dict,
    rewrite_word,
    update_dict,
)
from style_bert_vits2.tts_model import TTSModelHolder, TTSModelInfo



AIP_HEALTH_ROUTE = os.environ.get('AIP_HEALTH_ROUTE', '/health')
AIP_PREDICT_ROUTE = os.environ.get('AIP_PREDICT_ROUTE', '/predict')

# ---フロントエンド部分に関する処理ここまで---
# 以降はAPIの設定

# pyopenjtalk_worker を起動
## pyopenjtalk_worker は TCP ソケットサーバーのため、ここで起動する
pyopenjtalk.initialize_worker()

# pyopenjtalk の辞書を更新
update_dict()

# 事前に BERT モデル/トークナイザーをロードしておく
## ここでロードしなくても必要になった際に自動ロードされるが、時間がかかるため事前にロードしておいた方が体験が良い
## server_editor.py は日本語にしか対応していないため、日本語の BERT モデル/トークナイザーのみロードする
bert_models.load_model(Languages.JP)
bert_models.load_tokenizer(Languages.JP)


class AudioResponse(Response):
    media_type = "audio/wav"


origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

path_config = get_path_config()
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default=path_config.assets_root)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--inbrowser", action="store_true")
parser.add_argument("--line_length", type=int, default=None)
parser.add_argument("--line_count", type=int, default=None)
# parser.add_argument("--skip_default_models", action="store_true")
parser.add_argument("--skip_static_files", action="store_true")
args = parser.parse_args()
device = args.device
if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"
model_dir = Path(args.model_dir)
port = int(args.port)
# if not args.skip_default_models:
#     download_default_models()
skip_static_files = bool(args.skip_static_files)

model_holder = TTSModelHolder(model_dir, device)
if len(model_holder.model_names) == 0:
    logger.error(f"Models not found in {model_dir}.")
    sys.exit(1)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()


class MoraTone(BaseModel):
    mora: str
    tone: int


class TextRequest(BaseModel):
    text: str


class SynthesisRequest(BaseModel):
    model: str
    modelFile: str
    text: str
    moraToneList: list[MoraTone]
    style: str = DEFAULT_STYLE
    styleWeight: float = DEFAULT_STYLE_WEIGHT
    assistText: str = ""
    assistTextWeight: float = DEFAULT_ASSIST_TEXT_WEIGHT
    speed: float = 1.0
    noise: float = DEFAULT_NOISE
    noisew: float = DEFAULT_NOISEW
    sdpRatio: float = DEFAULT_SDP_RATIO
    language: Languages = Languages.JP
    silenceAfter: float = 0.5
    pitchScale: float = 1.0
    intonationScale: float = 1.0
    speaker: Optional[str] = None

def synthesis(request: SynthesisRequest):
    if args.line_length is not None and len(request.text) > args.line_length:
        raise HTTPException(
            status_code=400,
            detail=f"1行の文字数は{args.line_length}文字以下にしてください。",
        )
    try:
        model = model_holder.get_model(
            model_name=request.model, model_path_str=request.modelFile
        )
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model {request.model} from {request.modelFile}, {e}",
        )
    text = request.text
    kata_tone_list = [
        (mora_tone.mora, mora_tone.tone) for mora_tone in request.moraToneList
    ]
    phone_tone = kata_tone2phone_tone(kata_tone_list)
    tone = [t for _, t in phone_tone]
    try:
        sid = 0 if request.speaker is None else model.spk2id[request.speaker]
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Speaker {request.speaker} not found in {model.spk2id}",
        )
    sr, audio = model.infer(
        text=text,
        language=request.language,
        sdp_ratio=request.sdpRatio,
        noise=request.noise,
        noise_w=request.noisew,
        length=1 / request.speed,
        given_tone=tone,
        style=request.style,
        style_weight=request.styleWeight,
        assist_text=request.assistText,
        assist_text_weight=request.assistTextWeight,
        use_assist_text=bool(request.assistText),
        line_split=False,
        pitch_scale=request.pitchScale,
        intonation_scale=request.intonationScale,
        speaker_id=sid,
    )
    print(sr)
    print(audio)
    with BytesIO() as wavContent:
        wavfile.write(wavContent, sr, audio)
        return Response(content=wavContent.getvalue(), media_type="audio/wav")


@app.get(AIP_HEALTH_ROUTE, status_code=200)
async def health():
    return {'health': 'ok'}

@app.post(AIP_PREDICT_ROUTE,response_model=AudioResponse,
          response_model_exclude_unset=True) 
async def predict(instances: SynthesisRequest):
    response = synthesis(instances)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
