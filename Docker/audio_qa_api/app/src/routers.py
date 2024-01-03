from fastapi import APIRouter, Query, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from starlette.responses import RedirectResponse
import moviepy.editor as mp

from pathlib import Path
import os, shutil


import re

from .dependencies import(
    save_upload_file,
    save_txt_in_cognitive_search_vectorial_db,
    answer_question
)

from .settings import input_formats, video_formats, model_whisper

from .models import (
    UserMessageModel,
    AIResponseModel,
    ErrorMessageModel
)


router = APIRouter()

@router.get("/")
def root():
    return RedirectResponse(url="/docs/")


@router.post("/load_media", response_description="Preprocesa y almacena el audio o video que se le indique para sar usado posteriormente en la resolucion de preguntas")
async def load_media(
        user_id: str,
        session_id: str,
        file: UploadFile = File(...)
    ):


    extencion = re.findall(r"[^.]+$", file.filename)[0]

    if not extencion in input_formats:

        return {"error" : 'Archivo invalido!'}
    

    save_upload_file(file, Path(file.filename))

    if extencion in video_formats:

        my_clip = mp.VideoFileClip(file.filename)

        audio_location = f"{file.filename}.mp3"
        my_clip.audio.write_audiofile(audio_location)
        os.remove(file.filename)

    else:
        
        audio_location = file.filename

    
    result = model_whisper.transcribe(audio_location)

    os.remove(audio_location)

    # Almacenamiento del resultado del transcript en redis

    text_media = result["text"]

    redis_index_name = f"{user_id}_{session_id}_media"

    temp_txt_route = audio_location + ".doc"
    text_file = open(temp_txt_route, "w")
    text_file.write(text_media)
    text_file.close()

    response = save_txt_in_cognitive_search_vectorial_db(temp_txt_route, redis_index_name)
    
    os.remove(temp_txt_route)

    return response


@router.post("/get_response", response_description="Genera respuesta del agente en base al mensaje y la sesion de entrada")
async def get_response(input: UserMessageModel) -> AIResponseModel:

    redis_index_name = f"{input.user_id}_{input.session_id}_media"

    response = answer_question(input.user_message, redis_index_name)

    return AIResponseModel(message=response)