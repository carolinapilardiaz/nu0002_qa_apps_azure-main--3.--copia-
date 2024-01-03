from fastapi import APIRouter, Query, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
from starlette.responses import RedirectResponse

import os, shutil

import re

from .settings import (
    data_formats
)

from .models import (
    UserMessageModel,
    AIResponseModel,
    ErrorMessageModel
)

from .dependencies import (
    save_pdf_in_cognitive_search_vectorial_db,
    save_image_ocr_in_cognitive_search_vectorial_db,
    answer_question
)


router = APIRouter()

@router.get("/")
def root():
    return RedirectResponse(url="/docs/")

@router.post("/load_document", response_description="Preprocesa y almacena el documento que se le indique para sar usado posteriormente en la resolucion de preguntas")
async def load_document(
        user_id: str,
        session_id: str,
        file: UploadFile = File(...)
    ):

    extencion = re.findall(r"[^.]+$", file.filename)[0]

    if not extencion in data_formats:

        return ErrorMessageModel(error='Archivo invalido')
    
    with open(f'{file.filename}', "wb") as buffer:

        shutil.copyfileobj(file.file, buffer)
    

    redis_index_name = f"{user_id}_{session_id}_documents"


    if extencion == 'pdf':

        response = save_pdf_in_cognitive_search_vectorial_db(file.filename, redis_index_name)

    else:

        response = save_image_ocr_in_cognitive_search_vectorial_db(file.filename, redis_index_name)


    os.remove(file.filename)

    return response

@router.post("/get_response", response_description="Genera respuesta del agente en base al mensaje y la sesion de entrada")
async def get_response(input: UserMessageModel) -> AIResponseModel:

    redis_index_name = f"{input.user_id}_{input.session_id}_documents"

    response = answer_question(input.user_message, redis_index_name)

    return AIResponseModel(message=response)