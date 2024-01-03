from fastapi import APIRouter, Query, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.encoders import jsonable_encoder
from starlette.responses import RedirectResponse

import os, shutil

import re

from .models import (
    UserMessageModel,
    AIResponseModel,
    ErrorMessageModel
)

from .dependencies import (
    answer_question
)


router = APIRouter()

@router.get("/")
def root():
    return RedirectResponse(url="/docs/")


@router.post("/get_response", response_description="Genera respuesta del agente en base al mensaje y la sesion de entrada")
async def get_response(input: UserMessageModel) -> AIResponseModel:

    response = answer_question(input.user_message)

    return AIResponseModel(message=response)