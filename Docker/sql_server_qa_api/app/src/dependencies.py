import json
import hashlib
import os, shutil
import re
import time
from typing import List

from langchain.schema import BaseRetriever, Document
from langchain.chains import RetrievalQA

###########################################################################
# Modelos y clases

from .models import (
    AIResponseModel,
    ErrorMessageModel
)

from .settings import (
    db_chain
)

###########################################################################
# Dependencias internas


###########################################################################
# Dependencias externas

def answer_question(question) -> str:

    try:

        result = db_chain(question)

        response_agent = result['result']

        bot_message = response_agent

    except:

        bot_message = 'Lo siento ha ocurrido un error, escribe tu mensaje de nuevo por favor, o asegurate de haber cargado un documento pdf primero...'


    return bot_message



