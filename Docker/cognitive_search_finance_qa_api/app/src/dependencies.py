import json
import hashlib
import os, shutil
import re
import time
from typing import List
from src.models import  AIResponseModel,ErrorMessageModel
from src.settings import llm,retriever_cognitive_search,PROMPT
from langchain.schema import BaseRetriever, Document
from langchain.chains import RetrievalQA

###########################################################################
# Modelos y clases

from .models import (
    AIResponseModel,
    ErrorMessageModel
)

from .settings import (
    llm,
    retriever_cognitive_search,
    PROMPT
)

###########################################################################
# Dependencias internas

class CustomRetrieverFinanceData(BaseRetriever):
    
    def __init__(self, retriever_cognitive_search, k=10):
        
        self.retriever_cognitive_search = retriever_cognitive_search
        self.k = k
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        
        documents =  self.retriever_cognitive_search.get_relevant_documents(query)
        
        if len(documents) > self.k:
            documents = documents[:self.k]
        
        for document in documents:
            
            document.page_content = self.add_sources_to_document(document)
        
        return documents
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
        
    def add_sources_to_document(self, input_document):
        
        page_content_with_sources = f"Usuario_id: {input_document.metadata['usuario_id']}\nSaldo_actual: {input_document.metadata['saldo_actual']}\nIngresos_mensuales: {input_document.metadata['ingresos_mensuales']}\nGastos_mensuales: {input_document.metadata['gastos_mensuales']}\nDeuda_total: {input_document.metadata['deuda_total']}\nDias_ultima_actualizacion: {input_document.metadata['dias_ultima_actualizacion']}\nCategories: {input_document.metadata['historial_crediticio']}\nDescription: {input_document.page_content}"
        
        return page_content_with_sources


###########################################################################
# Dependencias externas

def answer_question(question, k=10, chain_type = "stuff") -> str:

    try:

        custom_retriever = CustomRetrieverFinanceData(retriever_cognitive_search, k)

        chain_type_kwargs = {"prompt": PROMPT}

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=custom_retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs)

        result = qa({"query": question})

        response_agent = result['result']

        bot_message = response_agent

    except:

        bot_message = 'Lo siento ha ocurrido un error, escribe tu mensaje de nuevo por favor, o asegurate de haber cargado un documento...'


    return bot_message



