import json
import hashlib
import os, shutil
import numpy as np
import PIL
from fastapi import FastAPI, File, UploadFile
from pathlib import Path
import moviepy.editor as mp
import string
import random

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.chains import RetrievalQA

from langchain.schema import BaseRetriever, Document
from typing import List

###########################################################################
# Modelos y clases

from .models import (
    AIResponseModel,
    ErrorMessageModel
)

from .settings import (
    vector_store,
    embeddings,
    llm
)

class CustomRetriever_AzureCognitiveSearch_SessionFilter(BaseRetriever):
    
    AzureCognitiveSearch_vectordb: AzureSearch
    session_filter: str
    k: int
    
    def __init__(self, AzureCognitiveSearch_vectordb, session_filter, k):
        
        super().__init__(AzureCognitiveSearch_vectordb=AzureCognitiveSearch_vectordb, session_filter=session_filter, k=k)
        
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        
        documents =  self.AzureCognitiveSearch_vectordb.similarity_search(query=query, k=self.k, search_type="similarity", filters=f"session eq '{self.session_filter}'")
        
        return documents
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError
   

###########################################################################
# Dependencias internas

def split_txt_file(txt_file_route, chunk_size = 1024, chunk_overlap = 200):

    loader = TextLoader(txt_file_route)

    text_splitter = RecursiveCharacterTextSplitter(        
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        length_function = len,
    )

    pages = loader.load_and_split(text_splitter)

    return pages

def add_data_to_document_metadata(documents, field_name, field_value):
    
    for document in documents:
        
        document.metadata[field_name] = field_value
        
    return documents

###########################################################################
# Dependencias externas

def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def cleanup(temp_file):
    os.remove(temp_file)



def save_txt_in_cognitive_search_vectorial_db(txt_file_route, index_name):

    try:

        txt_documents_preprocessed = split_txt_file(txt_file_route)
        txt_documents_preprocessed = add_data_to_document_metadata(txt_documents_preprocessed, 'session', index_name)


        vector_store.add_documents(documents=txt_documents_preprocessed)

        return AIResponseModel(message='Documento guardado con exito!')

    except Exception as e:
        
        return ErrorMessageModel(error='Error al procesar documento', description=e)


def answer_question(question, index_name, k=5, chain_type = "stuff") -> str:

    try:

        retriever_db = CustomRetriever_AzureCognitiveSearch_SessionFilter(vector_store, index_name, k)

        qa = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=retriever_db, return_source_documents=True)

        result = qa({"query": question})

        response_agent = result['result']

        references = "\n\nFuentes:"

        for reference in result['source_documents']:

            fuente = reference.metadata['source']

            if 'page' in reference.metadata:

                pagina = reference.metadata['page']

                references = references + f"\nDocumento: {fuente}, pag: {pagina}." 

            else:
                references = references + f"\nDocumento: {fuente}." 


        bot_message = response_agent + references

    except:

        bot_message = 'Lo siento ha ocurrido un error, escribe tu mensaje de nuevo por favor, o asegurate de haber cargado un documento pdf primero...'


    return bot_message




