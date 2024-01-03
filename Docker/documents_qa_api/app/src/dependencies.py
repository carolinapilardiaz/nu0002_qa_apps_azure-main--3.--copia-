import json
import hashlib
import os, shutil
import re
import time

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.chains import RetrievalQA

from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes

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
    llm,
    cv_client
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

def split_pdf_file(pdf_file_route, chunk_size = 1024, chunk_overlap = 200):

    loader = PyPDFLoader(pdf_file_route)

    text_splitter = RecursiveCharacterTextSplitter(        
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        length_function = len,
    )

    pages = loader.load_and_split(text_splitter)

    return pages


def split_txt_file(txt_file_route, chunk_size = 1024, chunk_overlap = 200):

    loader = TextLoader(txt_file_route)

    text_splitter = RecursiveCharacterTextSplitter(        
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        length_function = len,
    )

    pages = loader.load_and_split(text_splitter)

    return pages


def get_text_from_image_file(image_file_route, language="es"):
    
    image_data = open(image_file_route.name, "rb")    
    
    response = cv_client.read_in_stream(image_data, language=language, raw=True)
    
    operationLocation = response.headers["Operation-Location"]
    operationID = operationLocation.split("/")[-1]
    
    output = ''
    
    while True:
        # get the result
        results = cv_client.get_read_result(operationID)

        # check if the status is not "not started" or "running", if so,
        # stop the polling operation
        if results.status.lower() not in ["notstarted", "running"]:
            break

        # sleep for a bit before we make another request to the API
        time.sleep(2)
    
    if results.status == OperationStatusCodes.succeeded:
        
        read_results = results.analyze_result.read_results
        
        for result in read_results:
            for line in result.lines:
                output = output + line.text + " "
    
        return output
    
    else:

        return None

def add_data_to_document_metadata(documents, field_name, field_value):
    
    for document in documents:
        
        document.metadata[field_name] = field_value
        
    return documents

###########################################################################
# Dependencias externas

def save_pdf_in_cognitive_search_vectorial_db(pdf_file_route, index_name):

    try:

        pdf_documents_preprocessed = split_pdf_file(pdf_file_route)

        pdf_documents_preprocessed = add_data_to_document_metadata(pdf_documents_preprocessed, 'session', index_name)

        vector_store.add_documents(documents=pdf_documents_preprocessed)
        
        return AIResponseModel(message='Documento guardado con exito!')

    except Exception as e:
        
        return ErrorMessageModel(error='Error al procesar documento', description=e)
    

def save_image_ocr_in_cognitive_search_vectorial_db(image_file_route, index_name):

    image_data = open(image_file_route, "rb")

    text_ocr = get_text_from_image_file(image_data)

    temp_txt_route = image_file_route + ".doc"
    text_file = open(temp_txt_route, "w")
    text_file.write(text_ocr)
    text_file.close()

    try:

        text_documents_preprocessed = split_txt_file(temp_txt_route)

        text_documents_preprocessed = add_data_to_document_metadata(text_documents_preprocessed, 'session', index_name)

        vector_store.add_documents(documents=text_documents_preprocessed)

        return AIResponseModel(message='Documento guardado con exito!')

    except Exception as e:

        return ErrorMessageModel(error='Error al procesar documento', description=e)

    finally:
        os.remove(temp_txt_route)



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



