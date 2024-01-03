import json
import hashlib
import os, shutil
import re
import time

import pandas as pd
from collections import Counter

from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.chains import RetrievalQA

from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes, Details

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

def analysis_image(image_path):
    # Specify features to be retrieved
    features = [VisualFeatureTypes.description,
                VisualFeatureTypes.tags,
                VisualFeatureTypes.categories,
                VisualFeatureTypes.brands,
                VisualFeatureTypes.objects,
                VisualFeatureTypes.adult,
                VisualFeatureTypes.faces,
                VisualFeatureTypes.color,
                VisualFeatureTypes.image_type]
    remote_image_details = [Details.celebrities]
    
    # Get image analysis
    with open(image_path, mode="rb") as image_data:
        analysis = cv_client.analyze_image_in_stream(image_data , features,remote_image_details)
        
    return analysis

def descripcion_imagen(analysis):
    # Get image description
    description = []
    tags = []
    categories = []
    brands = []
    objects = []
    conteo_objetos= []
    landmarks = []
    referencias = []
    for caption in analysis.description.captions:
        description.append(caption.text)
    # Get image tags
    for caption in analysis.tags:
        tags.append(caption.name)
    # Get image categories 
    for category in analysis.categories:
        categories.append(category.name)
        if category.detail:
            # Get landmarks in this category
            if category.detail.landmarks:
                for landmark in category.detail.landmarks:
                    if landmark not in landmarks:
                        landmarks.append(landmark)
    # If there were landmarks, list them
    if len(landmarks) > 0:
        for landmark in landmarks:
            referencias.append(landmark.name)  
    #Get image brands
    if len(analysis.brands) == 0:
        for caption in analysis.brands:
            brands.append(caption)
    else:
        for brand0 in analysis.brands:
            brands.append(brand0.name)
    # Get objects in the image
    for caption in analysis.objects:
        objects.append(caption.object_property)
    lista_group = Counter(objects)
    
    if len(list(lista_group)) > 0:
        for i in range(0,len(lista_group)):
            conteo_objetos.append(str(list(lista_group.values())[i])+" "+list(lista_group.keys())[i])
    # Get moderation ratings
    adult_content = analysis.adult.is_adult_content
    racy_content = analysis.adult.is_racy_content
    if adult_content == False:
        is_adult = "No adult content"
    else:
        is_adult = "is adult content"
            
    # Detect Image Types - local
    if analysis.image_type.clip_art_type == 0:
        imagen_type = "Image is not clip art."
    elif analysis.image_type.line_drawing_type == 1:
        imagen_type = "Image is ambiguously clip art."
    elif analysis.image_type.line_drawing_type == 2:
        imagen_type = "Image is normal clip art."
    else:
        imagen_type = "Image is good clip art."

    if analysis.image_type.line_drawing_type == 0:
        imagen_draw = "Image is not a line drawing."
    else:
        imagen_draw = "Image is a line drawing"

    # Print results of color scheme
    if analysis.color.is_bw_img == True:
        black_white = "is black and white image"
    else:
        black_white = "is not black and white image"
    colors = analysis.color.dominant_colors
    

    resumen = f'Description: {", ".join(description)}\nTags: {", ".join(tags)}\nCategories: {", ".join(categories)}\nbrands: {", ".join(brands)}\nNumber of people and objects in the image: {", ".join(conteo_objetos)}\nAdult content: {is_adult}\nlandmarks: {", ".join(referencias)}\nImagen clip art: {imagen_type}\nDrawing: {imagen_draw}\nBlack and white: {black_white}\nDominant Colors: {",".join(colors)}'
    
    return resumen


def get_text_description_of_image(input_image_path):
    
    analysis_input_image = analysis_image(input_image_path)
    
    output_description = descripcion_imagen(analysis_input_image)
    
    return output_description

def add_data_to_document_metadata(documents, field_name, field_value):
    
    for document in documents:
        
        document.metadata[field_name] = field_value
        
    return documents

###########################################################################
# Dependencias externas

def save_image_in_cognitive_search_vectorial_db(image_file_route, index_name):

    try:

        print(image_file_route)

        image_description = get_text_description_of_image(image_file_route)

        image_data_dict = {
            "source": image_file_route,
            "image_description": image_description
        }

        image_data_df = pd.DataFrame([image_data_dict])

        loader = DataFrameLoader(image_data_df, page_content_column="image_description")
        image_documents_preprocessed = loader.load()
        image_documents_preprocessed = add_data_to_document_metadata(image_documents_preprocessed, 'session', index_name)

        vector_store.add_documents(documents=image_documents_preprocessed)
        
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

                references = references + f"\nImagen: {fuente}, pag: {pagina}." 

            else:
                
                references = references + f"\nImagen: {fuente}." 


        bot_message = response_agent + references

    except:

        bot_message = 'Lo siento ha ocurrido un error, escribe tu mensaje de nuevo por favor, o asegurate de haber cargado un documento pdf primero...'


    return bot_message



