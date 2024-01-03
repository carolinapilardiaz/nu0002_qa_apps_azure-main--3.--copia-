import os
import openai

from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.prompts import PromptTemplate

import graphsignal


###########################################################################
# Variables y parametros de operacion

os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"] = "cognitive-searchdemo"
os.environ["AZURE_COGNITIVE_SEARCH_INDEX_NAME"] = "cosmosdbfinance-index"
os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"] = "Ndqr05h9fitPrfEh9GxNBb2vsvelw9oUgIew1VpUvYAzSeD2n9Zi"

###########################################################################
# Modelos y clases

graphsignal.configure(
  api_key='272f74fdd9dd4e7284802acaf8d295d5', 
  deployment='cognitive_search_financw_qa_app_azure')

api_type = "azure"
api_base_url = "https://openaidemonubiral.openai.azure.com/"
api_version = "2023-03-15-preview"
azure_api_key = "ff5c606c134e4d1dae3426a412df834a"

openai.api_type = api_type
openai.api_base = api_base_url
openai.api_version = api_version
openai.api_key = azure_api_key

os.environ["OPENAI_API_BASE"] = api_base_url
os.environ["OPENAI_API_KEY"] = azure_api_key
#os.environ["LANGCHAIN_HANDLER"] = "langchain"


# Configuracion del modelo de embedding

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)

# Configuracion del modelo de chat gpt

llm = AzureChatOpenAI(
    deployment_name="nubiral-lab-01", 
    temperature=0, 
    openai_api_version=api_version)


# Retriever de cognitive search

retriever_cognitive_search = AzureCognitiveSearchRetriever(content_key="descripcion")

# Promt para el sistema de recomendacion

prompt_template = """Use the following context elements to answer the question at the end.
You are a fintech recommendation system, when asked by the user, you must recommend the clients that best meet the requirements to grant credits, in addition to that, try to provide reasons why the user should grant credits to the clients you recommend.
You can only recommend the clients provided below, if none of the clients meet the credit requirements, please indicate that you do not have something that meets the requirements. Don't try to make up an answer.


Clients:

{context}

Question: {question}
Answer in Spanish:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)