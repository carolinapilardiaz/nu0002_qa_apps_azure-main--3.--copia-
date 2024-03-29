import os
import openai

from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
import graphsignal

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

from langchain.vectorstores.azuresearch import AzureSearch


from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    ScoringProfile,
    TextWeights,
)

###########################################################################
# Variables y parametros de operacion

data_formats = ['pdf', 'jpeg', 'jpg', 'png']

vector_store_address: str = "https://cognitive-searchdemo.search.windows.net"
vector_store_password: str = "Ndqr05h9fitPrfEh9GxNBb2vsvelw9oUgIew1VpUvYAzSeD2n9Zi"

index_name: str = "document-session-index"

SUBSCRIPTION_KEY = "11c0ce07c68d4bec86d72364155c0995"
ENDPOINT_URL = "https://computer-vision-1.cognitiveservices.azure.com/"

###########################################################################
# Modelos y clases

graphsignal.configure(
  api_key='272f74fdd9dd4e7284802acaf8d295d5', 
  deployment='documents_qa_app_azure')



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

embeddings = OpenAIEmbeddings(deployment_id="text-embedding-ada-002", chunk_size=1)

# Configuracion del modelo de chat gpt

llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo", 
    temperature=0, 
    openai_api_version=api_version)

# Configuracion del cliente de OCR de Azure

cv_client = ComputerVisionClient(ENDPOINT_URL, CognitiveServicesCredentials(SUBSCRIPTION_KEY))


# Configuracion del cliente de Cognitivive Search 

fields = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SearchableField(
        name="content",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=len(embeddings.embed_query('test de emmbeding')),
        vector_search_configuration="default",
    ),
    SearchableField(
        name="metadata",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    # Additional field to store the session
    SearchableField(
        name="session",
        type=SearchFieldDataType.String,
        filterable=True,
    )
]

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    fields=fields
)