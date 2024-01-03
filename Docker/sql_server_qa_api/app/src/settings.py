import os
import openai
import pyodbc
from sqlalchemy.engine.url import URL

from langchain.chains import RetrievalQA
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain import SQLDatabaseChain
from langchain.chains import SQLDatabaseSequentialChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.prompts import PromptTemplate

import graphsignal


###########################################################################
# Variables y parametros de operacion

server = 'sql-serveropenai.database.windows.net'
database = 'database-demo'
username = 'admindemo'
password = 'nubiral2023!'   
driver= '{ODBC Driver 18 for SQL Server}'

db_config = {   'drivername': 'mssql+pyodbc',
                'username': username +'@'+ server,
                'password': password,
                'host': server,
                'port': 1433,
                'database': database,
                'query': {'driver': 'ODBC Driver 18 for SQL Server'}
            }

# Create a URL object for connecting to the database
db_url = URL.create(**db_config)

db = SQLDatabase.from_uri(db_url)

###########################################################################
# Modelos y clases

graphsignal.configure(
  api_key='272f74fdd9dd4e7284802acaf8d295d5', 
  deployment='sql_qa_app_azure')

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


# Configuracion del modelo de chat gpt

llm = AzureChatOpenAI(
    deployment_name="nubiral-lab-01", 
    temperature=0, 
    openai_api_version=api_version)

# Prompt agente SQL

MSSQL_PROMPT = """
You are an MS SQL expert. Given an input question, first create a syntactically correct MS SQL query to run, then look at the results of the query and return the answer to the input question.

Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the TOP clause as per MS SQL. You can order the results to return the most informative data in the database.

Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in square brackets ([]) to denote them as delimited identifiers.

Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

**Do not use double quotes on the SQL query**. 

Your response should be in Markdown. Answer the question in spanish.

** ALWAYS before giving the Final Answer, try another method**. Then reflect on the answers of the two methods you did and ask yourself if it answers correctly the original question. If you are not sure, try another method.
If the runs does not give the same result, reflect and try again two more times until you have two runs that have the same result. If you still cannot arrive to a consistent result, say that you are not sure of the answer. But, if you are sure of the correct answer, create a beautiful and thorough response. DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE. 

ALWAYS, as part of your final answer, explain how you got to the answer on a section that starts with: \n\nExplanation:\n. Include the SQL query as part of the explanation section.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here
Explanation:

For example:
<=== Beginning of example

Question: How many people died of covid in Texas in 2020?
SQLQuery: SELECT [death] FROM covidtracking WHERE state = 'TX' AND date LIKE '2020%'
SQLResult: [(27437.0,), (27088.0,), (26762.0,), (26521.0,), (26472.0,), (26421.0,), (26408.0,)]
Answer: There were 27437 people who died of covid in Texas in 2020.


Explanation:
I queried the covidtracking table for the death column where the state is 'TX' and the date starts with '2020'. The query returned a list of tuples with the number of deaths for each day in 2020. To answer the question, I took the sum of all the deaths in the list, which is 27437. 
I used the following query

```sql
SELECT [death] FROM covidtracking WHERE state = 'TX' AND date LIKE '2020%'"
```
===> End of Example

Only use the following tables:
{table_info}

Question: {input}"""

MSSQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "top_k"], 
    template=MSSQL_PROMPT
)

db_chain = SQLDatabaseChain(llm=llm, database=db, prompt=MSSQL_PROMPT, verbose=True)