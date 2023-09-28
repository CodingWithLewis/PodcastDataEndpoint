import os

ASTRA_DB_SECURE_BUNDLE_PATH = os.environ["ASTRA_DB_SECURE_BUNDLE_PATH"]
ASTRA_DB_TOKEN_JSON_PATH = os.environ["ASTRA_DB_TOKEN_JSON_PATH"]
ASTRA_DB_KEYSPACE = "mrbeast"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.schema import HumanMessage

# These are used to authenticate with Astra DB
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

# Support for dataset retrieval with Hugging Face
from datasets import load_dataset

import json
from fastapi import FastAPI
from models import Query


template = """
You are a podcast helper that uses the data provided to help with an answer. Try and make the best of the context that is provided. The user doesn't know that you are gathering documents, so we please don't reference them. 

Query: {query}
Documents: {context}

"""

app = FastAPI()

cloud_config = {"secure_connect_bundle": ASTRA_DB_SECURE_BUNDLE_PATH}

with open(ASTRA_DB_TOKEN_JSON_PATH) as f:
    secrets = json.load(f)
ASTRA_DB_APPLICATION_TOKEN = secrets[
    "token"
]  # token is pulled from your token json file

auth_provider = PlainTextAuthProvider("token", ASTRA_DB_APPLICATION_TOKEN)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

system_message_prompt = SystemMessagePromptTemplate.from_template(template)

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
myEmbedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

myCassandraVStore = Cassandra(
    embedding=myEmbedding,
    session=astraSession,
    keyspace=ASTRA_DB_KEYSPACE,
    table_name="mrbeast",
)


@app.post("/")
def read_root(query: Query):
    # vectorIndex = VectorStoreIndexWrapper(vectorstore=myCassandraVStore)
    # answer = vectorIndex.query(query.query, llm=llm)
    docs = []
    for doc, score in myCassandraVStore.similarity_search_with_score(
        query=query.query, k=8
    ):
        docs.append(
            {
                "content": doc.page_content,
                "score": score,
                "timestamp": doc.metadata["start"],
            }
        )

    print(docs)
    answer = llm([system_message_prompt.format(query=query.query, context=str(docs))])
    return {
        "answer": answer.content,
        "doc": docs[0],
    }
