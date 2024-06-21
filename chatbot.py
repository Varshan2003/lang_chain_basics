from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config as cfg
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains.llm import LLMChain
from langchain_community.chat_message_histories.upstash_redis import UpstashRedisChatMessageHistory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory

api_key = cfg.api_key
azure_endpoint = cfg.azure_endpoint
api_version = cfg.api_version
azure_deployment = cfg.azure_deployment

llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    model="gpt-35-turbo",
    azure_deployment=azure_deployment
)
def get_data_from_csv(path=None):
    loader = CSVLoader(file_path=path)
    docA = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 400,
        chunk_overlap = 20
    )
    data = splitter.split_documents(docA)
    return data

def create_db(data):
    embeddings = AzureOpenAIEmbeddings(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            model="text-embedding-ada-002",
            azure_deployment="text-embedding-ada-002"
    )
    vector_store = Chroma.from_documents(data,embedding=embeddings,persist_directory="VECTOR_DB")
    return vector_store
    

def create_chain(question):
    history = UpstashRedisChatMessageHistory(url = cfg.redis_url, token=cfg.redis_token,ttl=400,session_id="chat1")
    memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history,
)
    prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly AI assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
    vector_store = create_db(get_data_from_csv(path='/Users/vacharya/Projects/langchain/data/my_family.csv'))
    
    chain = llm | prompt | vector_store | memory
    response = chain.invoke({"input":{question}})
    return response['answer']

def process_chat():
    while True:
        question = input("Enter your question: ")
        answer = create_chain(question)
        print(answer)

process_chat()