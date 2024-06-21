from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config as cfg
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
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
        chunk_size = 200,
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
    

def create_chain():
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}  
    Question: {input}                                        
    """
    )
    vector_store = create_db(get_data_from_csv(path='/Users/vacharya/Projects/langchain/data/my_family.csv'))
    retriver = vector_store.as_retriever()
    
    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    retriver_chain = create_retrieval_chain(
        retriver,chain
    )
    response = retriver_chain.invoke({"input":"What is Varshan's age"})
    print(response['answer'])

create_chain()