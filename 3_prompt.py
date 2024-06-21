from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import config as cfg

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

# prompt = ChatPromptTemplate.from_template("Tell me the joke about {subject}")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","Give me the 10 dishes which can be prepared by below item,give in comma seperated"),
        ("human","{input}")
    ]
)

chain = prompt | llm

response = chain.invoke({"input":"snake"}).content
print(response)