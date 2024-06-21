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

prompt = ChatPromptTemplate.from_template("Tell me the joke about {subject}")

chain = prompt | llm

response = chain.invoke({"subject":"dog"}).content
print(response)