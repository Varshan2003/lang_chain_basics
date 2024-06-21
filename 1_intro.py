from langchain_openai import AzureChatOpenAI
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

response = llm.stream("hello, explain python coding")

for chunk in response:
    print(chunk.content,end = "", flush=True)