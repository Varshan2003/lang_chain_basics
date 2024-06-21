from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser,CommaSeparatedListOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
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

def getting_output_string():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system","He is pushing you to complete the project asap so that he can talk to prabhal to convert you"),
            ("human","{input}")
        ]
    )
    parser = StrOutputParser()
    chain = prompt | llm | parser
    return chain.invoke({"input":"snake"})

def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a list of 10 synonyms for the following word. Return the results as a comma seperated list."),
        ("human", "{input}")
    ])

    parser = CommaSeparatedListOutputParser()
    
    chain = prompt | llm | parser

    return chain.invoke({
        "input": "happy"
    })

def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract information from the following phrase.\nFormatting Instructions: {format_instructions}"),
        ("human", "{phrase}")
    ])

    class Person(BaseModel):
        recipe: str = Field(description="Name of the recipe")
        ingredients:str = Field(description="ingredients")

    parser = JsonOutputParser(pydantic_object=Person)
    
    chain = prompt | llm | parser

    return chain.invoke({
        "phrase": "The ingredients for a Margherita pizza are tomatoes, onions, cheese, basil",
        "format_instructions": parser.get_format_instructions()
    })


print(call_json_output_parser())