from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
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

docA = Document(page_content="""Royal Challengers Bangalore (RCB), officially Royal Challengers Bengaluru, is a professional franchise cricket team based in Bangalore, Karnataka, playing in the Indian Premier League (IPL). The franchise was founded in 2008 by United Spirits and was named after its liquor brand Royal Challenge. The M. Chinnaswamy Stadium serves as the team's home ground Before the 2024 Indian Premier League, the team was renamed Royal Challengers Bengaluru (previously Royal Challengers Bangalore).[3] The team finished as runners-up on three occasions in 2009, 2011 and 2016. The Royal Challengers are valued at $69.8 million, making them one of the most valuable IPL franchises.[4][5] As of 2024, the team is captained by South African cricketer Faf du Plessis.Royal Challengers Bangalore also hold two important IPL records, for the lowest score in an innings in IPL (49, against the Kolkata Knight Riders) and for the highest score conceded in an innings (287, against Sunrisers Hyderabad).[6]""")

# prompt = ChatPromptTemplate.from_template("Tell me the joke about {subject}")
prompt = ChatPromptTemplate.from_template("""
Answer the user's question:
Context: {context}  
Question: {input}                                        
"""
)

chain = prompt | llm

response = chain.invoke({"input":"Who is the captain of RCB","context":[docA]}).content
print(response)