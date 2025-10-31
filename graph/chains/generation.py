# from langchainhub import Client as hub
from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
client = Client()

prompt = client.pull_prompt("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()