import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

api_key = os.getenv("GENAI_API_KEY")
if not api_key:
    raise ValueError("GENAI_API_KEY not found in .env")
    
llm = ChatOpenAI(
    base_url="https://genai.rcac.purdue.edu/api",
    api_key=api_key,
    model="llama3.1:latest",
    streaming=True,
    max_tokens=100  
)

messages = [
    HumanMessage(content="What is your name?")
]

try:
    response = llm.invoke(messages)
    print("Response from LangChain:")
    print(response.content)
except Exception as e:
    print(f"Error invoking LLM: {e}")