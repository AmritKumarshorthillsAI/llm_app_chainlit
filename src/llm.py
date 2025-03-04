from langchain_openai import AzureChatOpenAI
from src.prompt import system_instruction
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize AzureChatOpenAI LLM
azure_chat_llm = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    temperature=0  # Set temperature here
)

# Defining role and content for the LLM
messages = [
    {"role": "system", "content": system_instruction}
]

def ask_order(messages):
    try:
        # Convert messages to the expected input format
        input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        response = azure_chat_llm.invoke(input=input_text)
        return response.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

