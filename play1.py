import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage
)

openai_api_key = os.environ.get("OPENAI_API_KEY")

chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
message = [HumanMessage(content="What is the name of the most populous state in the USA?")]
out = chat(message)
print(out.content)
