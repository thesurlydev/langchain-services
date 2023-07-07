import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage
)
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )


class Question(BaseModel):
    query: str


class Answer(BaseModel):
    query: str
    answer: str


@app.get('/')
def read_root():
    return {"Hello": "world"}


@app.post('/')
def answer_query(question: Question):
    try:
        query = question.query
        openai_api_key = os.environ.get("OPENAI_API_KEY")

        chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        message = [HumanMessage(content=query)]
        output = chat(message)
        answer = Answer(query=query, answer=output.content)
        return answer
    except:
        return {"message": "Error!"}
