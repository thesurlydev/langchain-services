import json
import os
from datetime import datetime, timedelta
from typing import Type

import openai
import tiktoken
import yfinance as yf
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain import LLMMathChain, SerpAPIWrapper, OpenAI
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.chains import AnalyzeDocumentChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
from peopledatalabs import PDLPY
from pydantic import BaseModel, Field

# See https://github.com/peopledatalabs/peopledatalabs-python
# from peopledatalabs import PDLPY

app = FastAPI()

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )

exclude_from_auth_paths = ["/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"]


@app.middleware("http")
async def authenticate_requests(request: Request, call_next):
    if request.url.path in exclude_from_auth_paths:
        response = await call_next(request)
    else:
        api_key = request.query_params.get("api_key")

        if api_key is None:
            return JSONResponse(status_code=400, content="Missing api_key query parameter")

        langchain_services_api_key = os.environ.get("LANGCHAIN_SERVICES_API_KEY")
        if langchain_services_api_key is None:
            return JSONResponse(status_code=500, content="Missing LANGCHAIN_SERVICES_API_KEY")

        if api_key != langchain_services_api_key:
            return JSONResponse(status_code=401, content="Invalid API key")
        response = await call_next(request)
    return response


class JobDescription(BaseModel):
    description: str


class Question(BaseModel):
    query: str


class Answer(BaseModel):
    query: str
    answer: str


@app.get('/health')
def health():
    return {
        "status": "ok"
    }


@app.get('/')
def read_root():
    return {
        "message": "Hello from langchain-services",
        "help": {
            "docs": "http://localhost:8000/docs",
            "redoc": "http://localhost:8000/redoc"
        }
    }


class CurrentStockPriceInput(BaseModel):
    """Inputs for get_current_stock_price"""
    ticker: str = Field(description="Ticker symbol of the stock")


def get_current_stock_price(ticker):
    """Method to get current stock price"""

    ticker_data = yf.Ticker(ticker)
    recent = ticker_data.history(period='1d')
    return {
        'price': recent.iloc[0]['Close'],
        'currency': ticker_data.info['currency']
    }


def get_stock_performance(ticker, days):
    """Method to get stock price change in percentage"""

    past_date = datetime.today() - timedelta(days=days)
    ticker_data = yf.Ticker(ticker)
    history = ticker_data.history(start=past_date)
    old_price = history.iloc[0]['Close']
    current_price = history.iloc[-1]['Close']
    return {
        'percent_change': ((current_price - old_price) / old_price) * 100
    }


class CurrentStockPriceTool(BaseTool):
    name = "get_current_stock_price"
    description = """
        Useful when you want to get current stock price.
        You should enter the stock ticker symbol recognized by the yahoo finance
        """
    args_schema: Type[BaseModel] = CurrentStockPriceInput

    def _run(self, ticker: str):
        price_response = get_current_stock_price(ticker)
        return price_response

    def _arun(self, ticker: str):
        raise NotImplementedError("get_current_stock_price does not support async")


class StockPercentChangeInput(BaseModel):
    """Inputs for get_stock_performance"""
    ticker: str = Field(description="Ticker symbol of the stock")
    days: int = Field(description='Timedelta days to get past date from current date')


class StockPerformanceTool(BaseTool):
    name = "get_stock_performance"
    description = """
        Useful when you want to check performance of the stock.
        You should enter the stock ticker symbol recognized by the yahoo finance.
        You should enter days as number of days from today from which performance needs to be check.
        output will be the change in the stock price represented as a percentage.
        """
    args_schema: Type[BaseModel] = StockPercentChangeInput

    def _run(self, ticker: str, days: int):
        response = get_stock_performance(ticker, days)
        return response

    def _arun(self, ticker: str):
        raise NotImplementedError("get_stock_performance does not support async")


@app.get('/transcribe')
def transcribe():
    audio_file = open("/home/shane/projects/transcribe-rust/testing-123.wav", "rb")
    transcript = openai.Audio.transcribe(model="whisper-1", file=audio_file, response_format="text")
    return transcript


@app.post('/agent')
def agent(question: Question):
    print(question)
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

    tools = [
        CurrentStockPriceTool(),
        StockPerformanceTool(),
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events and news such as the current "
                        "street address of a company or recent news about a company. You should ask targeted questions"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        )
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
    query = question.query
    output = agent.run(query)
    print(output)
    answer = Answer(query=query, answer=output)
    return answer


@app.post('/jobs/parse')
def job_parse(job_description: JobDescription):
    prompt = """
        Given a job description parse it for specific information such as: 
        company name, 
        technologies, 
        list of qualifications, 
        list of things about the role, 
        salary range, 
        list of benefits,
        list of requirements, 
        a paragraph about the team,
        a paragraph about what the company does.
        Return the parsed information in valid json format with snake casing. nothing else. If any data is not 
        available, then return an empty string or array. Here is the job description {0}
        """
    query = prompt.format(job_description.description)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    message = [HumanMessage(content=query)]
    output = chat(message)
    out = output.content
    return json.loads(out)


@app.post('/questions')
def answer_query(question: Question):
    query = question.query
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    message = [HumanMessage(content=query)]
    output = chat(message)
    answer = Answer(query=query, answer=output.content)
    return answer


@app.get('/companies/{name}/logo')
def company_info(name: str):
    prompt = """
    Given the name of a company return a url for a high resolution version of the company logo. The url returned must return an actual image. The logo should be in png or svg format.
    Return only the url for the company logo and the dimensions of the image in json format with snake casing. nothing else. The name of the company is {0}
    """
    query = prompt.format(name)
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    message = [HumanMessage(content=query)]
    output = chat(message)
    out = output.content
    return json.loads(out)


@app.get('/companies/{name}/focus')
def company_info(name: str):
    prompt = """
    I am a software engineer and writing a cover letter about why I want to work at a technology company.  
    Given the name of the company, I want you to provide no more than two paragraphs specific to the company's focus 
    that resonates with my professional ethos and makes me excited about the potential of working at the company. 
    The paragraphs should be in the first person and addressed to the company.
    Return only the company name and focus paragraphs in json format with the field names being company_name and focus with snake casing. nothing else.
    The name of the company is {0}.
    """
    query = prompt.format(name)
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    chat = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
    message = [HumanMessage(content=query)]
    output = chat(message)
    out = output.content
    return json.loads(out)


@app.get('/companies/{name}/address')
def company_info(name: str):
    prompt = """
    Given the name of a company return the company name, website, and full mailing address. Return the address with separate fields for street, city, state or province, postal code and country.
    Return only the company name, website, and full mailing address in json format with snake casing. nothing else. The name of the company is {0}
    """
    query = prompt.format(name)
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    message = [HumanMessage(content=query)]
    output = chat(message)
    out = output.content
    return json.loads(out)


@app.get('/companies/{name}')
def company_info(name: str):
    prompt = """
    Given the name of a company return the company name, github organization url, ratings from indeed.com and glassdoor.com, top revenue producing products, social media urls, estimated annual revenue in millions of dollars, associated companies and relationship and website, company description, sector, industry, the year it started, ticker symbol, headquarters address, coordinates of headquarters location, exchange,
    number of employees, the url on linkedin.com, glassdoor.com and indeed.com and teamblind.com, website, careers url, and top five competitors including website.
    Return only the company name, github organization url, ratings from indeed.com and glassdoor.com, top revenue producing products, social media urls, estimated annual revenue, associated companies and relationship and website, company description, sector, industry, the year it started, ticker symbol, exchange, headquarters address, coordinates of headquarters location, number of employees, urls,
    website, careers url, and top five competitors including website in json format with snake casing. nothing else. the name of the company is {0}
    """
    query = prompt.format(name)
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    message = [HumanMessage(content=query)]
    output = chat(message)
    out = output.content
    return json.loads(out)


@app.get('/companies/{name}/github')
def company_info(name: str):
    prompt = """
    Given the name of a company return the company name and github organization url.
    Return only the company name and github organization url in json format with snake casing. nothing else. the name of the company is {0}
    """
    query = prompt.format(name)
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    message = [HumanMessage(content=query)]
    output = chat(message)
    out = output.content
    return json.loads(out)


# Take string input and return number of tokens
# Usage: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens(text_data: str) -> int:
    """Returns the number of tokens in a text string."""
    # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text_data))
    return num_tokens


@app.post('/tokens')
# Usage using pipe to curl:
# echo "foo" | curl -X POST -H "Content-Type: text/plain" --data-binary @- http://127.0.0.1:8000/tokens
async def tokens(request: Request) -> int:
    """Returns the number of tokens in a text string."""
    text_data = await request.body()
    text_data = text_data.decode('utf-8')
    return num_tokens(text_data)


class JobDescriptionData(BaseModel):
    company_name: str
    # Add other fields as needed


@app.post("/jd/")
async def jd(request: Request):
    text_data = await request.body()
    text_data = text_data.decode('utf-8')
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0, model="gpt-3.5-turbo")

    parser = PydanticOutputParser(pydantic_object=JobDescriptionData)

    doc_chain = AnalyzeDocumentChain(parser=parser)
    output = doc_chain.run(input_document=text_data)
    return output


@app.get('/jd/parse')
def parse_jd(job_description: JobDescription):
    prompt = """
        Given the content of a job description, parse the job title, company name and salary range. 
        Return only the company name, job title and salary range in json format with snake casing. nothing else.
        Here's the content of the job description: {0}.
        """
    query = prompt.format(job_description.description)
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    message = [HumanMessage(content=query)]
    output = chat(message)
    out = output.content
    return json.loads(out)


@app.get('/pdl/companies/{name}')
def people_data_labs_company(name: str):
    pdl_key = os.environ.get("PEOPLE_DATA_LABS_API_KEY")

    # Create a client, specifying your API key
    CLIENT = PDLPY(
        api_key=pdl_key
    )

    # Create a parameters JSON object
    QUERY_STRING = {
        "name": name,
        "titlecase": "true"
    }

    # Pass the parameters object to the Company Enrichment API
    response = CLIENT.company.enrichment(**QUERY_STRING)

    # Print the API response
    # print(response.text)
    return json.loads(response.text)
