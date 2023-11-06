# langchain-services

Various [FastAPI](https://fastapi.tiangolo.com/) endpoints backed by [LangChain](https://langchain.com/).

Two endpoints are currently used by another project, [cover-letter](https://github.com/digitalsanctum/cover-letter), which is used for generating employment cover letters.

## prerequisites

* [poetry](https://python-poetry.org/)
* [uvicorn](https://www.uvicorn.org/) - an ASGI web server implementation for Python
* API keys for SERPAPI and OpenAI.

## run

Install dependencies and start the service:

```shell
poetry install
export OPENAI_API_KEY="your key here"
export SERPAPI_API_KEY="your key here"
export LANGCHAIN_SERVICES_API_KEY="your key here"
uvicorn main:app --reload
```

## test

```shell
curl -s "http://localhost:8000" -d '{"query": "whats the capital of usa"}' -H 'Content-Type: application/json' | jq
```

or using the IntelliJ HTTP Client:

```shell
ijhttp -L VERBOSE test.http
```

### IntelliJ HTTP Client install 

```shell
curl -f -L -o ijhttp.zip "https://jb.gg/ijhttp/latest"
unzip ijhttp.zip
```

Then place in your `PATH`