# langchain-play

An example of how to wrap LangChain with FastAPI.

The following will run `main.py` which wraps a simple example of [LangChain](https://langchain.com/) with [FastAPI](https://fastapi.tiangolo.com/)

## prerequisites

* [poetry](https://python-poetry.org/)
* [uvicorn](https://www.uvicorn.org/) - an ASGI web server implementation for Python

## run

Install dependencies and start the service:

```shell
poetry install
export OPENAI_API_KEY="your key here"
uvicorn main:app --reload
```

## test

```shell
curl -s "http://localhost:8000" -d '{"query": "whats the capital of usa"}' -H 'Content-Type: application/json' | jq
```


```shell
ijhttp -L VERBOSE test.http
```

The above requires the IntelliJ HTTP Client. You can install via: 

```shell
curl -f -L -o ijhttp.zip "https://jb.gg/ijhttp/latest"
unzip ijhttp.zip
```

Then place in your `PATH`