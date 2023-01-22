# Reference: https://fastapi.tiangolo.com/ko/tutorial/

from fastapi import FastAPI, Query
from enum import Enum

app = FastAPI()


# 경로: `"/"`
# Operation: `get`
# GET 동작을 사용하여 URL `"/"`에 대한 요청을 받을 때마다 `FastAPI`에 의해 호출됩니다.
@app.get("/")
async def root():
    # 가능한 type: dict, liast, str, int, ...
    return {"message": "Hello World"}


# `uvicorn main:app --reload`
    # `main``: 파일 "main.py".
    # `app``: "main.py" 내부의 `app = FastAPI()`` 줄에서 생성한 오브젝트.
    # `--reload``: 코드 변경 후 서버 재시작. 개발에만 사용.
# 자동 대화형 API 문서: http://127.0.0.1:8000/docs
# http://127.0.0.1:8000/redoc
# http://127.0.0.1:8000/openapi.json
# Operation: HTTP method 중 하나 (POST, GET, PUT, DELETE)

# 경로 매개변수 `item_id`의 값은 함수의 `item_id` 인자로 전달됩니다.
# 파이썬 표준 타입 어노테이션을 사용하여 함수에 있는 경로 매개변수의 타입을 선언할 수 있습니다:
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}


@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}


@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}


fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]
@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip: skip + limit]

# When you need to send data from a client (let's say, a browser) to your API, you send it as a request body.
# A request body is data sent by the client to your API. A response body is the data your API sends to the client.
# Your API almost always has to send a response body. But clients don't necessarily need to send request bodies all the time.

from typing import List, Union
from pydantic import BaseModel


class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None


app = FastAPI()


# @app.post("/items/")
# async def create_item(item: Item):
#     item_dict = item.dict()
#     if item.tax:
#         price_with_tax = item.price + item.tax
#         item_dict.update({"price_with_tax": price_with_tax})
#     return item_dict


@app.put("/items/{item_id}")
async def create_item(item_id: int, item: Item):
    return {"item_id": item_id, **item.dict()}

# The function parameters will be recognized as follows:
    # If the parameter is also declared in the path, it will be used as a path parameter.
    # If the parameter is of a singular type (like `int`, `float`, `str`, `bool`, etc) it will be interpreted as a query parameter.
    # If the parameter is declared to be of the type of a Pydantic model, it will be interpreted as a request body.

@app.get("/items/")
async def read_items(q: Union[List[str], None] = Query(default=None)):
    query_items = {"q": q}
    return query_items

