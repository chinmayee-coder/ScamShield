from fastapi import FastAPI
from pydantic import BaseModel

from predict import predict_message

app = FastAPI()


class Message(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "ScamShield API is running!"
    }


@app.post("/predict")
def predict(message: Message):

    result = predict_message(message.text)

    return result