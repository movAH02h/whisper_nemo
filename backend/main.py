# main.py
from fastapi import FastAPI

diploma = FastAPI()

@diploma.get("/")
async def root():
    return {"message": "Hello World"}