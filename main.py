import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from document_processor import qa_chain
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Define request and response data models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

@app.get('/')
async def read_root():
    return {'message': 'Welcome to the model API!'}

@app.post("/query", response_model=QueryResponse)
async def get_response(request: QueryRequest):
    try:
        response = qa_chain.invoke(request.query)
        return QueryResponse(response=response['result'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
