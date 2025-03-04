import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from document_processor import qa_chain
from dotenv import load_dotenv
import logging

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




logger = logging.getLogger(__name__)

@app.post("/query", response_model=QueryResponse)
async def get_response(request: QueryRequest):
    try:
        response = qa_chain.invoke(request.query)
        return QueryResponse(response=response['result'])
    except Exception as e:
        logger.error("Error occurred during query processing", exc_info=True)
        # Return a generic message to the client
        raise HTTPException(status_code=500, detail="Internal Server Error")


