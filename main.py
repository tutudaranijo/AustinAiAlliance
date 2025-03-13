from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from document_processor import qa_chain
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the model API!"}

@app.post("/query", response_model=QueryResponse)
async def get_response(request: QueryRequest):
    try:
        # Directly invoke the RAG chain for the query
        response = qa_chain.invoke(request.query)
        return QueryResponse(response=response["result"])
    except Exception as e:
        logging.error("Error during query processing", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
