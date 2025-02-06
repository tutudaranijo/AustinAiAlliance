# document_processor.py

import os
import io
from uuid import uuid4
from typing import List, Optional, Any, Mapping

# Google Drive Imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# LangChain & Other Imports
from langchain_unstructured import UnstructuredLoader
from dotenv import load_dotenv

# LangSmith Import
from langsmith import Client

# Milvus Import (Updated)
from langchain_milvus import Milvus

# Embeddings (OpenAI as example)
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains import RetrievalQA

# Import your LLM factory and base classes
from llm_support import BaseLLM, LLMFactory
from langchain.llms.base import LLM
from pydantic import Field

import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("document_processor.log"),
        logging.StreamHandler()
    ]
)

##################################
# 1. Custom LLM Wrapper (Fixed)
##################################
class CustomLLMWrapper(LLM):
    """A LangChain-compatible wrapper around your custom BaseLLM classes."""

    # Declare custom_llm as a Pydantic field
    custom_llm: BaseLLM = Field(...)

    class Config:
        # Allow arbitrary types like your custom LLM
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Unique parameters identifying this LLM."""
        return {"model_name": self.custom_llm.model_name}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Run the underlying BaseLLM's generate() method."""
        return self.custom_llm.generate(prompt)

################################
# 2. Google Drive Authentication
################################
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = 'elaborate-howl-415101-c308fb4eab27.json'

try:
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    drive_service = build('drive', 'v3', credentials=credentials)
    logging.info("Successfully authenticated with Google Drive.")
except Exception as e:
    logging.error(f"Failed to authenticate with Google Drive: {e}")
    exit(1)

################################
# 3. Set Up Embedding and LLM
################################
try:
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    logging.info("Initialized OpenAI Embeddings model.")
except Exception as e:
    logging.error(f"Failed to initialize embedding model: {e}")
    exit(1)

# Define which LLM provider / model to use from environment
LLM_PROVIDER = os.getenv('LLM_PROVIDER')
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
print("LLM_PROVIDER:", repr(LLM_PROVIDER))
try:
    # Create an LLM via our factory
    my_llm = LLMFactory.create_llm(LLM_PROVIDER, LLM_MODEL_NAME)
    logging.info(f"Initialized LLM provider '{LLM_PROVIDER}' with model '{LLM_MODEL_NAME}'.")
except Exception as e:
    logging.error(f"Failed to initialize LLM: {e}")
    exit(1)

# Wrap our custom LLM in the LangChain-compatible class
langchain_compatible_llm = CustomLLMWrapper(custom_llm=my_llm)

################################
# 4. LangSmith Client
################################
try:
    client = Client(
        api_key=os.getenv("LANGSMITH_API_KEY")
    )
    logging.info("Initialized LangSmith Client.")
except Exception as e:
    logging.error(f"Failed to initialize LangSmith Client: {e}")
    # Depending on your use case, you might choose to exit or continue without LangSmith
    # exit(1)

################################
# 5. Milvus Vector Store Setup
################################
# Get Milvus URI from environment or use default (Milvus Lite local file)
URI = os.getenv("MILVUS_URI", "./milvus_example.db")
try:
    vector_store = Milvus(
        embedding_function=embedding_model,
        collection_name="aaicollect",
        connection_args={"uri": URI},
        index_params={"index_type": "FLAT", "metric_type": "L2"},  # Optional: adjust as needed
    )
    logging.info("Connected to Milvus vector store.")
except Exception as e:
    logging.error(f"Failed to connect to Milvus vector store: {e}")
    exit(1)