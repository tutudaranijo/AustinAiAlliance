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
