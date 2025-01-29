import os
import io
from uuid import uuid4
from typing import List, Optional

# Google Drive Imports
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# LangChain & Other Imports
from langchain_unstructured import UnstructuredLoader
from dotenv import load_dotenv

# LangSmith Import
from langsmith import Client

# Milvus Import (Community)
from langchain_community.vectorstores import Milvus

# Embeddings (OpenAI as example)
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains import RetrievalQA

# Import your LLM factory and base classes
from llm_support import LLMFactory, BaseLLM

# LangChain's base LLM
from langchain.llms.base import LLM
from pydantic import Field

load_dotenv()

##################################
# 1. Custom LLM Wrapper (Fixed)
##################################
from langchain.llms.base import LLM
from typing import Optional, List, Any, Mapping

from llm_support import BaseLLM

class CustomLLMWrapper(LLM):
    """A LangChain-compatible wrapper around our custom BaseLLM classes."""

    def __init__(self, custom_llm: BaseLLM, **kwargs: Any):
        """Initialize with a custom_llm instance."""
        super().__init__(**kwargs)
        self.custom_llm = custom_llm

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Unique parameters identifying this LLM. Some versions of LangChain require this."""
        return {"model_name": self.custom_llm.model_name}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Run the underlying BaseLLM's generate() method."""
        return self.custom_llm.generate(prompt)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
my_llm = LLMFactory.create_llm(LLM_PROVIDER, LLM_MODEL_NAME)

# Wrap our custom LLM in the LangChain-compatible class
langchain_compatible_llm = CustomLLMWrapper(custom_llm=my_llm)