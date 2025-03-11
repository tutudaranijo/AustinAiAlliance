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

# Langfuse Callback Integration
from langfuse.callback import CallbackHandler

# Pinecone Imports (new API)
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import pinecone

# Updated LangChain imports for embeddings and vectorstores:
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains import RetrievalQA

# Import your LLM factory and base classes
from llm_support import BaseLLM, LLMFactory
from langchain.llms.base import LLM
from pydantic import Field

import logging

# Load environment variables
load_dotenv()

# Initialize Langfuse CallbackHandler using environment variables.
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_URL = os.getenv("LANGFUSE_URL")
langfuse_handler = CallbackHandler(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_URL
)
logging.info("Initialized Langfuse CallbackHandler.")

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
    custom_llm: BaseLLM = Field(...)

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "custom_llm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.custom_llm.model_name}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
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
    # Use the OpenAI embedding model "text-embedding-3-large" which returns 3072-dimensional embeddings.
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    logging.info("Initialized OpenAI Embeddings model with 'text-embedding-3-large'.")
except Exception as e:
    logging.error(f"Failed to initialize embedding model: {e}")
    exit(1)

LLM_PROVIDER = os.getenv('LLM_PROVIDER')
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
print("LLM_PROVIDER:", repr(LLM_PROVIDER))
try:
    my_llm = LLMFactory.create_llm(LLM_PROVIDER, LLM_MODEL_NAME)
    logging.info(f"Initialized LLM provider '{LLM_PROVIDER}' with model '{LLM_MODEL_NAME}'.")
except Exception as e:
    logging.error(f"Failed to initialize LLM: {e}")
    exit(1)

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
    # Optionally, continue without LangSmith

################################
# 5. Pinecone Vector Store Setup
################################
# Load Pinecone credentials and settings from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "aaiachatbot")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")

try:
    # Create a Pinecone client instance using the environment parameter.
    pc = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    logging.info("Initialized Pinecone client using environment parameter.")
except Exception as e:
    logging.error(f"Failed to initialize Pinecone: {e}")
    exit(1)

# IMPORTANT:
# Because "text-embedding-3-large" produces 3072-dimensional vectors,
# we must create the index with dimension=3072.
try:
    existing_indexes = pc.list_indexes().names()
    if INDEX_NAME in existing_indexes:
        logging.warning(f"Index '{INDEX_NAME}' already exists. Deleting it to create a new one with dimension 3072.")
        pc.delete_index(INDEX_NAME)
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,
        metric='cosine',
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_ENV
        )
    )
    logging.info(f"Created Pinecone index: {INDEX_NAME} with dimension 3072.")
except Exception as e:
    logging.error(f"Failed to create or list Pinecone indexes: {e}")
    exit(1)

# Retrieve the index instance via the Pinecone client instance.
try:
    index_instance = pc.Index(INDEX_NAME)
    vector_store = Pinecone(
        index=index_instance,
        embedding=embedding_model,  # Pass the embedding model directly.
        text_key="text"
    )
    logging.info("Connected to Pinecone vector store.")
except Exception as e:
    logging.error(f"Failed to connect to Pinecone vector store: {e}")
    exit(1)

# Convert the vector store to a retriever for the QA chain.
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

###############################
# 6. RetrievalQA Chain Setup
###############################
try:
    qa_chain = RetrievalQA.from_chain_type(
        llm=langchain_compatible_llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )
    logging.info("Initialized RetrievalQA chain.")
except Exception as e:
    logging.error(f"Failed to initialize RetrievalQA chain: {e}")
    exit(1)

###############################
# 7. Helper Functions
###############################
def list_all_files(folder_id: str) -> List[dict]:
    """
    Recursively list all files in a given Google Drive folder and its subfolders.
    """
    all_files = []
    folders_to_process = [folder_id]
    while folders_to_process:
        current_folder = folders_to_process.pop()
        query = f"'{current_folder}' in parents and trashed = false"
        try:
            page_token = None
            while True:
                response = drive_service.files().list(
                    q=query,
                    spaces='drive',
                    fields='nextPageToken, files(id, name, mimeType)',
                    pageToken=page_token
                ).execute()
                for file in response.get('files', []):
                    if file['mimeType'] == 'application/vnd.google-apps.folder':
                        folders_to_process.append(file['id'])
                        logging.info(f"Found subfolder: {file['name']} (ID: {file['id']})")
                    else:
                        all_files.append(file)
                        logging.info(f"Found file: {file['name']} (ID: {file['id']}, MIME Type: {file['mimeType']})")
                page_token = response.get('nextPageToken', None)
                if page_token is None:
                    break
        except Exception as e:
            logging.error(f"Error fetching files from folder ID {current_folder}: {e}")
    return all_files

def download_file(file_id: str, file_name: str, mime_type: str) -> Optional[str]:
    """
    Downloads a file from Google Drive, handling both binary and Docs Editors files.
    """
    EXPORT_MIME_TYPES = {
        'application/vnd.google-apps.document': 'application/pdf',
        'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.google-apps.presentation': 'application/pdf',
        'application/vnd.google-apps.form': 'application/pdf',
    }
    request = None
    file_path = None
    if mime_type in EXPORT_MIME_TYPES:
        export_mime = EXPORT_MIME_TYPES[mime_type]
        try:
            request = drive_service.files().export_media(fileId=file_id, mimeType=export_mime)
            extension = export_mime.split('/')[-1]
            EXTENSION_MAP = {
                'application/pdf': 'pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
            }
            file_extension = EXTENSION_MAP.get(export_mime, 'pdf')
            file_name = f"{os.path.splitext(file_name)[0]}.{file_extension}"
            file_path = os.path.join('downloads', file_name)
        except Exception as e:
            logging.error(f"Failed to prepare export for {file_name}: {e}")
            return None
    elif mime_type.startswith('application/vnd.google-apps'):
        logging.warning(f"Skipping unsupported Google Apps file: {file_name} (MIME Type: {mime_type})")
        return None
    else:
        request = drive_service.files().get_media(fileId=file_id)
        file_path = os.path.join('downloads', file_name)
    if request is None:
        logging.warning(f"No download request created for {file_name}. Skipping.")
        return None
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with io.FileIO(file_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    logging.info(f"Download {file_name}: {int(status.progress() * 100)}%.")
        logging.info(f"Downloaded to: {file_path}")
        return file_path
    except HttpError as http_err:
        logging.warning(f"HTTP error occurred while downloading {file_name}: {http_err}")
        return None
    except Exception as e:
        logging.error(f"Error downloading file {file_name}: {e}")
        return None

def load_docs_unstructured(local_paths: List[str]) -> (List[Any], List[str]):
    """
    Loads and chunks documents using UnstructuredLoader.
    """
    loader = UnstructuredLoader(
        file_path=local_paths,
        chunking_strategy="basic",
        api_key=os.getenv("UNSTRUCTURED_API_KEY"),
        partition_via_api=True,
    )
    try:
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents from UnstructuredLoader.")
    except Exception as e:
        logging.error(f"Failed to load documents with UnstructuredLoader: {e}")
        raise
    docs = filter_complex_metadata(documents)
    logging.info(f"Filtered to {len(docs)} documents after metadata filtering.")
    uuids = [str(uuid4()) for _ in range(len(docs))]
    return docs, uuids

def create_embeddings_and_store(docs: List[Any], uuids: List[str]):
    """
    Creates embeddings for documents and stores them in the Pinecone vector store.
    """
    MAX_METADATA_BYTES = 40000  # maximum allowed metadata size in bytes
    batch_size = 99
    total_docs = len(docs)
    logging.info(f"Number of docs: {total_docs}, batch size: {batch_size}")
    try:
        for i in range(0, total_docs, batch_size):
            batch = docs[i : i + batch_size]
            batch_ids = uuids[i : i + batch_size]
            actual_batch_size = len(batch)
            logging.info(f"Storing batch {i} to {i + actual_batch_size}")
            for doc in batch:
                if 'page_number' not in doc.metadata:
                    doc.metadata['page_number'] = 1
                if 'orig_elements' in doc.metadata:
                    orig_elements = doc.metadata['orig_elements']
                    if isinstance(orig_elements, str):
                        orig_bytes = orig_elements.encode('utf-8')
                        if len(orig_bytes) > MAX_METADATA_BYTES:
                            logging.warning(f"Truncating 'orig_elements' for document ID {doc.id} as its size is {len(orig_bytes)} bytes, exceeding {MAX_METADATA_BYTES} bytes.")
                            truncated_bytes = orig_bytes[:MAX_METADATA_BYTES]
                            truncated_str = truncated_bytes.decode('utf-8', errors='ignore')
                            doc.metadata['orig_elements'] = truncated_str
            try:
                vector_store.add_documents(batch, ids=batch_ids)
                logging.info("Stored current batch in Pinecone")
            except Exception as insert_err:
                logging.error(f"Failed to insert batch starting at entity {i}: {insert_err}")
                continue
        logging.info("Embeddings stored successfully in Pinecone.")
    except Exception as e:
        logging.error(f"An error occurred during embeddings creation and storage: {e}")

###############################
# 8. Main Script Execution
###############################
if __name__ == '__main__':
    folder_id = os.getenv("google_folder_id", "")
    if not folder_id:
        logging.error("Error: 'google_folder_id' environment variable is not set.")
        exit(1)
    local_paths = []
    files = list_all_files(folder_id)
    logging.info(f"Total files found: {len(files)}")
    for file in files:
        file_id = file['id']
        file_name = file['name']
        mime_type = file['mimeType']
        logging.info(f"Processing file: {file_name} (ID: {file_id}, MIME Type: {mime_type})")
        try:
            local_path = download_file(file_id, file_name, mime_type)
            if local_path:
                local_paths.append(local_path)
        except Exception as e:
            logging.error(f"Failed to download {file_name}: {e}")
    if not local_paths:
        logging.warning("No files were downloaded. Exiting.")
        exit(0)
    try:
        docs, uuids = load_docs_unstructured(local_paths)
        logging.info(f"Loaded {len(docs)} docs from UnstructuredLoader.")
    except Exception as e:
        logging.error(f"Failed to load and chunk documents: {e}")
        exit(1)
    if not docs:
        logging.warning("No documents were loaded. Exiting.")
        exit(0)
    create_embeddings_and_store(docs, uuids)
    # Optional: Query your chain with Langfuse callback integration.
    # Replace <user_input> with your query.
    # Uncomment the following lines to test chain invocation with callbacks:
    #
    # question = "What is this document about?"
    # try:
    #     response = qa_chain.invoke({"input": question}, config={"callbacks": [langfuse_handler]})
    #     logging.info(f"Response: {response}")
    # except Exception as e:
    #     logging.error(f"Failed to get response from QA chain: {e}")
    #
    # Optional: Log run with LangSmith Client.
    # try:
    #     client.create_run(
    #         run_type="qa",
    #         name="my_document_query_run",
    #         inputs={"question": question},
    #         outputs={"answer": response}
    #     )
    # except Exception as e:
    #     logging.error(f"Failed to log run with LangSmith Client: {e}")
