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

# Convert the vector store to a retriever for the QA chain
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

    Args:
        folder_id (str): The ID of the parent Google Drive folder.

    Returns:
        List[dict]: A list of file metadata dictionaries.
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
                        # If the file is a folder, add its ID to the list to process
                        folders_to_process.append(file['id'])
                        logging.info(f"Found subfolder: {file['name']} (ID: {file['id']})")
                    else:
                        # If the file is not a folder, add it to the list of all files
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

    Args:
        file_id (str): The ID of the file to download.
        file_name (str): The name of the file.
        mime_type (str): The MIME type of the file.

    Returns:
        Optional[str]: The local path to the downloaded file, or None if download failed or was skipped.
    """
    # Define export MIME types for Google Docs Editors files
    EXPORT_MIME_TYPES = {
        'application/vnd.google-apps.document': 'application/pdf',  # Or DOCX
        'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # Or CSV, XLSX
        'application/vnd.google-apps.presentation': 'application/pdf',  # Or PPTX
        'application/vnd.google-apps.form': 'application/pdf',  # Export Google Forms as PDF
        # Add more mappings if needed
    }

    request = None
    file_path = None

    if mime_type in EXPORT_MIME_TYPES:
        # It's a Google Docs Editors file; use export
        export_mime = EXPORT_MIME_TYPES[mime_type]
        try:
            request = drive_service.files().export_media(fileId=file_id, mimeType=export_mime)
            # Modify the file name's extension based on export_mime
            extension = export_mime.split('/')[-1]
            # Map MIME type to file extension if needed
            EXTENSION_MAP = {
                'application/pdf': 'pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
                # Add more mappings if needed
            }
            file_extension = EXTENSION_MAP.get(export_mime, 'pdf')  # Default to PDF if unknown
            file_name = f"{os.path.splitext(file_name)[0]}.{file_extension}"
            file_path = os.path.join('downloads', file_name)
        except Exception as e:
            logging.error(f"Failed to prepare export for {file_name}: {e}")
            return None
    elif mime_type.startswith('application/vnd.google-apps'):
        # Unsupported Google Apps file type; skip it
        logging.warning(f"Skipping unsupported Google Apps file: {file_name} (MIME Type: {mime_type})")
        return None
    else:
        # It's a binary file; use get_media
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

    Args:
        local_paths (List[str]): List of local file paths.

    Returns:
        Tuple[List[Any], List[str]]: A tuple containing the list of documents and their corresponding UUIDs.
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

    # Assign UUIDs
    uuids = [str(uuid4()) for _ in range(len(docs))]
    return docs, uuids

def create_embeddings_and_store(docs: List[Any], uuids: List[str]):
    """
    Creates embeddings for documents and stores them in the Milvus vector store.

    Args:
        docs (List[Any]): List of document objects.
        uuids (List[str]): List of UUIDs corresponding to each document.
    """
    batch_size = 99
    total_docs = len(docs)
    logging.info(f"Number of docs: {total_docs}, batch size: {batch_size}")

    try:
        for i in range(0, total_docs, batch_size):
            batch = docs[i : i + batch_size]
            batch_ids = uuids[i : i + batch_size]
            actual_batch_size = len(batch)
            logging.info(f"Storing batch {i} to {i + actual_batch_size}")

            # Ensure all documents have the required metadata
            for doc in batch:
                if 'page_number' not in doc.metadata:
                    doc.metadata['page_number'] = 1  # Assign a default value or compute appropriately

                # Truncate 'orig_elements' if it exists and exceeds max length
                if 'orig_elements' in doc.metadata:
                    orig_elements = doc.metadata['orig_elements']
                    if isinstance(orig_elements, str) and len(orig_elements) > 65535:
                        logging.warning(f"Truncating 'orig_elements' for document ID {doc.id} as it exceeds 65535 characters.")
                        doc.metadata['orig_elements'] = orig_elements[:65535]

            # Add documents to Milvus (embedding computation handled internally)
            try:
                vector_store.add_documents(batch, ids=batch_ids)
                logging.info("Stored current batch in Milvus")
            except Exception as insert_err:
                logging.error(f"Failed to insert batch starting at entity {i}: {insert_err}")
                continue  # Skip to the next batch

        logging.info("Embeddings stored successfully in Milvus.")
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

    # Use the recursive file listing function
    files = list_all_files(folder_id)
    logging.info(f"Total files found: {len(files)}")

    # 1. Download all files from the folder and subfolders
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

    # 2. Load and chunk documents
    try:
        docs, uuids = load_docs_unstructured(local_paths)
        logging.info(f"Loaded {len(docs)} docs from UnstructuredLoader.")
    except Exception as e:
        logging.error(f"Failed to load and chunk documents: {e}")
        exit(1)

    if not docs:
        logging.warning("No documents were loaded. Exiting.")
        exit(0)

    # 3. Create embeddings and store them in Milvus
    create_embeddings_and_store(docs, uuids)

    ## 4. Query your chain (Optional)
    # Uncomment and modify the following lines as needed
    # question = "What is this document about?"
    # try:
    #     response = qa_chain.invoke(question)
    #     logging.info(f"Response: {response}")
    # except Exception as e:
    #     logging.error(f"Failed to get response from QA chain: {e}")

    ## 5. (Optional) Log the run with LangSmith Client
    # Uncomment and modify the following lines as needed
    # try:
    #     client.create_run(
    #         run_type="qa",  # Added 'run_type' argument
    #         name="my_document_query_run",
    #         inputs={"question": question},
    #         outputs={"answer": response}
    #     )
    # except Exception as e:
    #     logging.error(f"Failed to log run with LangSmith Client: {e}")
