# document_processor.py

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
from llm_support import  BaseLLM, LLMFactory
from langchain.llms.base import LLM
from pydantic import Field

from typing import Optional, List, Any, Mapping
load_dotenv()

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

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
drive_service = build('drive', 'v3', credentials=credentials)

################################
# 3. Set Up Embedding and LLM
################################
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Define which LLM provider / model to use from environment
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

# Create an LLM via our factory
my_llm = LLMFactory.create_llm(LLM_PROVIDER, LLM_MODEL_NAME)

# Wrap our custom LLM in the LangChain-compatible class
langchain_compatible_llm = CustomLLMWrapper(custom_llm=my_llm)

################################
# 4. LangSmith Client
################################
client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY")
)
# If you set LANGSMITH_TRACING=true, runs can be automatically tracked

################################
# 5. Milvus Vector Store Setup
################################
MILVUS_HOST = os.getenv('MILVUS_HOST')
MILVUS_PORT = int(os.getenv('MILVUS_PORT'))

vector_store = Milvus(
    embedding_function=embedding_model,
    collection_name="aaicollect", 
    connection_args={
        "host": MILVUS_HOST,
        "port": MILVUS_PORT
    },
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

###############################
# 6. RetrievalQA Chain Setup
###############################
qa_chain = RetrievalQA.from_chain_type(
    llm=langchain_compatible_llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

###############################
# 7. Helper Functions
###############################
def list_files_in_folder(folder_id):
    query = f"'{folder_id}' in parents"
    try:
        results = drive_service.files().list(q=query).execute()
        items = results.get('files', [])
        if not items:
            print("No files found in the folder.")
        else:
            for item in items:
                print(f"Found file: {item['name']} (ID: {item['id']})")
        return items
    except Exception as e:
        print(f"Error fetching files: {e}")
        return []

test = list_files_in_folder(os.getenv("google_folder_id"))

def download_file(file_id, file_name):
    request = drive_service.files().get_media(fileId=file_id)
    file_path = os.path.join('downloads', file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with io.FileIO(file_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download {int(status.progress() * 100)}%.")
    return file_path

def load_docs_unstructured(local_paths):
    loader = UnstructuredLoader(
        file_path=local_paths,
        chunking_strategy="basic",
        api_key=os.getenv("UNSTRUCTURED_API_KEY"),
        partition_via_api=True,
    )
    documents = loader.load()
    docs = filter_complex_metadata(documents)
    uuids = [str(uuid4()) for _ in range(len(docs))]
    return docs, uuids

def create_embeddings_and_store(docs, uuids):
    batch_size = 99
    print(f"Number of docs: {len(docs)}, batch size: {batch_size}")

    try:
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            batch_ids = uuids[i : i + batch_size]
            print(f"Storing batch {i} to {i+len(batch)}")

            vector_store.add_documents(batch, ids=batch_ids)
            print("Stored current batch in Milvus")

        print("Embeddings stored successfully in Milvus.")
    except Exception as e:
        print(f"An error occurred: {e}")

###############################
# 8. Main Script Execution
###############################
if __name__ == '__main__':
    folder_id = os.getenv("google_folder_id", "")
    local_paths = []
    files = list_files_in_folder(folder_id)

    # 1. Download all files from the folder
    for file in files:
        file_id = file['id']
        file_name = file['name']
        print(f"Processing file: {file_name} (ID: {file_id})")

        local_path = download_file(file_id, file_name)
        local_paths.append(local_path)
        print(f"Downloaded to: {local_path}")

    # 2. Load and chunk documents
    docs, uuids = load_docs_unstructured(local_paths)
    print(f"Loaded {len(docs)} docs from UnstructuredLoader.")

    # 3. Create embeddings and store them in Milvus
    create_embeddings_and_store(docs, uuids)

    ## 4. Query your chain
    #question = "What is this document about?"
    #response = qa_chain.invoke(question)
    #print("Response:", response)#

    ## 5. (Optional) Log the run with LangSmith Client
    #client.create_run(
    #    run_type="qa",  # Added 'run_type' argument
    #    name="my_document_query_run",
    #    inputs={"question": question},
    #    outputs={"answer": response}
    #)
