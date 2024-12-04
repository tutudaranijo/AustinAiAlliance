import os
import io
from uuid import uuid4
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from langchain_unstructured import UnstructuredLoader
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings


load_dotenv()
# Define the scope and authenticate using the service account credentials
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SERVICE_ACCOUNT_FILE = './service_account.json'

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# Build the Drive API client
drive_service = build('drive', 'v3', credentials=credentials)

watsonx_apikey = os.getenv('WATSONX_APIKEY')
watsonx_url = os.getenv('WATSONX_URL')
project_id = os.getenv('PROJECT_ID')

# Use openAI for testing until WATSONX is provided
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

embed_params = {
    "truncate_input_tokens": 3,
    "return_options": {"input_text": True},
}

# # Initialize WatsonxEmbeddings
# embedding_model = WatsonxEmbeddings(
#     model_id="ibm/slate-125m-english-rtrvr",
#     url=watsonx_url,
#     project_id=project_id,
#     params=embed_params,
# )

vector_store = Chroma(
    collection_name="test-docs",
    embedding_function=embedding_model,
    persist_directory="./chroma_storage_db"
)

# Initialize retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Generation model parameters
generation_params = {
    "decoding_method": "sample",
    "max_new_tokens": 100,
    "temperature": 0.5,
    "top_k": 50,
}

# Initialize WatsonxLLM
# generative_model = WatsonxLLM(
#     model_id="ibm/granite-13b-instruct-v2",
#     url=watsonx_url,
#     project_id=project_id,
#     params=generation_params,
# )

# Use openAI for testing until WATSONX is provided
generative_model = ChatOpenAI(model="gpt-4o-mini")

# Set up RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=generative_model,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

def list_files_in_folder(folder_id):
    """List all files in a specified Google Drive folder."""
    query = f"'{folder_id}' in parents and trashed = false"
    results = drive_service.files().list(q=query).execute()
    items = results.get('files', [])
    return items

def download_file(file_id, file_name):
    """Download a file from Google Drive."""
    request = drive_service.files().get_media(fileId=file_id)
    file_path = os.path.join('downloads', file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with io.FileIO(file_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
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
    # docs = sanitize_metadata(documents)
    uuids = [str(uuid4()) for _ in range(len(docs))]
    return [docs, uuids]

#TODO: check which performs better filter_complex_metadata() or sanitize_metadata()
# def sanitize_metadata(docs):
#     """Convert list-type metadata values to strings."""
#     for doc in docs:
#         for key, value in doc.metadata.items():
#             if isinstance(value, list):
#                 # Convert list to a comma-separated string
#                 doc.metadata[key] = ', '.join(map(str, value))
#     return docs

#TODO: issue when batch_size is > 99
def create_embeddings_and_store(docs, uuids):
    """Generate embeddings and store them in Chroma DB."""

    batch_size = 99  # Adjust this value based on your system's constraints
    print('length of docs', len(docs), batch_size)
    try:
        for i in range(0, len(docs), batch_size):
            print('current batch is', i)
            batch = docs[i:i+batch_size]
            print(batch)
            vector_store.add_documents(documents=batch, ids=uuids[i:i+batch_size])
            print('stored current batch')
        print("Embeddings stored successfully in Chroma DB.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print("Embeddings stored successfully in Chroma DB.")

if __name__ == '__main__':
    # Replace 'your_folder_id' with the actual folder ID from Google Drive
    folder_id = ''
    local_paths = []
    files = list_files_in_folder(folder_id)
    for file in files:
        file_id = file['id']
        file_name = file['name']
        mime_type = file['mimeType']
        print(f"Processing file: {file_name} ({mime_type})")

        # Download the file
        local_path = download_file(file_id, file_name)
        print(local_path)
        local_paths.append(local_path)
    docs, uuids = load_docs_unstructured(local_paths)
    create_embeddings_and_store(docs, uuids)

