from googleapiclient.discovery import build
from google.oauth2 import service_account

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'elaborate-howl-415101-c308fb4eab27.json'

# Authenticate
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
iam_service = build('iam', 'v1', credentials=credentials)

# Get service account details
project_id = 'elaborate-howl-415101'
name = f'projects/{project_id}'

response = iam_service.projects().serviceAccounts().list(name=name).execute()
for account in response.get('accounts', []):
    print(f"Service Account Email: {account['email']}")