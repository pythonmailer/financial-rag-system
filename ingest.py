import os
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import uuid

# 1. Initialize our AI Model and Database Connection
print("Loading embedding model (this takes a few seconds the first time)...")
model = SentenceTransformer("all-MiniLM-L6-v2") # Fast, free, local embedding model
qdrant = QdrantClient(url="http://localhost:6333")

# 2. Download the SEC Filing
COMPANY_TICKER = "AAPL"
print(f"Downloading latest 10-K report for {COMPANY_TICKER} from the SEC...")
# The downloader requires an email address as a user-agent to comply with SEC rules
dl = Downloader("MyRAGProject", "your.email@example.com", "./sec_data")
dl.get("10-K", COMPANY_TICKER, limit=1, download_details=True)

# 3. Find and Read the Downloaded File
# The downloader creates a nested folder structure, so we search for the HTML file
download_dir = f"./sec_data/sec-edgar-filings/{COMPANY_TICKER}/10-K"
html_file_path = None

for root, dirs, files in os.walk(download_dir):
    for file in files:
        if file.endswith(".html") or file == "primary_document.html":
            html_file_path = os.path.join(root, file)
            break

if not html_file_path:
    print("Error: Could not find the downloaded HTML file.")
    exit()

print("File downloaded! Cleaning HTML and extracting text...")
with open(html_file_path, "r", encoding="utf-8") as f:
    raw_html = f.read()

# Clean the HTML to get pure text
soup = BeautifulSoup(raw_html, "html.parser")
clean_text = soup.get_text(separator="\n", strip=True)

# 4. Chunk the Text
print("Splitting document into manageable chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_text(clean_text)
print(f"Generated {len(chunks)} chunks from the 10-K report.")

# 5. Embed and Upload to Qdrant
print("Embedding chunks and uploading to Qdrant...")
points = []
for i, chunk in enumerate(chunks):
    # Convert text to vector math
    vector = model.encode(chunk).tolist()
    
    # Create the payload (Metadata)
    payload = {
        "ticker": COMPANY_TICKER,
        "document_type": "10-K",
        "text": chunk
    }
    
    # Create a unique point for Qdrant
    points.append(
        PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
    )

# Upload in batches so we don't crash the database
qdrant.upsert(
    collection_name="financial_documents",
    points=points
)

print(f"✅ Success! {len(chunks)} vectors uploaded to Qdrant database for {COMPANY_TICKER}.")