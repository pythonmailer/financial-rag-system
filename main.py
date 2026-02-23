import os
import boto3
import torch
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv

load_dotenv()

# Initialize AWS, Embedding, and Qdrant
s3 = boto3.client('s3')
model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url="http://localhost:6333")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

def process_s3_batch():
    print(f"🚀 Starting batch processing from S3 bucket: {BUCKET_NAME}")
    
    # 1. List files in S3 (looking for raw SEC HTML files)
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix='sec_data/')
    
    if 'Contents' not in response:
        print("ℹ️ No new files found in S3.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for obj in response['Contents']:
        file_key = obj['Key']
        if not file_key.endswith('.html'): continue
        
        # Define Ticker and Doc Type from folder structure (e.g., sec_data/AAPL/10-K/file.html)
        parts = file_key.split('/')
        ticker = parts[1]
        doc_type = parts[2]

        print(f"📦 Processing {doc_type} for {ticker}...")

        # 2. Download from S3
        file_obj = s3.get_object(Bucket=BUCKET_NAME, Key=file_key)
        html_content = file_obj['Body'].read().decode('utf-8')

        # 3. Clean and Split
        soup = BeautifulSoup(html_content, 'html.parser')
        raw_text = soup.get_text(separator=' ')
        chunks = text_splitter.split_text(raw_text)

        # 4. Vectorize and Upload to Qdrant
        points = []
        for i, chunk in enumerate(chunks):
            vector = model.encode(chunk).tolist()
            points.append(models.PointStruct(
                id=hash(f"{file_key}_{i}") & 0xFFFFFFFFFFFFFFFF, # Unique 64-bit int
                vector=vector,
                payload={
                    "text": chunk,
                    "ticker": ticker.upper(),
                    "doc_type": doc_type,
                    "source": f"s3://{BUCKET_NAME}/{file_key}"
                }
            ))
        
        # Upload in batches of 100 for efficiency
        qdrant.upsert(collection_name="financial_documents", points=points)
        print(f"✅ Successfully ingested {len(points)} chunks for {ticker}.")

if __name__ == "__main__":
    process_s3_batch()