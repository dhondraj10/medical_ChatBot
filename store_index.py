from src.helper import load_pdf_file,text_split,download_hugging_face_embedding
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
load_dotenv()

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY']=PINECONE_API_KEY



extracted_data=load_pdf_file(data='D:\GenAI\My\medical_ChatBot\Data/')
text_chunks=text_split(extracted_data)
embeddings=download_hugging_face_embedding()


# Initialize client
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalchatbotfinal"

# Check if index exists
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
            
        )
    )


docsearch=PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)
