from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app=Flask(__name__)

load_dotenv()
PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY=os.environ.get("GROQ_API_KEY")



os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
os.environ['GROQ_API_KEY']=GROQ_API_KEY

embeddings=download_hugging_face_embedding()

index_name='medicalchatbotfinal'

from langchain_pinecone import PineconeVectorStore
docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever=docsearch.as_retriever(search_type='similarity',search_kwargs={"k":3})

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

prompt=ChatPromptTemplate.from_messages(
    [('system',system_prompt),
    ('human',"{input}")]
)

question_answe_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,question_answe_chain)





@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get",methods=["GET","POST"])
def chat():
    msg=request.form['msg']
    input=msg
    print(input)
    response=rag_chain.invoke({"input":msg})
    print("Response: ", response["answer"])
    return str(response["answer"])


if __name__=='__main__':
    app.run(host="0.0.0.0", port=8080,debug=True)
