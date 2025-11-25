import os
import re
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_ACCESS_TOKEN")
INDEX_NAME = "shl-individual-tests"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
CSV_PATH = "shl_individual_tests.csv"

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <- For local dev and simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Pydantic model for input ---------
class AssessmentRequest(BaseModel):
    query: str

# --------- Data & RAG logic ---------
def load_shl_catalog(csv_path: str):
    loader = CSVLoader(file_path=csv_path, encoding="utf-8")
    documents = loader.load()
    return documents

def get_embeddings():
    from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
    import os

    HF_API_KEY = os.getenv("HUGGINGFACE_ACCESS_TOKEN")  # Loaded from .env

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_API_KEY,
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    return embeddings

#def get_embeddings():
    #return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def init_pinecone_and_vectorstore(documents, embeddings):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=INDEX_NAME,
        )
    else:
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
        )
    return vectorstore

def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=512,
    )

def build_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 30, "fetch_k": 60, "lambda_mult": 0.7}
    )
    prompt = ChatPromptTemplate.from_template(
        """
You are a catalog assistant for SHL test catalog.
Given catalog context and user requirement, pick and recommend 1-10 DISTINCT relevant tests.
Return your answer ONLY as valid JSON in this format:

{{
    "recommended_assessments": [
        {{
          "url": "...",
          "name": "...",
          "adaptive_support": "Yes|No",
          "description": "...",
          "duration": ...,
          "remote_support": "Yes|No",
          "test_type": ["..."]
        }},
        ...
    ]
}}

Do NOT return ANY explanation or text, only valid JSON!
<catalog_context>
{context}
</catalog_context>
User requirement:
{question}
"""
    )
    return (
        RunnableParallel(context=retriever, question=RunnablePassthrough())
        | prompt
        | llm
        | StrOutputParser()
    )

def recommend_individual_tests(user_requirement: str):
    documents = load_shl_catalog(CSV_PATH)
    embeddings = get_embeddings()
    vectorstore = init_pinecone_and_vectorstore(documents, embeddings)
    llm = get_llm()
    rag_chain = build_rag_chain(vectorstore, llm)
    answer = rag_chain.invoke(user_requirement)
    # Robust JSON extraction: extract the biggest {...} block from answer
    match = re.search(r"\{[\s\S]+\}", answer)
    result_json = answer if not match else match.group(0)
    try:
        result = json.loads(result_json)
    except Exception:
        result = {"recommended_assessments": []}
    return result

# --------- API Endpoints ---------
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/recommend")
async def recommend(request: AssessmentRequest):
    result = recommend_individual_tests(request.query)
    return JSONResponse(content=result)

# --------- For local testing ---------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_api_fastapi:app", host="localhost", port=8000, reload=True)