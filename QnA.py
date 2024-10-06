import json
import pandas as pd
import numpy as np
import textwrap
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
import urllib.request
import re
import vertexai
import asyncio
from vertexai.preview.generative_models import GenerativeModel, ChatSession
import nltk
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
load_dotenv()


GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
VECTOR_FOLDER = 'vector_store_dir'

def load_and_chunk(docs):
    loader = PyPDFLoader(docs)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separators="\n")
    chunks_docs = splitter.split_documents(documents=documents)
    # print(chunks_docs)
    return chunks_docs



def vectorize_and_store(chunks_docs, pdf_path):
    pdf_filename = os.path.basename(pdf_path)
    index_name = os.path.splitext(pdf_filename)[0] + "_index"
    index_path = os.path.join(VECTOR_FOLDER, index_name)
    if os.path.exists(index_path):
        print(f"Vector store already exists for {pdf_filename}. Skipping vectorization.")
        return index_path
    embeddings = embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store_db = FAISS.from_documents(chunks_docs, embedding=embeddings)
    vector_store_db.save_local(index_path)
    print(f"Vector store created for {pdf_filename} - {index_path}")
    return index_path


def get_conversational_chain():
    prompt_template = """
    You are tasked as a Medical Research Paper Analyst. Your role is to provide insightful answers to questions related to the provided medical research paper. Your response should be scientifically accurate and easily understandable by both medical students and the general public.
    Your primary objective is to comprehend the user's question, identify its components (if any), and generate a response that effectively addresses the query within the context of the specific medical field. You will leverage the provided corpus to provide accurate and helpful responses. In cases where the corpus lacks relevant context, rely on your own knowledge base to provide relevant information. 

    Context: {context}
    Question: {question}

    If the Corpus doesn't have sufficient context relevant to the user's query, you must refer to your own knowledge base in order to come up with the three components as mentioned above. However, make sure that they are precise. You must not provide fake information.
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1, google_api_key=GOOGLE_API_KEY)
    # model = genai.GenerativeModel("gemini-pro", generation_config=GenerationConfig(temperature=0.1))
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    # model = genai.GenerativeModel("gemini-pro")
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def generate_answer(pdf_path, query):
    chunks_doc = load_and_chunk(pdf_path)
    index_path = vectorize_and_store(chunks_doc, pdf_path)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(query)
    chain = get_conversational_chain()

    response = chain(
            {"input_documents":docs, "question":query},
            return_only_outputs=True
        )
    
    print(response)
    return response

# query = "what is the gestational age?"
query = "What are the key factors that influence and promote healthy brain development in children?"
generate_answer(r"Brain development in children.pdf", query)