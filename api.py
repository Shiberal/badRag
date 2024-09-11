import atexit
import os
from time import sleep
from typing import Union
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import UJSONResponse
from dotenv import load_dotenv
from nlp.nlp import NLP
from ollama_rag import OllamaRAG 
from fastapi.middleware.cors import CORSMiddleware

# CORS setup
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

load_dotenv()

# Initialize the NLP and OllamaRAG components
ollama = OllamaRAG('ollama.db', os.getenv('HOST_OLLAMA'))

nlp = NLP('nlp_documents.db')
print ("NLP and OllamaRAG initialized")

nlp.load_models()
print ("Models loaded")




# Register the cleanup function to save models and close resources
def on_exit():
    ollama.close()

atexit.register(on_exit)

# FastAPI app initialization
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/addDocument")
async def add_document(file: UploadFile = File(...)):
    """
    Endpoint to add a document with multiple fragments.
    Breaks down the uploaded document into fragments (paragraphs)
    and stores them in the database.
    """
    contents = await file.read()
    paragraphs = nlp.break_down_to_paragraphs(contents.decode('utf-8'))
    
    if not paragraphs:
        return UJSONResponse(status_code=400, content="Document is empty or cannot be processed")
    
    # Using the first line or a fixed value as the title, can be changed as needed
    title = file.filename or "Untitled Document"
    
    # Add the document with its paragraphs as fragments
    nlp.add_document_with_fragments(title, paragraphs)
    nlp.save_models()
    
    return UJSONResponse(status_code=200, content="Document with fragments added successfully")


@app.post("/query")
async def query(query: str, document_id: int = 0):
    """
    Endpoint to search for documents by querying similar fragments
    and returning aggregated document results.
    """
    res = nlp.search_similar_texts(query, document_id)
    
    if not res:
        return UJSONResponse(status_code=404, content="No similar documents found")

    return UJSONResponse(status_code=200, content=res)