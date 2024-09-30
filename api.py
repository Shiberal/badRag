import atexit
import os
from time import sleep
from fastapi import FastAPI, File, UploadFile, HTTPException
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

# Load environment variables
load_dotenv()

# Initialize the NLP and OllamaRAG components
ollama = OllamaRAG('ollama.db', os.getenv('HOST_OLLAMA'))
nlp = NLP('nlp_documents.db')
print("NLP and OllamaRAG initialized")

# Register the cleanup function to save models and close resources
def on_exit():
    ollama.close()
    nlp.close()

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
    return {"message": "Welcome to the NLP API"}

@app.post("/addDocument")
async def add_document(file: UploadFile = File(...)):
    """
    Endpoint to add a document with multiple fragments.
    Breaks down the uploaded document into fragments (paragraphs)
    and stores them in the database.
    """
    try:
        # Read the file contents
        contents = await file.read()
        paragraphs = nlp.break_down_to_paragraphs(contents.decode('utf-8'))
        
        if not paragraphs:
            raise HTTPException(status_code=400, detail="Document is empty or cannot be processed")
        
        # Using the file's name as the document title, or a default title if none
        title = file.filename or "Untitled Document"
        
        # Add the document and its fragments
        nlp.add_document_with_fragments(title, paragraphs)
        nlp.save_models()
        
        return UJSONResponse(status_code=200, content="Document with fragments added successfully")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query")
async def query(query: str, document_id: int = 0):
    """
    Endpoint to search for documents by querying similar fragments
    and returning aggregated document results.
    """
    try:
        results = nlp.search_similar_texts(query, document_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="No similar documents found")
        
        return UJSONResponse(status_code=200, content=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

# Register the app shutdown event
@app.on_event("shutdown")
def shutdown_event():
    on_exit()