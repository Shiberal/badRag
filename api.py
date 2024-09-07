from typing import Union
from fastapi import FastAPI # type: ignore

from ollama_rag import OllamaRAG

ollama = OllamaRAG('ollama.db', "http://192.168.1.44:11434")

app = FastAPI()

@app.get("/addFragment")
async def addFragment(text_fragment: str, document_name: str):
    return {"content": text_fragment, "document_name": document_name, "embedding": ollama.generate_embeddings(text_fragment, document_name)}

@app.get("/addWholeDocument")   
async def addWholeDocument(text: str, document_name: str):
    return ollama.add_full_document(document_name, text)

@app.get("/getBestMatch")
async def getBestMatch(text: str):
    best, best_indx, score, text = ollama.get_best_match(ollama.generate_embedding(text))
    return {"content":  text, "score" : score}

@app.get("/listAllDocuments")
async def listAllDocuments(limit: int = 0, offset: int = 0):
    return ollama.listAllDocuments( limit, offset)

@app.get("/removeWholeDocument")
async def removeDocument(document_name: str):
    ollama.remove_whole_document(document_name)
    return {"document_name": document_name}

@app.get("/removeFragment")
async def removeFragment(id):
    ollama.remove_fragment(id)
    return {"id": id}

@app.get("/generateEmbedding")
async def generateEmbedding(text: str, document_name: str):
    return ollama.generate_embeddings(text, document_name)

@app.get("/getWholeDocumentByFragment")
async def getWholeDocumentByFragment(id):
    return ollama.getDocumentByFragmentID(id)



