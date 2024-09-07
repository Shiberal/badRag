import json
import ollama
from ollama import Client
from database import Database
import numpy as np
from scipy.spatial.distance import cosine


class OllamaRAG:
    def __init__(self, db_name, host):
        self.db = Database(db_name)
        self.ollama = Client(host)

    def generate_embeddings(self, context, document_name):
        embedding = self.ollama.embeddings(model="nomic-embed-text:latest", prompt=context)["embedding"]
        #check if embedding already exists
        if self.db.cursor.execute('SELECT * FROM embeddings WHERE document_name = ?', (document_name,)).fetchone():
            return embedding
        self.db.insert_embedding(document_name, context, embedding)
        return embedding

    def get_all_embeddings(self):
        embeddings = []
        texts = []
        for row in self.db.cursor.execute('SELECT * FROM embeddings'):
            embeddings.append(row[3])
        return embeddings
    
    def generate_embedding(self, context):
        return self.ollama.embeddings(model="nomic-embed-text:latest", prompt=context)["embedding"]
    
    def get_best_match(self, embedding):
        best_match = None
        best_similarity = -1
        best_indx = -1

    
        # Ensure input embedding is a list of floats, not a string
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        
        npArrInput = np.array(embedding, dtype=float)
    
        # Iterate over all stored embeddings
        for indx, emb in enumerate(self.get_all_embeddings()):
            # Ensure each stored embedding is a list of floats, not a string
            if isinstance(emb, str):
                emb = json.loads(emb)
            
            emb_norm = emb / np.linalg.norm(emb)
            emb2_norm = npArrInput / np.linalg.norm(npArrInput)
            

            cos_sim = 1 - cosine(emb_norm, emb2_norm)
            similarity = cos_sim
            
            # Check if the similarity is the best so far
            if similarity >= best_similarity:
                best_match = emb
                best_similarity = similarity
                best_indx = indx+1

        text = self.db.cursor.execute('SELECT * FROM embeddings WHERE id = ?', (best_indx,)).fetchone()[2]
        
        return best_match, best_indx, best_similarity, text

    def create_table(self):
        self.db.create_table()
        
    def listAllDocuments(self, limit: int = 0, offset: int = 0):
        if limit == 0:
            return self.db.cursor.execute('SELECT * FROM embeddings ').fetchall()
        else:
            return self.db.cursor.execute('SELECT * FROM embeddings LIMIT ? OFFSET ?', (limit, offset)).fetchall()
    
    def remove_whole_document(self, document_name):
        self.db.cursor.execute('DELETE FROM embeddings WHERE document_name = ?', (document_name,))
        self.db.conn.commit()
    
    def remove_fragment(self, id):
        self.db.cursor.execute('DELETE FROM embeddings WHERE id = ?', (id,))
        self.db.conn.commit()
    
    def getDocumentByFragmentID(self, id):
        document_name = self.db.cursor.execute('SELECT * FROM embeddings WHERE id = ?', (id,)).fetchone()[1]
        fragments = self.db.cursor.execute('SELECT * FROM embeddings WHERE document_name = ?', (document_name,)).fetchall()
        return fragments
    
    def add_full_document(self, document_name, context):

        #Break document into fragments, split by \n\n
        fragments = context.split('\n\n')

        for i, fragment in enumerate(fragments):
            embedding = self.generate_embedding(fragment)
            self.db.insert_embedding(document_name, fragment, embedding)
            print(f"Added fragment {i+1} of {len(fragments)} to document '{document_name}'")

        return True
    
    def addPDFDocument(self, document_name, file_path):
        pass

    
    def addWordDocument(self, document_name, file_path):
        pass
    
    
    def addHTMLDocument(self, document_name, file_path):
        pass