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
        embedding = self.ollama.embeddings(model="nomic-embed-text:latest", prompt=context, options={"temperature": 0.1, "num_ctx": 10096})["embedding"]
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
            
            #print (emb_norm, emb2_norm)
            
            if (len(emb_norm) != len(emb2_norm)):
                continue

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

        fragments = context.split('. ')
        print (f"Adding {len(fragments)} fragments to document '{document_name}'")

        #for i, fragment in enumerate(fragments): if fragment is less than 10 words sum it up to the next fragment and skip the next cause is already added
        for i, fragment in enumerate(fragments):
            if len(fragment.split(' ')) < 10:
                fragments[i] = fragments[i] + '. ' + fragments[i+1]
                fragments.pop(i+1)

        print (f"Adding {len(fragments)} fragments to document '{document_name}'")

        for i, fragment in enumerate(fragments):
            
            
                
            print (f"Adding fragment {i+1} of {len(fragments)} to document '{document_name}'")
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
    
    from scipy.spatial.distance import cosine

    def get_n_best_matches(self, embedding, n=5):
        # Get all stored embeddings from the database
        db_embeddings = self.get_all_embeddings()

        # Ensure input embedding is a list of floats, not a string
        if isinstance(embedding, str):
            embedding = json.loads(embedding)

        # Normalize the input embedding
        npArrInput = np.array(embedding, dtype=float)
        npArrInput_norm = npArrInput / np.linalg.norm(npArrInput)

        # List to store (similarity, embedding_id) tuples
        similarities = []

        # Iterate over all stored embeddings
        for emb_id, emb in  enumerate(db_embeddings):  # Assuming db_embeddings is a list of (id, embedding) tuples
            # Ensure each stored embedding is a list of floats, not a string
            if isinstance(emb, str):
                emb = json.loads(emb)

            # Normalize the stored embedding
            emb_norm = np.array(emb) / np.linalg.norm(emb)

            # Check if the dimensions match
            if len(emb_norm) != len(npArrInput_norm):
                continue

            # Compute cosine similarity
            cos_sim = 1 - cosine(emb_norm, npArrInput_norm)

            # Store the similarity and the embedding id
            similarities.append((cos_sim, emb_id))

        # Sort based on cosine similarity in descending order
        similarities = sorted(similarities, key=lambda x: x[0], reverse=True)

        # Get the top N best matches
        best_matches = similarities[:n]

        # Fetch the corresponding texts from the database using the embedding IDs
        results = []
        for _, emb_id in best_matches:
            result = self.db.cursor.execute('SELECT context FROM embeddings WHERE id = ?', (emb_id,)).fetchone()
            if result:
                results.append(result[0])  # Assuming the text is in the first column

        return results
        
    
    def summarize_fragment(self, system_prompt, fragment):
        prompt = f"{system_prompt} {fragment}"
        return self.get_n_best_matches(prompt, 1)[0]
        