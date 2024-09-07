import sqlite3

class Database:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                document_name TEXT,
                context TEXT,
                embedding TEXT,
                tags TEXT
            )
        ''')
        self.conn.commit()

    def insert_embedding(self, document_name, context, embedding):
        # Convert the embedding dictionary into a string representation of the array
        embedding_str = str(embedding)
        
        # Automate tagging process (for now, just append 'example' to tags) <-- might be a todo.
        tags = 'example' # 
        
        self.cursor.execute('INSERT INTO embeddings (document_name, context, embedding, tags) VALUES (?, ?, ?, ?)', (document_name, context, embedding_str, tags))
        self.conn.commit()
