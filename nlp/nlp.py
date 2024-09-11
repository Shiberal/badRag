import os
import string
from time import time
import spacy
import gensim
import re
from sqlite3 import connect
from operator import itemgetter
from gensim import similarities
from gensim import corpora


class NLP:
    def __init__(self, db_name):
        self.db = connect(db_name)
        self.punctuation = string.punctuation
        self.spacy_nlp = spacy.load('en_core_web_sm')
        self.stopwords = spacy.lang.en.stop_words.STOP_WORDS
        self.documents = []  # Store fragments as dictionaries with 'fragment_id' and 'text'
        self.corpus = []  # Store the corpus (bag-of-words for each fragment)
        self.dictionary = None
        self.tfidf_model = None
        self.lsi_model = None
        self.index = None
        
        # Create tables if they don't exist
        self.create_tables()
        
        # Load documents and fragments from the database
        self.load_documents()
        self.load_fragments()
        

    def load_documents(self):
        self.documents = self.db.execute("SELECT title FROM documents;").fetchall()
        print(f"Loaded documents: {len(self.documents)}")
    
    def create_tables(self):
        """
        Create the tables for storing documents, fragments, and their relationships.
        """
        queries = [
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS fragments (
                fragment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS document_fragment (
                doc_id INTEGER,
                fragment_id INTEGER,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
                FOREIGN KEY (fragment_id) REFERENCES fragments(fragment_id)
            );
            """
        ]
        for query in queries:
            self.db.execute(query)
        self.db.commit()

    def clean(self, text):
        text = text.lower()
        text = re.sub('\'', '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(' +', ' ', text)
        text = re.sub(r'\n: \'\'.*', '', text)
        text = re.sub(r'\n!.*', '', text)
        text = re.sub(r'^:\'\'.*', '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)

        text = text.replace('\r', ' ')
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        text = text.replace('  ', ' ')
        
        text = text.replace("\r\n\r\n", " ")
        
        return text

    def tokenizer(self, text):
        text = self.clean(text)
        tokens = self.spacy_nlp(text)
        tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]
        tokens = [word for word in tokens if word not in self.stopwords and word not in self.punctuation and len(word) > 2]
        return tokens

    def buildWordDict(self, tokens_list):
        from gensim import corpora
        dictionary = corpora.Dictionary(tokens_list)
        return dictionary

    def filterWordDict(self, dictionary, tokens='hello and if this can would should could tell ask stop come go'):
        stoplist = set(tokens.split())
        stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
        dictionary.filter_tokens(stop_ids)
        return dictionary

    def add_document_with_fragments(self, title, fragments):
        """
        Add a new document with its fragments to the database.
        """
        # Insert the document into the documents table
        cursor = self.db.cursor()
        cursor.execute("INSERT INTO documents (title) VALUES (?);", (title,))
        self.db.commit()
        doc_id = cursor.lastrowid  # Get the document ID

        # Insert fragments and associate them with the document
        for fragment_text in fragments:
            cursor.execute("INSERT INTO fragments (text) VALUES (?);", (fragment_text,))
            fragment_id = cursor.lastrowid
            cursor.execute("INSERT INTO document_fragment (doc_id, fragment_id) VALUES (?, ?);", (doc_id, fragment_id))

        self.db.commit()
        
        # Update in-memory structures
        for fragment_text in fragments:
            tokens = self.tokenizer(fragment_text)

            if not self.dictionary:
                self.dictionary = self.buildWordDict([tokens])
                
            else:
                self.dictionary.add_documents([tokens])

            self.dictionary = self.filterWordDict(self.dictionary)
            
            bow = self.dictionary.doc2bow(tokens)
            
            self.corpus.append(bow)
            
        
        self._update_models()

        print (self.tfidf_model, self.lsi_model)

    def load_document_with_fragments(self, doc_id):
        """
        Load a document and its fragments by document ID.
        """
        cursor = self.db.cursor()

        # Fetch the document
        cursor.execute("SELECT title FROM documents WHERE doc_id = ?;", (doc_id,))
        document = cursor.fetchone()

        # Fetch the associated fragments
        cursor.execute("""
            SELECT f.fragment_id, f.text 
            FROM fragments f
            JOIN document_fragment df ON f.fragment_id = df.fragment_id
            WHERE df.doc_id = ?;
        """, (doc_id,))
        fragments = cursor.fetchall()


        return {
            'doc_id': doc_id,
            'title': document[0],
            'fragments': [{'fragment_id': frag[0], 'text': frag[1]} for frag in fragments]
        }

    def load_fragments(self):
        """
        Load all fragments from the database and populate the in-memory structures.
        """
        cursor = self.db.cursor()
        cursor.execute("SELECT fragment_id, text FROM fragments;")

        rows = cursor.fetchall()
        print ("Loading {} fragments from database".format(len(rows)))
        for idx,row in enumerate(rows):
            print("Loading fragment % {}".format( idx / len(rows) * 100))
            fragment_id, text = row
            self.documents.append({'fragment_id': fragment_id, 'text': text})
            tokens = self.tokenizer(text)
            if self.dictionary:
                bow = self.dictionary.doc2bow(tokens)
            else:
                # Initialize dictionary with the first document
                self.dictionary = self.buildWordDict([tokens])
                bow = self.dictionary.doc2bow(tokens)
            self.corpus.append(bow)
            self._update_models()
            
    
    def break_down_to_paragraphs(self, text):
        """
        Break down the text into paragraphs. Here, paragraphs are assumed to be separated by new lines.
        This can be customized as per the specific document structure.
        """
        return [para.strip() for para in text.split("\n") if para.strip()]
    
    def _update_models(self):
        """
        Internal function to update TF-IDF, LSI models and index after corpus changes.
        """
        if self.corpus:
            # Create the TF-IDF model
            self.tfidf_model = gensim.models.TfidfModel(self.corpus, id2word=self.dictionary)

            # Create the LSI model based on the TF-IDF model
            self.lsi_model = gensim.models.LsiModel(self.tfidf_model[self.corpus], id2word=self.dictionary, num_topics=200)

            # Create the index for similarity searches
            self.index = similarities.MatrixSimilarity(self.lsi_model[self.tfidf_model[self.corpus]], num_features=200)
        else:
            print("Corpus is empty. Cannot update models.")

    def search_similar_texts(self, search_term, document_id = 0):
        time_start = time()
        """
        Search for texts similar to the input search term.
        """
        query_bow = self.dictionary.doc2bow(self.tokenizer(search_term))
        print(f"Query: {query_bow}")
        query_tfidf = self.tfidf_model[query_bow]
        print(f"Query TF-IDF: {query_tfidf}")
        query_lsi = self.lsi_model[query_tfidf]
        print(f"Query LSI: {query_lsi}")

        self.index.num_best = 10
        similar_texts = self.index[query_lsi]
        similar_texts.sort(key=itemgetter(1), reverse=True)
        print (f"Similar texts: {similar_texts}")
        results = []

        for j, text in enumerate(similar_texts):
            # Adjust the document index by subtracting 1 from the database ID
            document_index = text[0] + document_id
            if document_index < 0 or document_index >= len(self.documents):
                continue

            results.append({
                'Relevance': round((text[1] * 100), 2),
                'Document ID': self.documents[document_index]['fragment_id'],
                'Document Text': self.documents[document_index]['text']
            })

            if j == (self.index.num_best - 1):
                break

        print(f"Search time: {time() - time_start:.4f}s")
        return results
    
    def break_down_to_paragraphs(self, text):
        paragraphs = text.split('. ')
        return paragraphs
    

    def save_models(self):
        print ("Saving models to disk...")
        """
        Save the models and index to disk.
        """
        if self.tfidf_model:
            self.tfidf_model.save('model_tfidf.model')
        if self.lsi_model:
            self.lsi_model.save('model_lsi.model')
        if self.index:
            self.index.save('model_index.index')
        if self.dictionary: 
            self.dictionary.save('model_dictionary.dict')
                
        print("Saved models to disk.")
        
    def load_models(self):
        print ("Loading models from disk...")
        """ 
        Load the models and index from disk.
        """
        if os.path.exists('model_tfidf.model'):
            self.tfidf_model = gensim.models.TfidfModel.load('model_tfidf.model')
            print ("TF-IDF model loaded.")
        else:
            print("TF-IDF model not found.")
            
        if os.path.exists('model_lsi.model'):
            self.lsi_model = gensim.models.LsiModel.load('model_lsi.model')
            print ("LSI model loaded.")
        else:
            print("LSI model not found.")
            
        if os.path.exists('model_index.index'):
            self.index = similarities.MatrixSimilarity.load('model_index.index')
            print ("Index loaded.")
        else:
            print("Index not found.")
        if os.path.exists('model_dictionary.dict'):
            self.dictionary = corpora.Dictionary.load('model_dictionary.dict')
            print ("Dictionary loaded.")