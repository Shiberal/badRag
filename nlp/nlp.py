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
        
        # Load models and dictionary
        self.load_dictionary()
        self.load_bow_from_db()
        self.load_documents()

    def create_tables(self):
        """
        Create the tables for storing documents, fragments, their relationships, and dictionary.
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
            """,
            """
            CREATE TABLE IF NOT EXISTS bow_data (
                doc_id INTEGER,
                word_id INTEGER,
                frequency INTEGER NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
                FOREIGN KEY (word_id) REFERENCES dictionary_data(word_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS dictionary_data (
                word_id INTEGER PRIMARY KEY,
                word TEXT NOT NULL
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

            # Tokenize and update dictionary
            tokens = self.tokenizer(fragment_text)
            if not self.dictionary:
                self.dictionary = self.buildWordDict([tokens])
            else:
                self.dictionary.add_documents([tokens])
            
            self.dictionary = self.filterWordDict(self.dictionary)
            bow = self.dictionary.doc2bow(tokens)
            self.corpus.append(bow)
            
            self.save_bow_to_db(doc_id, bow)

        self.db.commit()
        self._update_models()
        self.save_models()

    def save_bow_to_db(self, doc_id, bow):
        """
        Save the bag of words to the database.
        """
        cursor = self.db.cursor()
        for word_id, frequency in bow:
            cursor.execute("INSERT INTO bow_data (doc_id, word_id, frequency) VALUES (?, ?, ?);", (doc_id, word_id, frequency))
        self.db.commit()

    def load_documents(self):
        self.documents = self.db.execute("SELECT title FROM documents;").fetchall()
        print(f"Loaded documents: {len(self.documents)}")

    def load_dictionary(self):
        """
        Load the dictionary from the database if it exists.
        """
        cursor = self.db.cursor()
        cursor.execute("SELECT word_id, word FROM dictionary_data ORDER BY word_id;")
        rows = cursor.fetchall()
        
        if rows:
            self.dictionary = corpora.Dictionary()
            self.dictionary.id2token = {row[0]: row[1] for row in rows}
            self.dictionary.token2id = {v: k for k, v in self.dictionary.id2token.items()}
            print("Dictionary loaded from database.")
        else:
            print("No dictionary data found in the database. Dictionary needs to be rebuilt.")



    def load_bow_from_db(self):
        """
        Load BOW data from the database and reconstruct the corpus.
        """
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT doc_id, word_id, frequency
            FROM bow_data
            ORDER BY doc_id;
        """)
        rows = cursor.fetchall()

        if rows:
            # Determine the highest doc_id to initialize the corpus list
            max_doc_id = max(row[0] for row in rows)
            self.corpus = [[] for _ in range(max_doc_id + 1)]

            for doc_id, word_id, frequency in rows:
                if doc_id >= len(self.corpus):
                    continue
                self.corpus[doc_id].append((word_id, frequency))

            print("BOW data loaded from database.")
            self._update_models()  # Ensure models are updated after loading BOW
        else:
            print("No BOW data found in the database.")


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

    def search_similar_texts(self, search_term, document_id=0):
        time_start = time()
        """
        Search for texts similar to the input search term.
        """
        if self.dictionary is None:
            print("Error: Dictionary is not initialized.")
            return []

        query_bow = self.dictionary.doc2bow(self.tokenizer(search_term))
        print (f"Query BOW: {query_bow}")
        query_tfidf = self.tfidf_model[query_bow] if self.tfidf_model else []
        print (f"Query TF-IDF: {query_tfidf}")
        query_lsi = self.lsi_model[query_tfidf] if self.lsi_model else []
        print (f"Query LSI: {query_lsi}")

        self.index.num_best = 10
        similar_texts = self.index[query_lsi] if self.index else []
        similar_texts.sort(key=itemgetter(1), reverse=True)
        print (f"Similar texts: {similar_texts}")

        results = []
        for j, text in enumerate(similar_texts):
            # Adjust the document index by subtracting 1 from the database ID
            document_index = text[0] + document_id
            print (f"Document ID: {document_index}, Relevance: {text[1]}")
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


    def save_models(self):
        """
        Save the models to disk.
        """
        if self.tfidf_model:
            print ("Saving TF-IDF model to disk...")
            self.tfidf_model.save('model_tfidf.model')
        if self.lsi_model:
            print ("Saving LSI model to disk...")
            self.lsi_model.save('model_lsi.model')
        if self.index:
            print ("Saving index to disk...")
            self.index.save('model_index.index')
        if self.dictionary:
            print ("Saving dictionary to database...")
            cursor = self.db.cursor()
            for word_id, word in self.dictionary.id2token.items():
                cursor.execute("INSERT INTO dictionary_data (word_id, word) VALUES (?, ?)", (word_id, word))
            self.db.commit()
            
            

    def load_models(self):
        """
        Load the models from disk.
        """
        if os.path.exists('model_tfidf.model'):
            self.tfidf_model = gensim.models.TfidfModel.load('model_tfidf.model')
        if os.path.exists('model_lsi.model'):
            self.lsi_model = gensim.models.LsiModel.load('model_lsi.model')
        if os.path.exists('model_index.index'):
            self.index = similarities.MatrixSimilarity.load('model_index.index')
        

    def break_down_to_paragraphs(self, text):
        """
        Break down the text into paragraphs. Here, paragraphs are assumed to be separated by new lines.
        This can be customized as per the specific document structure.
        """
        return [para.strip() for para in text.split("\n") if para.strip()]