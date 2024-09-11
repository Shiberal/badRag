import os
from time import sleep
from requests import get, post
from ollama import Client
from dotenv import load_dotenv # type: ignore

load_dotenv()
'''
GET
/addFragment
Addfragment


GET
/addWholeDocument
Addwholedocument


GET
/getBestMatch
Getbestmatch


GET
/listAllDocuments
Listalldocuments


GET
/removeWholeDocument
Removedocument


GET
/removeFragment
Removefragment


GET
/generateEmbedding
Generateembedding


GET
/getWholeDocumentByFragment
Getdocumentbyfragment
'''


class Chat:
    def __init__(self):
        self.context = ""
        self.conversation = []
        self.base_ragurl = "http://localhost:8000"

    def send_message(self, message):
        # self.get_context_from_rag(message)
        self.get_mutliple_contexts_from_rag(message, 4)
       
        response = Client(os.getenv('HOST_OLLAMA')).chat(model="llama3.1", messages=[{"role": "user", "content": message}], options={"temperature": 0.1 , "num_ctx": 10096})
        self.conversation.append({"user": message, "llm": response['message']["content"]})
        self.context += message + "\n" + str(response['message']["content"]) + "\n"

    def get_conversation(self):
        return '\n'.join([f"{k}: {v}" for pair in self.conversation for k,v in pair.items()])
    
 
            
    def get_mutliple_contexts_from_rag(self, message, n_contexts):   
        #http://127.0.0.1:8000/getNBestMatch?text=who%20the%20doctor%20was%3F&n=4
        results = []
        results = (get(f"{self.base_ragurl}/getNBestMatch?text={message}&n={n_contexts}").json()) #outputs a list

        
        for result in results:
            #try to summarize each result
            print (result , end="\n\n\n")
            self.conversation.append({"user": str(result)})
            
        return results
    
    
            
    
    def get_context_from_rag(self, message):
        result = get(f"{self.base_ragurl}/getBestMatch?text={message}")
        #res_sum = self.summarize_fragment(system_prompt = "summarize this text in one sentence", fragment = result.json()["content"])
        self.conversation.append({"user": str(result.json()["content"])})
    
    def add_full_document(self, document_name, context):
        #Break document into fragments, split by \n\n
        fragments = context.split('\n\n')

        for i, fragment in enumerate(fragments):
            post(f"{self.base_ragurl}/addFragment?document_name={document_name}", {"text": fragment})
            #print(f"Added fragment {i+1} of {len(fragments)} to document '{document_name}'")
    
    def summarize_fragment(self, system_prompt = "summarize this text in one sentence", fragment = ""):
        if fragment == "":
            return ""
        response = Client("http://192.168.1.44:11434").chat(model="llama3.1", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": fragment}], options={"temperature": 0.1 , "num_ctx": 10096})
        return response['message']["content"]
    

chat = Chat()
while True:
    message = input("You: ")
    
    if message == "/exit":
        break
    if message == "/add":
        fragment = input("Fragment: ")
        document = input("Document: ")
        chat.add_full_document(document, fragment)
        continue

    if message == "/add file":
        document = input("Document: ")
        with open(document, 'r') as f:
            context = f.read()
        chat.add_full_document(document, context)
        continue

    if message == "/clear":
        chat.conversation = []
        continue
    
    if message == "/get":
        #get fragment
        message = input("Fragment: ")
        continue
    
    chat.send_message(message)
    #clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    print(chat.get_conversation())