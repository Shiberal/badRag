from ollama_rag import OllamaRAG

def main():
    db_name = 'ollama.db'
    rag = OllamaRAG(db_name, "http://192.168.1.44:11434")
    rag.create_table()

    context = 'Hello, world!'
    document_name = 'example_document.txt'
    
    if __name__ == '__main__':
        import sys
        args = sys.argv[1:]
        
        if "--generate_embedding" in args:
            embedding = rag.generate_embedding(context, document_name)
            print(embedding)

if __name__ == '__main__':
    main()
