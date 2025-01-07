from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_docs(doc_path, vs_name):
    # embedding definition
    embedding = OllamaEmbeddings(model='llama3.1')

    # txt reading
    print('Reading .txt ...')
    loader = TextLoader(file_path=doc_path)
    document = loader.load()

    # split en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=200)
    chunks = text_splitter.split_documents(document)
    print(f'\nText splited in {len(chunks)} chunks')

    # evaluamos chunks con embeding y alamcenamos en vector store
    print(f'\nAdding vector indexes into vector store ...')
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(f'vector_stores/{vs_name}') # guarda en local

    print('\nDone!')

if __name__ == '__main__':
    ingest_docs(doc_path='troubleshooting.txt', vs_name='troubleshooting')