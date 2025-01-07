# Customer-Support-Agents
Los agentes aprovechan el poder de modelos de lenguaje grandes (LLMs por sus siglas en ingles) para implementar soluciones a tareas no deterministas, es decir, que no siempre se resuelven de la misma manera. Por ejemplo, en este proyecto se estudia el caso de un chat automtizado para servicio al cliente, el cual se encarga de resolver las dudas mas comunes de los usuarios del producto fictisio SmartFlow.

Para implementar este sistema se recurrira a un sistema multiagente. En primer lugar, se diseñaran dos agentes basados en RAG (Retrieval Augmented Generation), los cuales se encargan de responder preguntas considerando un conocimiento ingresado por el diseñador (nosotros en este caso). Cada agente se especializara en dudas relacionadas a dos documentos: el manual de instrucciones del producto y una guia de los problemas mas coumnes. Para acceder a estos sistemas, se implementa un tercer agente, cuya tarea consiste en dada una pregunta del usuario, identificar que agente esta mejor preparado para responderla. El flujo del sistema se muestra a continuacion:

# Implementación
La implementacion del sistema se realizo con Llama 3.1, el cual es un LLM open source de 7B de parametros, por lo cual es ideal para implementarse en una gran variedad de dispositivos.

## Vector Store
En primer lugar, se implemento el proceso de escritura en un vector store ([vector_store_ingestion.py](vector_store_ingestion.py)). Para ello se dividio cada documento en chunks, y se evaluo a cada uno con el embedding de Llama 3.1. Posteriormente, dado que los documentos son pequeno, se almacenaron los resultados en el local vector store FAISS.

Esta implementacion se usa para procesar los archivos [instructions.txt](instructions.txt) y [troubleshooting.txt](troubleshooting.txt), resultando en los archivos generados en la carpeta [vector_stores](vector_stores).


```python
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

    # chunks splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=200)
    chunks = text_splitter.split_documents(document)
    print(f'\nText splited in {len(chunks)} chunks')

    # evaluation of chunks with embedding and vectore store ingestion
    print(f'\nAdding vector indexes into vector store ...')
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(f'vector_stores/{vs_name}') # save to local

    print('\nDone!')

if __name__ == '__main__':
    ingest_docs(doc_path='troubleshooting.txt', vs_name='troubleshooting')

```