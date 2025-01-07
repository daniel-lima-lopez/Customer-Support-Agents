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

## RAG Agents
Ambos agentes se implementaron con la misma metodologia, en primer lugar, se define un embedding y se procede a realizar la lectura del local vector store con FAISS, y se define un retriever para realizar el proceso de recuperacion de informacion.

Posteriormente se define la plantilla del prompt a usar para aprovechar el modelo Llama 3.1. Este prompt se compone de la siguiente manera:

```
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise.

context: {context}

question: {question}
```

De esta manera, en cada llamada su usa al LLM para responder a una pregunta considerando unicamente la informacion identificada por el retriever. 

Finalmente se define la cadena de LangChain usada para ejecutar todo el proceso, la cual se compone de un placeholder para recibir tanto la pregunta del usuario como la informacion recuperado por el retriever para contestarla, seguido de la plantilla a evaluar, el modelo LLM y un OutputParser para extraer la respueta del modelo.

```python
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.agents import Tool # tool decorator
from langchain.tools.render import render_text_description # render the description of each tool to include in the prompt
from langchain_core.output_parsers.string import StrOutputParser
from langchain_community.vectorstores import FAISS # to implement vector stores

class InstructionsRagAgent:
    def __init__(self):
        # create embedding
        self.embedding = OllamaEmbeddings(model='llama3.1')

        # load information from vector store and create retriever
        self.vectorstore = FAISS.load_local('vector_stores/instructions', self.embedding, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # create llm prompt template
        prompt = '''
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise.

        context: {context}

        question: {question}
        '''
        self.promt_template = PromptTemplate.from_template(prompt)

        # lambda input dictionary
        input_dic = {'context': lambda x: x['context'],
                     'question': lambda x: x['question']}

        # llm definition
        self.llm = ChatOllama(model='llama3.1', temperature=0.6)

        # definition of rag agent chain
        self.agent = input_dic | self.promt_template | self.llm | StrOutputParser()
```