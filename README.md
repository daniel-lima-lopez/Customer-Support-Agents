# Customer Support Agents
Los agentes aprovechan el poder de modelos de lenguaje grandes (LLMs por sus siglas en inglés) para implementar soluciones a tareas no deterministas, es decir, que no siempre se resuelven de la misma manera. Por ejemplo, en este proyecto se estudia el caso de un chat automatizado para servicio al cliente, el cual se encarga de resolver las dudas mas comunes de los usuarios del producto ficticio SmartFlow.

Para implementar este sistema se recurre a un sistema multi-agente. En primer lugar, se diseñarán dos agentes basados en RAG (Retrieval Augmented Generation), los cuales se encargan de responder preguntas considerando un conocimiento ingresado por el diseñador (nosotros en este caso). Cada agente se especializará en dudas relacionadas con dos documentos: el manual de instrucciones del producto y una guía de los problemas más comunes. Para acceder a estos sistemas, se implementa un tercer agente, cuya tarea consiste en dada una pregunta del usuario, identificar que agente está mejor preparado para responderla.

# Implementación
La implementación del sistema se realizó con Llama 3.1, el cual es un LLM open source de 7B de parámetros, por lo cual es ideal para implementarse en una gran variedad de dispositivos.

## Vector Store
En primer lugar, se implementó el proceso de escritura en un vector store ([vector_store_ingestion.py](vector_store_ingestion.py)). Para ello se dividió cada documento en chunks, y se evaluó a cada uno con el embedding de Llama 3.1. Posteriormente, dado que los documentos son pequeños, se almacenaron los resultados en el local vector store FAISS.

Esta implementación se usa para procesar los archivos [instructions.txt](instructions.txt) y [troubleshooting.txt](troubleshooting.txt), resultando en los archivos generados en la carpeta [vector_stores](vector_stores).


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

## Agentes RAG
Ambos agentes (`InstructionsRagAgent()` y `TroubleshootingRagAgent()`) se implementaron con la misma metodología, en primer lugar, se define un embedding y se realiza la lectura del local vector store con FAISS, posteriormente se define un retriever para realizar el proceso de recuperación de información.

Posteriormente, se define la plantilla del prompt a usar para aprovechar el modelo Llama 3.1. Este prompt se compone de la siguiente manera:

```
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise.

context: {context}

question: {question}
```

De esta manera, en cada llamada se usa al LLM para responder a una pregunta, considerando únicamente la información identificada por el retriever. 

Finalmente, se define la cadena de LangChain usada para ejecutar todo el proceso, la cual se compone de un placeholder para recibir tanto la pregunta del usuario como la información recuperado por el retriever para contestarla, seguido de la plantilla a evaluar, el modelo LLM y un OutputParser para extraer la respuesta del modelo.

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

Posteriormente, se implementa el método `invoke_agent`, el cual se encarga de extraer la información más relevante para contestar la pregunta con el retriever, y de esta manera usar el LLM para responder a la pregunta de entrada considerando la información recuperada.

```python
    def invoke_agent(self, question, verbose=True):
        # get retriever context from question
        docs = self.retriever.invoke(question)
        context =  "\n\n".join([doc.page_content for doc in docs])

        if verbose:
            print(f'\nInformation retrieved by the Instruction Rag Agent:\n{context}')

        # get llm response considering the retrieved information
        output = self.agent.invoke({'context': context, 'question': question})

        return output
```



## Agente de decisión
Por último, se implementó un agente encargado de, dada la pregunta de entrada, identificar al agente con la información más adecuada para responder. Para ello, se define el conjunto de herramientas que tendrá disponible el agente para realizar su tarea, incluyendo las rutinas `invoke` de los agentes RAG.

Note que la definición de estas herramientas incluye la descripción de cada herramienta, con el fin de incluir esta información como contexto para que el agente sea capaz de identificar la herramienta más adecuada.

La plantilla elegida para este agente es la siguiente:

```
Considering the available tools and their tool names, return only the name of the tool to use to answer the following question:
        
tools: {tools description}

tool names: [{tool_names}]

question: {input}
```

La cadena de LangChain de este agente se construye considerando la platilla anterior, el modelo Llama 3.1 con temperatura 0 para devolver la respuesta más concisa y un output parser para extraer el nombre de la función a usar.

```python
class ToolsAgent:
    def __init__(self, verbose=False):
        self.verbose = verbose # to activate verbose

        # define rag agents
        self.instr_agent = InstructionsRagAgent()
        self.trbl_agent = TroubleshootingRagAgent()

        # tools list (to invoke instructions and troubleshooting agents)
        self.tools = [
            Tool(name='invoke_instructions_agent', func=self.invoke_instructions_agent,
            description='Given a question, returns information about the description of the product, product installation instructions, the packet content, its box content, installation steps, main features and maintenance'),
            Tool(name='invoke_troubleshooting_agent', func=self.invoke_troubleshooting_agent,
                 description='Given a question, returns information about the Troubleshooting Guide, including Wi-Fi connection issues, inaccurate water usage data, leak detection false alarms, system offline, automated watering not working')
            ]
        self.tool_names = ", ".join([t.name for t in self.tools])                                   
        
        # prompt definition
        tools_prompt = ('''
        Considering the available tools and their tool names, return only the name of the tool to use to answer the following question:
        
        tools: {tools}

        tool names: [{tool_names}]

        question: {input}
        
        ''')
        self.tools_promt = PromptTemplate.from_template(tools_prompt).partial(
            tools=render_text_description(self.tools),
            tool_names=self.tool_names
        )

        # lambda input dictionary
        input_dic = {'input': lambda x: x['input']}

        # llm and callback definition
        self.llm = ChatOllama(model='llama3.1', temperature=0.0)

        # definition of the tools pipe
        self.tools_agent = input_dic | self.tools_promt | self.llm | StrOutputParser()

    def invoke_agent(self, question):
        output = self.tools_agent.invoke({'input': question})

        # print choosen agent
        if self.verbose:
            print('\n-------TOOLS AGENT-------')
            if output=='invoke_instructions_agent':
                print('The Tools Agent chose the Instruction Rag Agent')
            else:
                print('The Tools Agent chose the Troubleshooting Rag Agent')
        
        # invoke the rag agent
        if output=='invoke_instructions_agent':
            answer = self.invoke_instructions_agent(question)
        else:
            answer = self.invoke_troubleshooting_agent(question)
        return answer
    
    # tool1
    def invoke_instructions_agent(self, question):
        '''Given a question, returns information about the description of the product, the packet content, its box content, installation steps, main features and maintenance'''
        return self.instr_agent.invoke_agent(question, verbose=self.verbose)

    # tool2
    def invoke_troubleshooting_agent(self, question):
        '''Given a question, returns information about the Troubleshooting Guide, including Wi-Fi connection issues, inaccurate water usage data, leak detection false alarms, system offline, automated watering not working'''
        return self.trbl_agent.invoke_agent(question, verbose=self.verbose)
```

La implementación completa se encuentra en el archivo [implementation.py](implementation.py).

## Ejemplo de uso
Podemos usar la clase `ToolsAgent()` para usar el modelo implementado para responder preguntas relacionadas con el producto SmartFlow. A continuación se muestran algunos ejemplos:

```python
# define the agent
test = ToolsAgent(verbose=True)

# input question
question = 'My device is not connecting to the Wi-Fi, what should I do?'
print(f'\n-------QUESTION-------\n{question}')

# get the answer by invoking the agent
answer = test.invoke_agent(question)
print(f'\n-------ANSWER-------\n{answer}')
```

```
-------QUESTION-------
My device is not connecting to the Wi-Fi, what should I do?

-------TOOLS AGENT-------
The Tools Agent chose the Troubleshooting Rag Agent

Information retrieved by the Troubleshooting Rag Agent:
3. Leak Detection False Alarms

Inspect the sensors for dirt or moisture.
Reset the leak detection sensitivity in the app settings.        

4. System Offline

Verify the power supply to the main unit.
Ensure the firmware is updated to the latest version.
Contact support if the issue persists.

Troubleshooting Guide for SmartFlow
1. Device Not Connecting to Wi-Fi

Ensure your router is within 15 meters of the main unit.
Restart the main unit and your router.
Verify that the Wi-Fi credentials entered in the app are correct.

2. Inaccurate Water Usage Data

4. System Offline

Verify the power supply to the main unit.
Ensure the firmware is updated to the latest version.
Contact support if the issue persists.

5. Automated Watering Not Working

Ensure the schedule is correctly configured in the app.
Verify the water valve is open and not obstructed.

-------ANSWER-------
Ensure your router is within 15 meters of the main unit. Restart the main unit and your router. Verify that the Wi-Fi credentials entered 
in the app are correct.
```

```python
# define the agent
test = ToolsAgent(verbose=True)

# input question
question = 'What are the installation steps to connect the main unit?'
print(f'\n-------QUESTION-------\n{question}')

# get the answer by invoking the agent
answer = test.invoke_agent(question)
print(f'\n-------ANSWER-------\n{answer}')
```

```
-------QUESTION-------
What are the installation steps to connect the main unit?

-------TOOLS AGENT-------
The Tools Agent chose the Instruction Rag Agent

Information retrieved by the Instruction Rag Agent:
2. Box Contents

SmartFlow Main Unit
SmartFlow Sensors (3 units)
Power Adapter
Mounting Kit
Quick Start Guide

3. Installation Steps

Position the Sensors: Place sensors near the main water inlet, garden area, and any critical appliances (e.g., washing machine).
Connect the Main Unit:
Plug the main unit into a power outlet.
Connect it to your Wi-Fi network using the mobile app.

Instruction Manual for SmartFlow
1. Introduction
Thank you for choosing SmartFlow! This intelligent home water management system is designed to help you save water, detect leaks, and maintain optimal hydration for your garden.

2. Box Contents

-------ANSWER-------
Connect the Main Unit:
1. Plug the main unit into a power outlet.
2. Connect it to your Wi-Fi network using the mobile app.
```
