from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.agents import tool # tool decorator
from langchain.tools.render import render_text_description # render the description of each tool to include in the prompt
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.callbacks.base import BaseCallbackHandler # for custom callbacks
from langchain_community.vectorstores import FAISS # to implement vector stores


# @tool
# def instructions_agent(input:str) -> str:
#     '''Given a question, returns information about the description of the product, the packet content, its box content, installation steps, main features and maintenance'''
#     return 'instruction agent'

# @tool
# def troubleshooting_agent(input:str) -> str:
#     '''Given a question, returns information about the Troubleshooting Guide, including Wi-Fi connection issues, inaccurate water usage data, leak detection false alarms, system offline, automated watering not working'''
#     return 'troubleshooting agent'

# class ToolsCallback(BaseCallbackHandler):
#     def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id = None, tags = None, metadata = None, **kwargs):
#         print(f'- Input to Tools Agent: {prompts[0]}')        
#         return super().on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, **kwargs)

#     def on_llm_end(self, response, *, run_id, parent_run_id = None, **kwargs):
#         print('-------------------------------------------------')
#         print(print(f'    Answer: {response.generations[0][0].text}'))
#         print('-------------------------------------------------')
#         return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)    


class ToolsAgent:
    def __init__(self, verbose=False):
        self.verbose = verbose # to activate verbose

        # define instructions and troubleshooting RAG agents
        self.trlb_agent = TroubleshootingRagAgent()
        self.instr_agent = InstructionsRagAgent()

        # tools list (instructions and troubleshooting agents)
        self.tools = [self.instructions_agent, self.troubleshooting_agent]
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
        output = self.tools_agent.invoke({'input': input})
        
        if output=='instructions_agent':
            self.instructions_agent.invoke(input={'input': question})
            if self.verbose:
                print('\nThe Tools Agent chose the Instruction Rag Agent')
        else:
            self.troubleshooting_agent.invoke(input={'input': question})
            if self.verbose:
                print('\nThe Tools Agent chose the Troubleshooting Rag Agent')

        return output
    
    @tool
    def instructions_agent(self, input):
        '''Given a question, returns information about the description of the product, the packet content, its box content, installation steps, main features and maintenance'''
        return self.instr_agent.invoke_agent(input, verbose=self.verbose)

    @tool
    def troubleshooting_agent(self, input):
        '''Given a question, returns information about the Troubleshooting Guide, including Wi-Fi connection issues, inaccurate water usage data, leak detection false alarms, system offline, automated watering not working'''
        return self.trlb_agent.invoke_agent(input, verbose=self.verbose)
    

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

        # llm and callback definition
        self.llm = ChatOllama(model='llama3.1', temperature=0.8)

        # definition of rag agent
        self.agent = input_dic | self.promt_template | self.llm | StrOutputParser()
    
    def invoke_agent(self, question, verbose):
        # get retriever context from question
        docs = self.retriever.invoke(question)
        context =  "\n\n".join([doc.page_content for doc in docs])

        if verbose:
            print(f'\nInformation retrieved by the Instruction Rag Agent:\n{context}')

        # get llm response considering the retrieved information
        output = self.agent.invoke({'context': context, 'question': question})

        return output
        

class TroubleshootingRagAgent:
    def __init__(self):
        # create embedding
        self.embedding = OllamaEmbeddings(model='llama3.1')

        # load information from vector store and create retriever
        self.vectorstore = FAISS.load_local('vector_stores/troubleshooting', self.embedding, allow_dangerous_deserialization=True)
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

        # llm and callback definition
        self.llm = ChatOllama(model='llama3.1', temperature=0.8)

        # definition of rag agent
        self.agent = input_dic | self.promt_template | self.llm | StrOutputParser()
    
    def invoke_agent(self, question, verbose):
        # get retriever context from question
        docs = self.retriever.invoke(question)
        context =  "\n\n".join([doc.page_content for doc in docs])
        
        if verbose:
            print(f'\nInformation retrieved by the Instruction Rag Agent:\n{context}')

        # get llm response considering the retrieved information
        output = self.agent.invoke({'context': context, 'question': question})
        return output



if __name__ == '__main__':
    # test = ToolsAgent()
    # output = test.invoke_agent('I have problems with the Wi-Fi connection')
    # print(output)

    #question = 'What are the installation steps to connect the main unit?'
    question = 'My device is not connecting to the Wi-Fi, what should I do?'
    print(f'\n-------QUESTION-------\n{question}')
    # test = TroubleshootingRagAgent()
    # anws = test.invoke_agent(question)
    
    test = ToolsAgent(verbose=True)
    answer = test.invoke_agent(question)
    print(f'\n-------ANSWER-------\n{answer}')