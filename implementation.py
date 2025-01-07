from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.agents import Tool # tool decorator
from langchain.tools.render import render_text_description # render the description of each tool to include in the prompt
from langchain_core.output_parsers.string import StrOutputParser
from langchain_community.vectorstores import FAISS # to implement vector stores


class ToolsAgent:
    def __init__(self, verbose=False):
        self.verbose = verbose # to activate verbose

        # define rag agents
        self.instr_agent = InstructionsRagAgent()
        self.trbl_agent = TroubleshootingRagAgent()

        # tools list (instructions and troubleshooting agents)
        #self.tools = [instructions_agent, troubleshooting_agent]
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
    
    def invoke_instructions_agent(self, question):
        '''Given a question, returns information about the description of the product, the packet content, its box content, installation steps, main features and maintenance'''
        return self.instr_agent.invoke_agent(question, verbose=self.verbose)

    def invoke_troubleshooting_agent(self, question):
        '''Given a question, returns information about the Troubleshooting Guide, including Wi-Fi connection issues, inaccurate water usage data, leak detection false alarms, system offline, automated watering not working'''
        return self.trbl_agent.invoke_agent(question, verbose=self.verbose)
    

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
        self.llm = ChatOllama(model='llama3.1', temperature=0.6)

        # definition of rag agent
        self.agent = input_dic | self.promt_template | self.llm | StrOutputParser()
    
    def invoke_agent(self, question, verbose=True):
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
        self.llm = ChatOllama(model='llama3.1', temperature=0.6)

        # definition of rag agent
        self.agent = input_dic | self.promt_template | self.llm | StrOutputParser()
    
    def invoke_agent(self, question, verbose=True):
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

    question = 'What are the installation steps to connect the main unit?'
    #question = 'What does the box contains?'
    #question = 'My device is not connecting to the Wi-Fi, what should I do?'
    print(f'\n-------QUESTION-------\n{question}')
    # test = TroubleshootingRagAgent()
    # anws = test.invoke_agent(question)

    test = ToolsAgent(verbose=True)
    answer = test.invoke_agent(question)
    print(f'\n-------ANSWER-------\n{answer}')


    # test = ToolsAgent(verbose=True)
    # answer = test.invoke_agent(question)
    # print(f'\n-------ANSWER-------\n{answer}')

    # define all agents
    # tools_agent = ToolsAgent()
    # instr_agent = InstructionsRagAgent()
    # trbl_agent = TroubleshootingRagAgent()

    # define the tools for the tools agent
    # @tool
    # def instructions_agent(input:str) -> str:
    #     '''Given a question, returns information about the description of the product, the packet content, its box content, installation steps, main features and maintenance'''
    #     return instr_agent.invoke_agent(input)

    # @tool
    # def troubleshooting_agent(input:str) -> str:
    #     '''Given a question, returns information about the Troubleshooting Guide, including Wi-Fi connection issues, inaccurate water usage data, leak detection false alarms, system offline, automated watering not working'''
    #     return trbl_agent.invoke_agent(input)

    # given an input query, ask for the right agent
    # output = tools_agent.invoke_agent(question)
    
    # invoke the right agent
    # if output=='instructions_agent':
    #     answer = instructions_agent(question)
    # else:
    #     answer = troubleshooting_agent(question)
    # print(f'\n-------ANSWER-------\n{answer}')





    # if output=='instructions_agent':
    #         self.instructions_agent.invoke(input={'input': question})
    #         if self.verbose:
    #             print('\nThe Tools Agent chose the Instruction Rag Agent')
    #     else:
    #         self.troubleshooting_agent.invoke(input={'input': question})
    #         if self.verbose:
    #             print('\nThe Tools Agent chose the Troubleshooting Rag Agent')