from langchain_groq import ChatGroq
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_community.tools import WikipediaQueryRun, TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper
import os
from langchain import hub

embeddings = HuggingFaceEndpointEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1", task="feature-extraction")

GROQ_API_KEY = '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'
TAVILY_API_KEY = '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'

os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

llm = ChatGroq(model_name='Gemma2-9b-It', api_key=GROQ_API_KEY)
if __name__ == "__main__":
    # простой помощник

    # while True:
    #     question = input()
    #     if question != 'q':
    #         print(llm.invoke(question).content)
    #     else:
    #         break


    # простой помощник с памятью

    # store = {}
    #
    #
    # def get_session_history(session_id: str) -> BaseChatMessageHistory:
    #     if session_id not in store:
    #         store[session_id] = InMemoryChatMessageHistory()
    #     return store[session_id]
    #
    #
    # config = {"configurable": {"session_id": "firstchat"}}
    # model_with_memory = RunnableWithMessageHistory(llm, get_session_history)
    #
    # while True:
    #     question = input()
    #     print(model_with_memory.invoke(question, config=config).content)
    #     print(store)


    # простой RAG
    # docs = TextLoader('docs.txt').load()
    # print(docs)
    #
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=50,
    #     chunk_overlap=10,
    #     length_function=len
    # )
    #
    # new_docs = text_splitter.split_documents(docs)
    # doc_strings = [doc.page_content for doc in new_docs]
    # print(doc_strings)
    #
    # db = Chroma.from_documents(new_docs, embeddings)
    # retriever = db.as_retriever()
    #
    # template = """
    # Answer the question based only on the following context:
    # {context}
    #
    # Question: {question}
    # """
    #
    # prompt = PromptTemplate.from_template(template)
    #
    # chain = (
    #         RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    #         | prompt
    #
    #         | llm
    #         | StrOutputParser()
    # )
    #
    # print(chain.invoke("types of schools in the English"))


    # простые инструменты, википедия

    # api_wrapper = WikipediaAPIWrapper()
    # tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    # print(tool.name, tool.description)
    #
    # from langchain.agents import AgentType, load_tools, initialize_agent
    #
    # tool = load_tools(['wikipedia'], llm=llm)
    #
    # agent = initialize_agent(
    #     tool,
    #     llm,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True
    # )
    #
    # print(agent.run('what is Apple?'))


    # свои простые функции
    from langchain_core.tools import tool
    from langchain.agents import load_tools


    @tool
    def get_word_count(word: str) -> int:
        """Returns the length of a word"""
        return len(word)


    @tool
    def multiply(a: int, b: int) -> int:
        """Multiplies two numbers"""
        return a * b


    search = TavilySearchResults()
    # tools = [search]
    # print(search.invoke("what is whe weather in SF?"))

    prompt = hub.pull("hwchase17/openai-functions-agent")
    # print(prompt.messages)

    from langchain.agents import create_tool_calling_agent
    from langchain.agents import AgentExecutor

    # agent = create_tool_calling_agent(llm, tools, prompt)
    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    #
    # result = agent_executor.invoke({"input": "hellow, what is weather is SF?"})
    # print(result)
    # появляется ненужный вызов функции когда модель может ответить сама


    # добавляем инстурмент
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = WebBaseLoader("https://blog.langchain.dev/langgraph-studio-the-first-agent-ide/")
    docs = loader.load()
    # print(docs)

    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)

    vectore_db = FAISS.from_documents(documents, embeddings)
    retriver = vectore_db.as_retriever()

    # print(retriver.invoke("what is laggraph studio?"))

    from langchain.tools.retriever import create_retriever_tool

    retriever_tool = create_retriever_tool(
        retriver,
        "langgraph_search",
        "Search for information about LangGraph. For any question about LangGraph you must use this tool",
    )

    tools = [search, retriever_tool, multiply]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke({"input": "what is 398 * 169?"})
    print(result)


