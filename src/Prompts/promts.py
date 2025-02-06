"""
Промпты принимают в качестве входных данных словарь, где каждый ключ
представялет собой переменную в шаблоне, которую нужно заполнить

Существует несколько видов Prompt Templates:

1.String PromptTemplates
-используются для форматирования одной отдельной строки

2.ChatPromptTemplates
-используются для форматирования нескольких сообщений

3.MessagesPlaceholder
-позволяет втсавить список сообщений в определенное места в ChatPromptTemplates

Все они реализуют интерфейс Runnable поэтому поддерживают знакомые функции
"""


from langchain_core.prompts import PromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage
from src.LLMs.init_model import model
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma

prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")
print(prompt_template.invoke({"topic": "cats"}))
# подставит переменную. Получим:text='Tell me a joke about cats'

# пример ChatPromptTemplates
prompt_template = ChatPromptTemplate([
    ("system", "You should only give answers in Spanish."),
    ("user", "Tell me a joke about {topic}")
])
print('Caht', prompt_template.invoke({"topic": "cats"}))

# пример MessagesPlaceholder
prompt_template = ChatPromptTemplate([
    ("system", "You must get answers on Spanish"),
    MessagesPlaceholder("msgs")
])

print('Holder', prompt_template.invoke({'msgs': [HumanMessage(content='Hi'), HumanMessage(content="Hello")]}))
print()

prompt = MessagesPlaceholder("history")
prompt = prompt.format_messages(
    history=[
        ("system", "You must get answers on Spanish"),
        ("human", "Hello")
    ]
)
print('history prompt', prompt)

# пример 3
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You must get answers on Spanish"),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ]
)

print('full history', prompt.invoke(
    {
        "history": [('human', "what is 5 +2?"), ("ai", "5+2 is 7")],
        "question": "now now multiply that by 4"
    }
))

# FewShotChatMessagePromptTemplate
print()
examples = [
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

print(few_shot_prompt.invoke({}))

# пример 2
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

chain = final_prompt | model
print(chain.invoke({"input": "What is 2 🦜 9?"}))

# Динамический few-shot prompting
"""
Мы можем иметь набор с большим количеством примеров работы, 
но не всегда нам нужно использовать их все. 
Поэтому мы можем выбирать примеры, которые больше всего подходят
под наш запрос на основе семантического сходства

Chroma  - векторная база данных с откртым исходным кодом

#install langchain_chroma 
"""
embeddings = HuggingFaceEndpointEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1", task="feature-extraction")
examples = [
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
    {"input": "2 🦜 4", "output": "6"},
    {"input": "7 🦜 5", "output": "12"},
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},
    {
        "input": "Write me a poem about the moon",
        "output": "One for the moon, and one for me, who are we to talk about the moon?",
    },
    {
        "input": "horse",
        "output": "small horse"
    }
]

to_vectorize = [" ".join(example.values()) for example in examples]
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)

# k - количество ближайших примеров, которые будут использоваться
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2
)
print(example_selector.select_examples({"input": "horse"}))

# создаем prompt
few_shot_prompt = FewShotChatMessagePromptTemplate(
    input_variables=["input"],
    example_selector=example_selector,

    # определяем как будет отформатирован каждый пример
    # в нашем случае 1 сообщение пользователя, 1 ответ ai

    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)

print(few_shot_prompt.invoke(input="What is 3 🦜 3").to_messages())

chain = few_shot_prompt | model
print('answer', chain.invoke(input={"input": "what is the result of the expression 5 🦜 7 = ?"}))

# этот prompt можно добавить к другим
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)
chain = final_prompt | model
print(chain.invoke({"input": "What's 3 🦜 3?"}))  # ебааааать, удалось
