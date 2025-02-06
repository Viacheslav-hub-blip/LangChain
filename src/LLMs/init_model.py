"""
BaseLanguageModel  - базовый класс для всех языковых моделей, который
реализует интерфейс Runnable. Плэтому он поддерживает invoke, batch, assign и др

Стандартные парметры при инициализации модели:

model  - название или id модели, которую будем использовать
temperature - температура моедли/показатель креативности
timeout  - максимально время ожидания ответа модели
max_tokens
stop - специальный токен, который будет означать что модель должна остановиться

"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace

# пример использования модели с HuggingFace
huggingface_token = "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"
model_repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

"""
Два способа использовать модель с HuggingFace:
1.HuggingFacePipeline - загружает модель в кэш устройства и
использовать аппаратное обеспечение вашего компьютера

2. HuggingFaceEndpoint  - использует serverless API, для этого
необходимо создать аккаунт и созать huggingface_token

"""

"""
дополнительно можно указать:
top_k=10,
top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03, # параметр для штрафа за повторение
"""

llm = HuggingFaceEndpoint(
    repo_id=model_repo_id,
    temperature=0.8,
    task="text-generation",
    max_new_tokens=1000,
    do_sample=False,
)

# llm = HuggingFacePipeline.from_model_id(
#     model_id=model_repo_id,
#     task="text-generation",
#     pipeline_kwargs={
#         "max_new_tokens": 100,
#         "top_k": 50,
#         "temperature": 0.1,
#     }
# )

# print(llm.invoke('Who is Elon Musk?'))

# есйчас вопрос модели подается в исзодном виде, но это
# не совсем правильно, так как лучше использовать
# специальные токены, которые понятны модели

llm_chat = ChatHuggingFace(llm=llm)

# использование сообщений
model = llm_chat

prompt = ChatPromptTemplate.from_messages([
    ("system", "You should only give answers in Spanish."),
    ("user", "Hello, how are you?")
])

chain = prompt | model
print('chain', chain.invoke({}))

# print(model.invoke([HumanMessage(content="Hello, how are you?").content]))
print(
    model.invoke([SystemMessage(content="You must get answer on Spanish"), HumanMessage(content="Hello, how are you")]))
