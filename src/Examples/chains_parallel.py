"""

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Мы просим модель написать основные плючы и минусы какой нибудь вещи, исходя из ее функционала

На первом шаге модель генерирует текст, в котором описывает основные возможности (features)

Затем модель генерирует плюсы и минусы исходя из сгнерированного ранее текста

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace

huggingface_token = "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"
model_repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=model_repo_id,
    temperature=0.8,
    task="text-generation",
    max_new_tokens=1000,
    do_sample=False,
)

llm_chat = ChatHuggingFace(llm=llm)

model = llm_chat

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        ("human", "List the main features of the product {product_name}.")
    ]
)


def analyze_pros(features):
    pros_templates = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the pros of these features.",
            )
        ]
    )
    return pros_templates.format_prompt(features=features)


def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the cons of these features.",
            ),
        ]
    )
    return cons_template.format_prompt(features=features)


def combine_pros_cons(pros, cons):
    return f"Pros:\n{pros}\n\nCons:\n{cons}"


pros_branch_chain = (
        RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
        RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

chain = (
        prompt_template
        | model
        | StrOutputParser()
        | RunnableLambda(lambda x: print(x))
        | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
        | RunnableLambda(lambda x: combine_pros_cons(x['branches']['pros'], x['branches']['cons']))
)

result = chain.invoke({"product_name": "MacBook Pro"})
print(result)
