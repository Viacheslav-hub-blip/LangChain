from typing import Annotated
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

model_repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=model_repo_id,
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.8,
    task="text-generation",
    max_new_tokens=1000,
    do_sample=False,
)

chat_model = ChatHuggingFace(llm=llm)


@tool
def multiply(
        a: Annotated[int, "first number"],
        b: Annotated[int, "second number"]
) -> int:
    """ Multiply a and b"""
    return a * b


chat_model_with_tools = chat_model.bind_tools([multiply])

query = "What is 3 * 12?"

messages = [HumanMessage(query)]

ai_msg = chat_model_with_tools.invoke(query)
print(ai_msg.tool_calls)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    selected_tool = {"multiply": multiply}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

print(messages)
print(chat_model_with_tools.invoke(messages))
