from langchain import hub

from test import llm, embeddings
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.tools import tool
import os
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, Tool

google_api_key = '4f41f5bbb734b775757e3a31d47be2e7c438cffc'
os.environ['SERPER_API_KEY'] = google_api_key

google_search = GoogleSerperAPIWrapper()

tools = [
    Tool(
        name="Intermediate Answer",
        func=google_search.run,
        description="useful for when you need to ask with search",
        verbose=True  # будем получать ифнормацию о вызове
    )
]

template = """
Answer the following question as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Though: you should always think about what to do
Action: the action to take, should be one of the [{tool_names}]
Action Input: the input of the action
Observation: the result of the action

Though: I now know the final answer
Final Answer: the final answer to the original input question

Question:{input}
Though: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

search_agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=search_agent, tools=tools, verbose=True, return_intermediate_steps=True)


# result = agent_executor.invoke(
#     {"input": "who was the first president of Russian?"})
#
# print(result)


# добавим свой инструмент

@tool
def get_employee_id(name: str):
    """
    To get Employee id, it takes employee name and returns employee id.
    """

    fake_employees = {
        "Slava": "1",
        "Alice": "2",
        "Evan": "3",
        "Ian": "4"
    }
    return fake_employees.get(name, "Employee not found")


@tool
def get_employee_salary_by_id(employee_id: str):
    """
    To get the salary of employee, it takes employee id and returns salary of employee.
    """

    employee_salaries = {
        "1": "1000",
        "2": "1500",
        "3": "1400",
        "4": "2000",
    }

    return employee_salaries.get(employee_id, "Salary not found")


prompt = hub.pull("hwchase17/react")
print(prompt.template)

tools = [get_employee_id, get_employee_salary_by_id]

agent = create_react_agent(llm, tools, prompt)
agent_executor_2 = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = agent_executor_2.invoke({"input": "hi, how are you?"})

print(result)

