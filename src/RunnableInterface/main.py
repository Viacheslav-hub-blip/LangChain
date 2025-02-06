# интерфейс Runnable - основа для работы с компонентами  LangChain
"""
Его реализуют все основные сущности с котромы приходится работать в LangChain(LLM,
Output Parsers, retrievers и т.д

Основные методы, который предостаавляет этот интерфейс:

ainvoke - асинхронный вариант
invoke/ainvoke: преобразует одиночный входной сигнал в выходной
batch/abatch: преобразует множество входных данных в выходные
stream/astream: потоковая передача выходных данных данных с одного входного сигнала

Все эти методы поддерживают оптциональныйьаргумент - RunnableConfig
Он представялет собой словарь, содержащий настройки для Runnable, которые будут использоватьс
во время выполнения

Для обьединения нескольких свущностей, реализующих Runnable в одну цепочку
используется специальный язык выражений LangChain - LCEL(LangChain Expression Language)

Цкпочка тоже реализует Runnable

Преимущества LCEL:
поддержка асинхронного режима. Любая цепочка, созданная с помошью
LCEL поддерживает асинхронное использование

цепочки LCEL можно передавать в потоковом режиме, что позволяет получать
дополнительные выводы по мере выполнения этапов цепочки

Цепочки создаются путем обьединения существующих Runnable в два основных примитиваL
RunnableSequence и RunnableParallel

RunnableSequence - примитив, который позволяет обьединять сущности в последовательную
цепочку, где выходные данные одного элемента цепочки служат входными данными для следующего

"""

# пример создания Runnable и использования RunnableSequence
from langchain_core.runnables import RunnableLambda, RunnableSequence

runnable1 = RunnableLambda(lambda x: x + 1)
runnable2 = RunnableLambda(lambda x: x + 2)
chain = RunnableSequence(runnable1, runnable2)

print(chain.invoke(2))  # 5

# можно переписать без использования RunnableSequence
output1 = runnable1.invoke(1)
output2 = runnable2.invoke(output1)

print(output2)  # 4

# перепишем с использования LCEL
chain = runnable1 | runnable2
print(chain.invoke(2))  # 5

# вместо | можно использовать .pipe
chain = runnable1.pipe(runnable2)
print(chain.invoke(2))  # 5

# Пример использования RunnableParallel
# запускает несколько Runnable с одинаковыми значениями

from langchain_core.runnables import RunnableParallel

chain = RunnableParallel({
    "key1": runnable1,
    "key2": runnable2,
})
print(chain.invoke(2))  # {'key1': 3, 'key2': 4}

# комбинированное использование
# словарь автоматически преобразуется в RunnableParallel
# на выходе снова получаем словарь
mapping = {
    "key1": runnable1,
    "key2": runnable2,
}

runnable3 = RunnableLambda(lambda x: x['key1'] + x['key2'])

chain = mapping | runnable3

print(chain.invoke(2))  # 7

# можно переписать
chain = RunnableSequence(RunnableParallel(mapping), runnable3)
print(chain.invoke(2))  # 7


# Внутри LCEL функкция автоматически преобразуется в RunnableLambda
def some_func(x):
    return x


chain = some_func | runnable1
print(chain.invoke(2))  # 3

# -------------------------------
# возвращение к Runnable
"""
помимо invoke Runnable поддерживает вызов с помошью batch
"""
chain = runnable1 | runnable2
print(chain.batch([1, 2, 3]))  # [4, 5, 6]

batch_runnable = RunnableLambda(lambda x: str(x + 1))
print('batch', batch_runnable.batch([2, 3, 4]))


# использование генератора вместе с batch

def func(x):
    for y in x:
        yield str(y)*2


runnable_gen = RunnableLambda(func)
for chunk in runnable_gen.stream(range(5)):
    print('chunk', chunk)

# -------------------------------
# пример использования дополнительных настроек

import random


def add_one(x: int) -> int:
    return x + 1


def buggy_double(x: int) -> int:
    if random.random() > 0.0001:
        print('Code failed')
        raise ValueError('bad value')
    return x * 2


def failed_func(x: int):
    return x * 2


chain = RunnableLambda(add_one) | RunnableLambda(buggy_double).with_retry(
    stop_after_attempt=10,
    wait_exponential_jitter=False  # следует ли добавлять задержку между потоврными вызовами
).with_fallbacks([RunnableLambda(failed_func)])
# Если спустя 10 попыток не будет достигнуто нужное значение, закончит с ошибкой.

print()
print(chain.invoke(2))

# -------------------------------
# RunnablePassthrough
# позволяет использовать входные данные в цепочке без изменений
from langchain_core.runnables import RunnablePassthrough

runnable = RunnableParallel(
    origin=RunnablePassthrough(),
    modified=lambda x: x + 1
)

print(runnable.batch([1, 2, 3]))


# [{'origin': 1, 'modified': 2}, {'origin': 2, 'modified': 3}, {'origin': 3, 'modified': 4}]

def fake_llm(prompt: str) -> dict:
    return {'origin': prompt, 'answer': 'complete'}


chain = RunnableLambda(fake_llm) | {
    'orig': RunnablePassthrough(),
    "parsed": lambda text: text['answer'][::-1]
}

print(chain.invoke('hello'))


# пример 2
def fake_llm(prompt: str) -> str:
    return "complete"


runnable = {'llm1': fake_llm,
            'llm2': fake_llm,
            } | RunnablePassthrough.assign(
    total_chars=lambda inputs: len(inputs['llm1'] + inputs['llm2'])
)
"""RunnalbePasstrough реализует интерфейс Runnable, поэтому поддерживает 
дополнительные методы такие как with_retry, asising, bind и др

assing позвояет добавить новое поле в выходной словарь
С bind познакомимся позже. 
"""
print(runnable.invoke('hello'))
