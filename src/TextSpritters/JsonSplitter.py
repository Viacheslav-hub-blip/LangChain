"""
Некоторые документы имеют структуру, например, JSON, HTML, CODE

JsonSplitter разделяет данные json, позволяя контролировать размеры блоков.
Сначала он просматривает грубину данных и создает меньшие блоки json.
Он пытается сохранить обьекты json целыми, но при необходимости разделяет их
чтобы уместить обьекты в максимальный размер фрагмента
"""
import requests
import json
from langchain_text_splitters import RecursiveJsonSplitter

# with open("test.jsonl") as file:
#     data = json.load(file)
#
# print(data)

data = requests.get("https://jsonplaceholder.typicode.com/users").json()

splitter = RecursiveJsonSplitter(max_chunk_size=600)

json_chunks = splitter.split_json(data, convert_lists=True)

print('всего фрагментов', len(json_chunks))

for chunk in json_chunks:
    print(chunk)

docs = splitter.create_documents(texts=[data], convert_lists=True)
for doc in docs:
    print(doc)
