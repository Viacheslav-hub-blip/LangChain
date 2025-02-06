"""
Векторные хранилища  - специальные хранилища, которые помогают
получать информацию на основе векторных представлений

Эти векторы называются Embeddings и представляют собой числовое прдставление
семнатики данных

Их часто используют для поиска по неструктирированным данным, таким как текст, изобрадения и тд

LnagChain поддерживает большое количество векторных хранилищ с различными реализациями
и предоставляет стандартный интерфейс для работы с ними, позволяющий легко менять хранилища

Оснонвые методы:
1. add_documents - добавление списка текстов в хранилище
2. delete_documents  - удалние документов
3. similarity_search  - посик документов похожих на заданный запрос

Большинстов хранилищ в LangChain принимают embedding model
в каечестве аргумента при инициализации

embedding model мы использовали в Prompts(HuggingFaceEndpointEmbeddings)
модель преобразования текста в векторное прдстваление для конкретной модели
"""

from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

# загрзука embeddings models
embeddings = HuggingFaceEndpointEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1", task="feature-extraction")

# пример 1, используем  InMemoryVectorStore, используется косинусное подобие для поиска
vectorstore = InMemoryVectorStore(embedding=embeddings)

# для добавления документа используется add_documents.
# работает со списком обьектов класса Document
document_1 = Document(
    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"}
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

documents = [document_1, document_2]

# Способ 1
vectorstore.add_documents(documents=documents)

# Способ 2
# можно указать ID для документов, в будущем будет удобно удалять или обновлять документы используя эти id
vectorstore.add_documents(documents, ids=["doc1", "doc2"])

# получить документ по id
print(vectorstore.get_by_ids(["doc1"]))

vectorstore.delete(ids=["doc1"])

# получить документ по id
print(vectorstore.get_by_ids(["doc1"]))  # [] так как такого уже нет

# ПОИСК
"""Векторные хранилища внедряют и сохраняют добавленные документы. 
Если мы передадим запрос, vectorstore внедрит запрос, выполнит поиск сходства 
по внедренным документам и вернет наиболее похожие. Это отражает две важные концепции: 
во-первых, должен быть способ измерить сходство между запросом и любым встроенным документом.
 Во-вторых, должен существовать алгоритм для эффективного выполнения поиска сходства по всем 
 встроенным документам.
 
Способы сравнения векторов:

Косинусное подобие: измеряет косинус угла между двумя векторами.
Евклидово расстояние: измеряет расстояние по прямой между двумя точками.
Скалярное произведение: измеряет проекцию одного вектора на другой.


Для посика документа используется метод similarity_search
При этом будет создан embedding найден схожий документ, который
будет возвращен как писок Documents
 """

# пример 1
query = "scrambled eggs for breakfast"
docs = vectorstore.similarity_search(query)
print(docs)

"""
Многие хранилища поддерживают дополнительные параметры поиска
k  - количество возвращаемых документов
filter  - фильтр документов по metadata
"""


def filter_function(doc: Document) -> bool:
    print('source', doc.metadata.get("source"))
    return doc.metadata.get("source") == "tweet"


print(vectorstore.similarity_search(
    "scrambled eggs for breakfast",
    k=2,
    filter=filter_function
))

# Search with score
results = vectorstore.similarity_search_with_score(
    query="scrambled eggs for breakfast",
    k=2,
)

for doc, score in results:
    print(score, doc.page_content)

# use a Retriever
# fetch_k - количество документов, которые будут переданы в алгоритм
# k - количество возвращаемых документов

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
)
print(retriever.invoke("scrambled eggs for breakfast"))
