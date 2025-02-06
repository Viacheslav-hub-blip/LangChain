"""
Использовать Chroma можно без создания дополнительных
учетных записей
"""

from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEndpointEmbeddings(model="mixedbread-ai/mxbai-embed-large-v1", task="feature-extraction")

store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
    id=2,
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
    id=3,
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
    id=4,
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
    id=5,
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
    id=6,
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
    id=7,
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
    id=8,
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
    id=9,
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
    id=10,
)

document_11 = Document(
    page_content="I had chocalate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"}
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
    document_11
]

uuids = [str(uuid4()) for _ in range(len(documents))]

store.add_documents(documents=documents, ids=uuids)

print(uuids)
# #обновление документов
# updated_document_1 = Document(
#     page_content="I had chocolate chip pancakes and fried eggs for breakfast this morning.",
#     metadata={"source": "tweet"},
#     id=1,
# )
#
# updated_document_2 = Document(
#     page_content="The weather forecast for tomorrow is sunny and warm, with a high of 82 degrees.",
#     metadata={"source": "news"},
#     id=2,
# )
#
# store.update_document(document_id=uuids[0], document=updated_document_1)
# # You can also update multiple documents at once
# store.update_documents(
#     ids=uuids[:2], documents=[updated_document_1, updated_document_2]
# )

print('последний пример')
retriever = store.as_retriever(
    search_type="mmr", search_kwargs={"k": 1}
)
print(retriever.invoke("scrambled eggs for breakfast"))
print()