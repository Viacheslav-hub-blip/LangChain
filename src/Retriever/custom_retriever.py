"""
 retrieval systems - посиковые системы
 к ним относятся уже изученные векторные хранилища, графичекские базы баднных
 и обычные реляционные базы данных

 для создания собсвенной системы посика необходимо создать класс, унаследовать
 его от BaseRetriever  и реализовать методы
 _get_relevant_documents

логика внутри _get_relevant_documents может включать в сбея что угодно, запросы к базам
данных или к интернет ресурсам

СУТЬ retrieval В ТОМ ЧТО ОН КАК ЛИБО ОБРАЗОМ МОЖЕТ ИСКАТЬ ИНФОРМАИЮ
В РАЗЛИЧНЫХ ИСТОЧНИКАХ В ТОМ ЧМСЛЕ В ВЕКТОРНЫХ БАЗАХ И + ДОБАВЛЯТЬ
СВОЙ ФУНКЦИОНАЛ

"""

from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class MyRetriever(BaseRetriever):
    """
    Будем возвращать k документов, которые содержат в себе
    запрос
    """
    documents: List[Document]
    k: int

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        matching_documents = []

        for document in self.documents:
            if len(matching_documents) > self.k:
                return matching_documents

            if query.lower() in document.page_content.lower():
                matching_documents.append(document)
        return matching_documents


documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"type": "dog", "trait": "loyalty"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"type": "cat", "trait": "independence"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"type": "fish", "trait": "low maintenance"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"type": "bird", "trait": "intelligence"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"type": "rabbit", "trait": "social"},
    ),
]
retriever = MyRetriever(documents=documents, k=3)

#поддерживает все возможности Runnable
print(retriever.invoke("--------------"))

print(retriever.batch(["dog", "cat"]))