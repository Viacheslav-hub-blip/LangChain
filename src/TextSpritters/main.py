"""
Разбиение документа  - важный этап предварительной обрабоки данных
Он включает в себя разбиение больших текстов на более маленькие

Существует несколько стратегий разделения документов

1.Length-base
-самый простой подход заключается в разделениии документов
в зависимости от их длины. Это гарантирует, что каждый фрагмент не превысит
заданного ограничения длины

В этом подходе различают два вида:
-разбиение на основе токенов
-разделение на основе символов
"""

# пример использования сплитера на символах

# """
# Разбивает текст на основе заданной последовадельности символов, по умолчанию
# она равна "\n\n", Длина блока (chunk) Измеряется количеством символом
# """
from langchain_text_splitters import CharacterTextSplitter
from transformers import GPT2TokenizerFast
from transformers import AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("docs.txt") as f:
    text_union = f.read()

text_splitter = CharacterTextSplitter(
    separator=r'\d',
    chunk_size=100,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=True
)

texts = text_splitter.create_documents([text_union])
for text in texts:
    print(len(text.page_content), text)
    print("---------------------------------")

metadatas = [{"document": 1}, {"document": 2}, {"document": 2}]
documents = text_splitter.create_documents(
    [text_union, text_union], metadatas=metadatas
)
print(documents)

# исполбзование Hugging Face tokenizer

model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

with open("docs.txt") as f:
    state_of_the_union = f.read()
# print(state_of_the_union)

text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(state_of_the_union)
print('Всего фрагментов', len(texts))
for text in texts:
    print(len(text), text)
    print("------------------------------")
    print()

# Текст разделился, но есть фрагменты, которые больше установленной максимальной длины
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=100, chunk_overlap=0
)

texts = text_splitter.split_text(state_of_the_union)
print('всего текстов', len(texts))
for text in texts:
    print(len(text), text)
    print("------------------------------")
    print()

# с использованием рекурсирвного сплитера тексты все еще больше 100 симловов, но вместо 5 чанков получили 9. Размер
# одного фрагмента стал меньше

# """
# Помимо этих спилтеров также существуют:
# -tiktoken
# -spaCy
# -SentenceTransformers
# -NLTK
# -KoNLPY
#
# больше информации:
# https://python.langchain.com/docs/how_to/split_by_token/
# """
#
