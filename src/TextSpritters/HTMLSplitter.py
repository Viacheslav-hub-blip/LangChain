"""
LangChain предоставляет 3 разделителя для HTML

КАЖДЫЙ ИЗ НИХ ТАК ЖЕ ДОБАВЛЯЕТ METADATES (ОПИСАНИЕ ПАРАГРАФА)
1. HTMLHeaderTextSplitter
-используется,когда нужно сохранить иерархичную структуру, основанную на заголовках
Разделение основано на тэгах заголовков (h1, h2)

2.HTMLSectionSplitter
-используется, когда необходимо разделить документ на разделы
(div, section)

3.HTMLSemanticPreservingSplitter
-используется для сохранения контекстуальную структуру

необходимо установить
pip install lxml
"""


from langchain_text_splitters import HTMLSectionSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter
# используем HTMLHeaderTextSplitter

html_content = """
<html>
  <body>
    <h1>Introduction</h1>
    <p>Welcome to the introduction section.</p>
    
    <h2>Background</h2>
    <p>Some background details here.</p>
    
    <h3>Conclusion</h3>
    <p>Final thoughts.</p>
    <div> div block </div>
  </body>
</html>
"""

# указываем заголовки на которые следует разбивать
headers_to_split_on = [
    ("h1", "Header 1"),
    ("div", "Header 3"),
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on)
splits = splitter.split_text(html_content)

for split in splits:
    print(split)

splitter = HTMLHeaderTextSplitter(headers_to_split_on, return_each_element=True)
splits = splitter.split_text(html_content)

for split in splits:
    print(split)

# HTMLSectionSplitter необходимо установить bs4
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("p", "Header 2"),
    ("div", "Header 2"),
]

html_splitter = HTMLSectionSplitter(headers_to_split_on)
html_header_splits = html_splitter.split_html_by_headers(html_content)

print(html_header_splits)
print("----------------------------")

# Документ разбит четко по секциям

# Using HTMLSemanticPreservingSplitter
"""предназначен для разделения контента, например, списки
таблицы, которые могут разбиты на несколько элементов при использовании 
других разделителей 

Даже при устанолвенном max_chunk_size фрагменты могут быть большего размера, чтобы 
не потреять смысл фрагмента
"""