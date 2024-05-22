from googlesearch import search
import requests
from bs4 import BeautifulSoup
import html2text
from typing import List, Dict
import re
from vectorstores import TableVectorStore
from textsplitters import CodeSplitter


class UrlParser:

    def __init__(self, extension):
        
        self.splitter = CodeSplitter(
                                    extension=extension,
                                    chunk_size=156,
                                    chunk_overlap=0
                                    )
    

    def get_website_contents(self, url: str) -> str:
    
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        h = html2text.HTML2Text()
        markdown = h.handle(soup.prettify())
        lines = markdown.split('\n')
        new_lines = []
        for l in lines:
            if l.strip() != '' and l.strip()[0].isalpha() and re.search('[a-zA-Z]', l) is not None:
                new_lines.append(l)
        processed_result = '\n'.join(new_lines)
        return processed_result

    def google_search_worker(self, query: str, num_docs) -> List[str]:
        results = []
        try:
            for url in search(query, tld="co.in", num=num_docs, stop=num_docs):
                try:
                    search_contents = self.get_website_contents(url)
                    assert isinstance(search_contents,str)
                    results.append(search_contents)
                except:
                    pass
        except Exception as e:
            print(e)
        return results
    
    def split_data(self, data: List[str]) -> List[str]:

        result = []
        for text in data:
            result += self.splitter.split_text(text)

        return result
    
    def __call__(self, query: str, 
                 search_type: str = 'similarity', 
                 k: int = 3,
                 ) -> List[str]:
        
        search_results = self.google_search_worker(query, k-1)
        data = self.split_data(search_results)
        vectorstore = TableVectorStore.from_texts(texts=data)
        retriever = vectorstore.as_retriever(search_kwargs={'k': k}, search_type=search_type)
        docs = retriever._get_relevant_documents(query)
        return docs