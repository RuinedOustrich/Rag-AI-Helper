from typing import List
from textsplitters.base import Language
from files.file_utils import extract_content_from_files, get_extension_files
import ast
import astunparse

class DirectoryParser:
    def __init__(self, extension: str, path: str):

        self.extension = extension
        self.path = path
        self.files = get_extension_files(path=self.path, extension=self.extension)

    def parse_file_names(self, query: str) -> List[str]:

        contexts = []
        for file in self.files:
            if file in query:
                contexts.append(extract_content_from_files(files = [file], 
                                                           extension = self.extension)[0], 
                                                           remove_docstr = True
                                                           )
        
        return contexts

    def parse_structs_names(self, query: str) -> List[str]:

        classes = []
        functions = []
        for file in self.files:
            with open(file,"r") as f:
                p = ast.parse(f.read())
                classes += [c for c in ast.walk(p) if isinstance(c,ast.ClassDef)]
                functions += [c for c in ast.walk(p) if isinstance(c,ast.FunctionDef)]
        
        if len(classes) == 0 and len(functions) == 0:
            return None
        
        classes = list(set(classes))
        functions = list(set(functions))
        all = classes + functions

        contexts = []
        for struct in all:
            if struct.name in query:
                contexts.append(astunparse.unparse(struct))
        
        return contexts


    def parse(self, query: str) -> List[str]:

        contexts = self.parse_file_names(query)
        contexts += self.parse_structs_names(query)
        if len(contexts) == 0:
            return None
        return contexts