from __future__ import annotations

from files.file_utils import extract_content_from_files, get_extension_files
from textsplitters import CodeSplitter, TextSplitter
from vectorstores import Chroma, FAISS, TableVectorStore, VectorStoreRetriever
from embeddings.base import Embeddings

import pandas as pd
import os
import uuid
from vectorstores.base import VectorStore
from typing import (List,
                    Tuple,
                    Optional,
                    Dict,
                    Literal)

class DataBaseBuilder():
    def __init__(
                self, 
                embedding_model: Embeddings,
                path: str = "./",
                extension: str = 'py',
                splitter: str = 'code',
                remove_docstr: bool = False,
                chunk_size: int = 1000,
                chunk_overlap: int = 0,
                database_path: str = './database',
                database_type: Literal['chroma', 'faiss', 'csv'] = 'csv',
                retriever_kwargs: Optional[Dict[str]] = None
                ):

        """
        Initialize the object with the given path and optional parameters.
        Args:
            path (str): The path to the directory containing the code files.
            extension (str, optional): The file extension to filter the files. Defaults to 'go'.
            remove_docstr (bool, optional): Whether to remove docstrings from the code files. Defaults to False.
            chunk_size (int, optional): The size of each chunk for text extraction. Defaults to 1000.
            chunk_overlap (int, optional): The overlap size between chunks for text extraction. Defaults to 0.
            database_path (str, optional): The path to the database to store the extracted content. Defaults to './database'.
            database_type (str, optional): The type of database to use for storing the extracted content. Defaults to 'csv'.
            embedding_model (str, optional): The name or path of the embedding model to use for text representation. Defaults to "thenlper/gte-base".
        """

        self.path = path
        self.extension = extension
        self.splitter = splitter
        self.remove_docstr = remove_docstr
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.database_path = database_path
        self.database_type = database_type
        self.embedding_model = embedding_model
        self.retriever_kwargs = retriever_kwargs
        
        if database_type == 'table':
            self.embedding_model = embedding_model

    def split_data(self, file_list: Optional[List[str]] = None) -> Tuple[List[str], List[str], Dict[str, float, str], List[str]]:
        """
        Split the data extracted from files into chunks using a specified CodeTextSplitter.

        Parameters:
            file_list (Optional[List[str]]): A list of file paths to process. If not provided, all files in the current directory are processed.

        Returns:
            Tuple[List[str], List[str], pd.DataFrame]: A tuple containing the splitted data, corresponding indices and a dataframe of the processed files.
        """
        print("splitting")#dbdjsbfjdsbf
        data, files = extract_content_from_files(
                                                self.path,
                                                extension=self.extension,
                                                remove_docstr=self.remove_docstr,
                                                file_list=file_list
                                                )

        if self.splitter == 'code':
            splitter = CodeSplitter(
                                    extension=self.extension,
                                    chunk_size=self.chunk_size,
                                    chunk_overlap=self.chunk_overlap
                                    )
        elif self.splitter == 'text':

            splitter = TextSplitter(
                                    chunk_size=self.chunk_size,
                                    chunk_overlap=self.chunk_overlap,
                                    )
            
        d = []
        splitted_data = []
        idxs = []
        for idx, file in enumerate(files):
            splitted = splitter.split_text(data[idx])
            idx = [str(uuid.uuid4()) for _ in splitted]
            splitted_data += splitted
            idxs += idx
            time = os.stat(file).st_mtime
            d.append({
                "file": file,
                "time": time,
                "idxs": idx
            })
        return splitted_data, idxs, d
    
    def build_database(self) -> VectorStoreRetriever:
        """
        A function that builds a database based on the specified database type.
        If the database type is 'csv', it saves the data to a CSV file.
        If the database type is 'chroma' or 'faiss', it creates a vector store using the provided embeddings.
        Args:
            self: The object instance.
        Returns:
            VectorStore
        """
        splitted_data, idxs, d = self.split_data()

        print("Building database...")
        os.mkdir(self.database_path) if not os.path.exists(self.database_path) else None
        if self.database_type == 'chroma':
            vectorstore = Chroma.from_texts(splitted_data,
                                            ids=idxs,
                                            embedding=self.embedding_model,
                                            persist_directory=self.database_path,)

        elif self.database_type == 'faiss':
            vectorstore = FAISS.from_texts(splitted_data,
                                           ids=idxs,
                                           embedding=self.embedding_model)

            vectorstore.save_local(self.database_path)

        elif self.database_type == 'table':
            vectorstore = TableVectorStore.from_texts(texts=splitted_data, ids=idxs)
            vectorstore.save_local(self.database_path)
        else:
            raise ValueError(f"Database type {self.database_type} not supported.")

        os.mkdir("./tmp") if not os.path.exists("./tmp") else None
        d = pd.DataFrame(d)
        d.to_feather("./tmp/tmp.feather")

        print(
            f"Database built successfully at {self.database_path}"
        )

        return vectorstore.as_retriever(**self.retriever_kwargs)

    def check_file_mtime(self) -> bool:
        """
        Check the modification time of files in the specified path with the given extension.

        Parameters:
            self (obj): The object instance.

        Returns:
            bool: True if the modification times match, False otherwise.
        """
        file_list = get_extension_files(path=self.path, extension=self.extension)
        d = []
        for file_ in file_list:
            time = os.stat(file_).st_mtime
            d.append({
                "file": file_,
                "time": time,
            })
        new_tmp = pd.DataFrame(d)
        tmp = pd.read_feather("./tmp/tmp.feather")

        if new_tmp['time'].equals(tmp['time']):
            self.need_to_update = None
            return True
        else:
            self.need_to_update = tmp[~tmp['time'].isin(new_tmp['time'])].dropna()
            return False

    def update_database(self) -> VectorStoreRetriever:
        """
        Updates the database based on the specified database type.
        Returns:
        VectorStore: The updated database.
        """
        files, idxs = self.need_to_update['file'].values, self.need_to_update['idxs'].values
        tmp = pd.read_feather("./tmp/tmp.feather")
        if self.database_type == 'table':

            idxs = [x for xs in idxs for x in xs]

            vectorstore = TableVectorStore(persistent_path=self.database_path)
            vectorstore.delete_by_index(idxs)

            splitted_data, ids, d = self.split_data(file_list=files)

            vectorstore.add_texts(data=splitted_data, ids=ids)
            vectorstore.save_local(self.database_path)

        elif self.database_type == 'chroma':

            idxs = [x for xs in idxs for x in xs]
            vectorstore = Chroma(persist_directory=self.database_path, embedding_function=self.embedding_model)
            vectorstore.delete(idxs)

            splitted_data, ids, d = self.split_data(file_list=files)
            vectorstore.add_texts(splitted_data, ids=ids)
            vectorstore.persist()

        elif self.database_type == 'faiss':

            idxs = [x for xs in idxs for x in xs]
            vectorstore = FAISS.load_local(self.database_path, embeddings=self.embedding_model, allow_dangerous_deserialization=True)
            vectorstore.delete(idxs)

            splitted_data, ids, d = self.split_data(file_list=files)
            vectorstore.add_texts(splitted_data, ids=ids)
            vectorstore.save_local(self.database_path)
        else:
            raise ValueError(f"Unsupported database type: {self.database_type}")

        d = pd.DataFrame(d)
        for file in files:
            tmp.loc[tmp['file'] == file, 'time'] = os.path.getmtime(file)
            tmp.loc[tmp['file'] == file, 'idxs'] = d.loc[d['file'] == file, 'idxs'].values
        tmp.to_feather("./tmp/tmp.feather")

        return vectorstore.as_retriever(**self.retriever_kwargs)
    @classmethod
    def load_vectorstore(
                        self,
                        database_path: str,
                        embedding_model: Embeddings,
                        database_type: Literal['chroma', 'faiss', 'csv'],
                        retriever_kwargs: Dict, 
                        ) -> VectorStoreRetriever:
        """
        Loads a vector store based on the specified database type.
        Returns:
            VectorStore: The loaded vector store.
        Raises:
            ValueError: If the database type is not supported.
        """
        if database_type == 'table':
            return TableVectorStore(database_path).as_retriever(**retriever_kwargs)

        elif database_type == 'faiss':

            return FAISS.load_local(database_path,
                                    embeddings=embedding_model,
                                    allow_dangerous_deserialization=True).as_retriever(**retriever_kwargs)

        elif database_type == 'chroma':

            return Chroma(persist_directory=database_path,
                          embedding_function=embedding_model).as_retriever(**retriever_kwargs)

        else:
            raise ValueError(f"Unsupported database type: {database_type}")
        
    def run(self) -> VectorStoreRetriever:
        """
        Build a database using the provided engine at the specified database path.

        Args:
            engine (DataBaseBuilder): The engine used to build the database.
            database_path (str): The path where the database will be stored.

        Returns:
            None
        """
        
        if self.check_database_exists() is False:
            store = self.build_database()
        else:
            if self.check_file_mtime() is False:
                store = self.update_database()
            else:
                store = self.load_vectorstore(
                                            database_path=self.database_path,
                                            embedding_model=self.embedding_model,
                                            database_type=self.database_type,
                                            retriever_kwargs=self.retriever_kwargs
                                            )

        return store

    def check_database_exists(self) -> bool:
        """
        Check if the database exists at the specified path.

        Returns:
            bool: True if the database exists, False otherwise.
        """
        if not os.path.exists(self.database_path):
            return False
        
        db_to_format = {'table': 'feather', 'faiss': 'faiss', 'chroma': 'sqlite3'}
        
        for file in os.listdir(self.database_path):
            if file.endswith(db_to_format[self.database_type]):
                return True
        return False