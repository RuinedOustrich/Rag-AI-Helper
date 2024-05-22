from langchain_community.llms import LlamaCpp
from pipeline.databasebuilder import DataBaseBuilder
from embeddings import HuggingFaceEmbeddings
from config import load_config
from prompts import PromptTemplate
from embeddings.base import Embeddings
from textsplitters.code import EXT_TO_LANG
from parsers import DirectoryParser, UrlParser
from transformers import AutoTokenizer
import torch
import os

class Agent:
    def __init__(self, CONFIG_PATH: str, ROOT_DIR: str) -> None:

        """
        Initialize the Agent with the given configuration file path and root directory.

        Parameters:
            CONFIG_PATH (str): The path to the configuration file.
            ROOT_DIR (str): The root directory of the Agent.

        Returns:
            None
        """

        self.ROOT_DIR = ROOT_DIR
        self.config = load_config(CONFIG_PATH)

        self.language = EXT_TO_LANG[self.config['repo_database']['extension']]

        self.dir_parser = DirectoryParser(
                                        extension = self.config['repo_database']['extension'],
                                        path = "./"
                                        )
        
        self.url_parser = UrlParser( 
                                    extension=self.config['repo_database']['extension']
                                    )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.embedding_function = self.define_embedding(device)
        
        if device != "cuda":
            self.config['repo_database']["database_type"] = "table"

        self.external_retriever = None
        self.repo_retrieve = None

        if self.config["repo_database"]["database_path"] != "None":
            self.repo_retriever = DataBaseBuilder(
                                                    **self.config['repo_database'],
                                                    embedding_model=self.embedding_function
                                                ).run()

        if self.config["external_database"]["database_path"] != "None":
            self.external_retriever = DataBaseBuilder.load_vectorstore(
                                                                        **self.config['external_database'], 
                                                                        embedding_model=self.embedding_function
                                                                    )

        model_path = os.path.join(self.ROOT_DIR, *self.config['agent']['model_path'].split(','))

        self.llm = LlamaCpp(
                            model_path=model_path,
                            **self.config['llm']
                            )

        tokenizer_path = os.path.join(self.ROOT_DIR, *self.config['agent']['tokenizer_path'].split(','))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast = True)

    def define_embedding(self, device) -> Embeddings:
        """
        Define the embedding based on the configuration settings.
        If the database type is not a CSV, create the embedding function using the specified model and device.
        If the embedding type is HuggingFaceEmbedding, use HuggingFaceEmbeddings with the provided model and device settings.
        Otherwise, raise a NotImplementedError.
        Return the created embedding function.
        """
        #if self.config['database']['database_type'] != 'csv':
        if self.config['embedding']['embedding_type'] == 'HuggingFaceEmbedding':
                embedding_function = HuggingFaceEmbeddings(model_name=self.config['embedding']['embedding_name_or_path'],
                                                           model_kwargs={"device": device},
                                                           encode_kwargs={"device": device, 'batch_size': 64}
                                                           )
        else:
                raise NotImplementedError

        return embedding_function

        #return None

    def trim_context(self, context: str, query: str, prompt: str) -> str:
        """
        Trim the context to the specified maximum length.
        """
        max_length = self.config['llm']['n_ctx']
        context_ids = self.tokenizer.encode(context)
        query_ids = self.tokenizer.encode(query)
        prompt_ids = self.tokenizer.encode(prompt)

        if len(context_ids) + len(query_ids) + len(prompt_ids) > max_length:
            context_ids = context_ids[:max_length - len(query_ids) - len(prompt_ids)]
            if len(context_ids) + len(query_ids) + len(prompt_ids) > max_length:
                query_ids = query_ids[:max_length - len(context_ids) - len(prompt_ids)]
        else:
            return context, query

        return self.tokenizer.decode(context_ids), self.tokenizer.decode(query_ids)

    def build_prompt(self, query: str) -> str:

        """
        Builds a prompt for the agent to generate an answer to a given query.

        Args:
            query (str): The query for which the prompt is being built.

        Returns:
            str: The prompt string.
        """
        max_k = self.config["agent"]["max_k"]
        
        contexts = []
        dir_out = self.dir_parser.parse(query)

        if dir_out is not None:
            contexts += dir_out

        elif dir_out is None and self.repo_retriever is not None:
            
            contexts = self.repo_retriever._get_relevant_documents(query)

        elif len(contexts) < max_k or dir_out is None and self.external_retriever is not None:
            contexts += self.external_retriever._get_relevant_documents(query)

        elif len(contexts) < max_k:
            contexts += self.url_parser(query, k = max_k-len(contexts))


        contexts = '\n\n'.join(contexts)
        
        if query.startswith("#"):
            query = query.replace("#", "")
            template = "You are an {language} code assistant for solving programming problems. You are given several context, feel free to use them. Write {language} function or class to answer the Query.\nContexts: {context}\nQuery: {query} \nAnswer:"

        else:
            template = "You are an assistant for {language} code completion. Use the following pieces of retrieved context to complete the query.\nContext: {context}\n Query: {query}"

        prompt_template = PromptTemplate(template=template, input_variables=["language", "context", "query"])

        if self.config["agent"]["trim_context"]:
            contexts, query = self.trim_context(contexts, query, template)

        return prompt_template.format(language=self.language,
                                      context=contexts,
                                      query=query)

    def __call__(self, query: str) -> str:
        """
        Calls the llm with the given query.

        Args:
            query (str): The query to be passed to the function.

        Returns:
            The response generated by the llm.
        """

        prompt = self.build_prompt(query)
        response = self.llm.invoke(prompt)

        return response

    def run(self) -> None:
        """
        Runs the agent in an infinite loop, prompting the user for input and generating a response based on the input.
        """
        while True:
            
            if self.config["repo_database"]["database_path"] != "None":
                self.repo_retriever = DataBaseBuilder(
                                                    **self.config['repo_database'],
                                                    embedding_model=self.embedding_function
                                                ).run()
            query = input("User: ")
            response = self(query)
            print("Assistant: ", response.strip())