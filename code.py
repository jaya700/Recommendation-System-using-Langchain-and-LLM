!pip install chromadb langchain openai tiktoken

import pandas as pd
import tiktoken
import os
import openai

from openai.embeddings_utils import get_embedding

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader

anime = pd. read_csv('/kaggle/input/anime-recommendation-database-2020/anime_with_synopsis.csv')
anime.head()

#Remove NA's
anime = anime.dropna()

#Combine title, synopsis, and Genre
anime['combined_info'] = anime.apply(lambda row: f"Title: {row['Name']}. Overview: {row['sypnopsis']} Genres: {row['Genres']}", axis=1)
anime['combined_info'][0]

#Save processed dataset - combined_info for Langchain
anime[['combined_info']].to_csv('anime_updated.csv', index=False)

pd.read_csv('/kaggle/working/anime_updated.csv')

api_key = 'OPEN AI API KEY' #Put your open ai api key

#data loader
loader = CSVLoader(file_path="/kaggle/working/anime_updated.csv")
data = loader.load()

#data transformers
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

#embeddings model, this can be a local LLM as well
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
llm = OpenAI(openai_api_key=api_key)

#Vector DB
docsearch = Chroma.from_documents(texts, embeddings)

query = "I'm looking for an animated action movie. What could you suggest to me?"
docs = docsearch.similarity_search(query, k=1)
docs

import os

os.environ['OPENAI_API_KEY'] = api_key

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
qa = RetrievalQA.from_chain_type(llm,
                                 chain_type="stuff", 
                                 retriever=docsearch.as_retriever(), 
                                 return_source_documents=True)
query = "I'm looking for an action anime. What could you suggest to me?"
result = qa({"query": query})
result['result']

result['source_documents'][0]

from langchain.prompts import PromptTemplate

template = """You are a movie recommender system that help users to find anime that match their preferences. 
Use the following pieces of context to answer the question at the end. 
For each question, suggest three anime, with a short description of the plot and the reason why the user migth like it.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Your response:"""


PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0) 
qa = RetrievalQA.from_chain_type(llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

query = "I'm looking for an action anime with animals, any suggestions?"
result = qa({'query':query})
print(result['result'])

from langchain.prompts import PromptTemplate

template_prefix = """You are a movie recommender system that help users to find anime that match their preferences. 
Use the following pieces of context to answer the question at the end. 
For each question, take into account the context and the personal information provided by the user.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}"""

user_info = """This is what we know about the user, and you can use this information to better tune your research:
Age: {age}
Gender: {gender}"""

template_suffix= """Question: {question}
Your response:"""

user_info = user_info.format(age = 18, gender = 'female')

COMBINED_PROMPT = template_prefix +'\n'+ user_info +'\n'+ template_suffix
print(COMBINED_PROMPT)

PROMPT = PromptTemplate(
    template=COMBINED_PROMPT, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

query = "I'm looking for an action anime with animals, any suggestions?"
result = qa({'query':query})
print(result['result'])

result['source_documents']
