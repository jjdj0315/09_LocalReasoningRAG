from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

reasoning_llm = ChatOllama(model="deepseek-r1:7b", stop=["</think>"])

answer_llm = ChatOllama(
    model="exaone3.5",
    temperature=0,
)


embeddings = OllamaEmbeddings(
    model="bge-m3:latest",
)
