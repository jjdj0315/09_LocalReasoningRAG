# utils/model.py
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

# reasoning_llm 스트리밍 활성화 확인
reasoning_llm = ChatOllama(
    model="deepseek-r1:7b",  # 실제 사용하시는 모델
    stop=["</think>"],
    streaming=True,  # <--- 여기가 True여야 함
)

# answer_llm 스트리밍 활성화 확인
answer_llm = ChatOllama(
    model="exaone3.5",  # 실제 사용하시는 모델
    temperature=0,
    streaming=True,  # <--- 여기가 True여야 함
)

embeddings = OllamaEmbeddings(
    model="bge-m3:latest",  # 실제 사용하시는 임베딩 모델
)
