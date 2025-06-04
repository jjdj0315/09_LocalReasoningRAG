from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

# reasoning_llm은 스트리밍 필요 없음 (invoke 사용)
reasoning_llm = ChatOllama(model="deepseek-r1:7b", stop=["</think>"])

# answer_llm은 스트리밍 활성화
answer_llm = ChatOllama(
    model="exaone3.5",  # 또는 사용하시는 LLM 모델
    temperature=0,
    streaming=True,  # <--- 이 부분이 반드시 True여야 합니다.
)

embeddings = OllamaEmbeddings(
    model="bge-m3:latest",  # 또는 사용하시는 임베딩 모델
)
