from langchain_ollama import ChatOllama

reasoning_llm = ChatOllama(
    model="deepseek-r1:7b",
    stop=["</think>"]
)

answer_llm = ChatOllama(
    model="exaone3.5",
    temperature=0,
    )