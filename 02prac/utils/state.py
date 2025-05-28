from typing import Annotated, List, TypedDict, Literal
from langgraph.graph.message import add_messages
from langchain_core.documents import Document

# RAG 상태 정의
class RAGState(TypedDict):
    """RAG 시스템의 상태를 정의합니다."""
    query: str  # 사용자 질의
    thinking: str  # reasoning_llm이 생성한 사고 과정
    documents: List[Document]  # 검색된 문서
    answer: str  # 최종 답변
    messages: Annotated[List, add_messages]
    mode: str