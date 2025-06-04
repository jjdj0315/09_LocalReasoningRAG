# utils/graph.py

from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils.state import RAGState
from utils.model import (
    reasoning_llm,
    answer_llm,
)  # answer_llm이 streaming=True로 설정되어 있어야 함
import streamlit as st  # Streamlit 세션 상태에 접근하기 위해 추가 (retriever를 가져올 때 사용)


classify_llm_chain = (
    ChatPromptTemplate.from_template(
        """사용자의 질문이 문서 검색이 필요한 질문인지, 
        아니면 일반적인 상식이나 대화로 답변할 수 있는 질문인지 판단해주세요.
        문서 검색이 필요하다고 판단되면 'retrieve', 일반적인 답변이면 'generate'라고만 정확히 응답해주세요.

        사용자 질문: {query}

        판단:
        """
    )
    | reasoning_llm  # reasoning_llm을 사용
    | StrOutputParser()
)


# 1. 질문 분류 함수
def classify_node(state: RAGState):
    """질문을 분류하여 처리 모드를 결정합니다."""
    query = state["query"]
    print(f"===== 노드 시작: classify ({query}) =====")  # <--- 상태 출력 추가

    classification = classify_llm_chain.invoke({"query": query})
    mode = classification.strip().lower()

    # 정확히 'retrieve' 또는 'generate'만 반환하도록 보정
    if "retrieve" in mode:
        return {"mode": "retrieve"}
    else:
        return {"mode": "generate"}


# 2. 문서 검색 노드 (Retrieval)
def retrieve(state: RAGState):
    """주어진 질문에 대한 관련 문서를 검색합니다."""
    query = state["query"]
    print(f"===== 노드 시작: retrieve ({query}) =====")  # <--- 상태 출력 추가

    # Streamlit 세션 상태에서 retriever 가져오기
    retriever = st.session_state.get("compression_retriever")
    if not retriever:
        print(
            "오류: compression_retriever가 세션에 없습니다. PDF를 먼저 업로드해주세요."
        )
        return {"documents": []}  # 문서가 없으면 빈 리스트 반환

    documents = retriever.invoke(query)
    # print(f"===== 문서 검색 완료: {len(documents)}개 문서 =====") # 상세 로깅
    return {"documents": documents}


# 3. 추론 노드 (Reasoning LLM)
def reasoning(state: RAGState):
    """검색된 문서와 질문을 기반으로 추론 과정을 생성합니다."""
    query = state["query"]
    documents = state.get("documents", [])  # 안전하게 접근
    print(f"===== 노드 시작: reasoning ({query}) =====")  # <--- 상태 출력 추가

    # 문서 내용 추출
    context = "\n\n".join([doc.page_content for doc in documents])

    reasoning_prompt = ChatPromptTemplate.from_template(
        """주어진 문서를 활용하여 사용자의 질문에 가장 적절한 답변을 작성해주세요.
        문서가 없다면 일반적인 지식으로 답변을 시도하세요.

        질문: {query}

        문서 내용:
        {context}


        상세 추론:"""
    )

    reasoning_chain = reasoning_prompt | reasoning_llm | StrOutputParser()

    thinking = reasoning_chain.invoke({"query": query, "context": context})
    # print("===== 추론 완료 =====") # 상세 로깅
    return {"thinking": thinking}


# 4. 답변 생성 노드 (Answer LLM) - 스트리밍 핵심 수정 부분
def generate(state: RAGState):
    """문서와 추론 과정을 기반으로 최종 답변을 생성합니다."""

    query = state["query"]
    thinking = state.get("thinking", "")
    documents = state.get("documents", [])
    print(f"===== 노드 시작: generate ({query}) =====")  # <--- 상태 출력 추가

    # 문서 내용 추출
    context = "\n\n".join([doc.page_content for doc in documents])

    # 최종 답변 생성을 위한 프롬프트
    answer_prompt = ChatPromptTemplate.from_template(
        """사용자 질문에 한글로 답변하세요. 제공된 문서와 추론 과정이 있다면, 최대한 활용하세요.
        문서나 추론 과정이 부족하더라도 사용자 질문에 자연스럽게 답변을 시도하세요.

        질문: {query}

        추론 과정: {thinking}

        문서 내용:
        {context}

        답변:"""
    )

    answer_chain = answer_prompt | answer_llm | StrOutputParser()

    # 여기서 스트리밍을 직접 처리하고, 이를 yield를 통해 LangGraph 상태에 점진적으로 추가합니다.
    current_answer_content = ""
    for chunk in answer_chain.stream(
        {"query": query, "context": context, "thinking": thinking}
    ):
        if chunk:  # chunk가 빈 문자열이 아닐 경우만 처리
            current_answer_content += chunk
            # yield를 사용하여 'answer' 필드의 점진적인 업데이트를 LangGraph에 알립니다.
            # 이 yield가 handler.py의 stream_handler로 전달되어 Streamlit UI를 업데이트합니다.
            yield {"answer": current_answer_content}

    # 모든 청크가 완료되면 최종 상태를 한 번 더 반환합니다.
    # print("===== 답변 생성 완료 =====") # 상세 로깅
    return {"answer": current_answer_content}


# 조건부 라우팅 함수
def route_by_mode(state: RAGState) -> Literal["retrieve", "generate"]:
    """분류된 모드에 따라 다음 노드를 결정합니다."""
    # print(f"===== 라우팅: {state['mode']} =====") # 상세 로깅
    return state["mode"]
