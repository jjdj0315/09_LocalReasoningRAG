# utils/graph.py (이전 수정 내용 그대로)

from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils.state import RAGState
from utils.model import reasoning_llm, answer_llm
import streamlit as st


classify_llm_chain = (
    ChatPromptTemplate.from_template(
        """사용자의 질문이 문서 검색이 필요한 질문인지, 
        아니면 일반적인 상식이나 대화로 답변할 수 있는 질문인지 판단해주세요.
        문서 검색이 필요하다고 판단되면 'retrieve', 일반적인 답변이면 'generate'라고만 정확히 응답해주세요.

        사용자 질문: {query}

        판단:
        """
    )
    | reasoning_llm
    | StrOutputParser()
)


def classify_node(state: RAGState):
    query = state["query"]
    print(f"===== 노드 시작: classify ({query}) =====")
    classification = classify_llm_chain.invoke({"query": query})
    mode = classification.strip().lower()
    if "retrieve" in mode:
        return {"mode": "retrieve"}
    else:
        return {"mode": "generate"}


def retrieve(state: RAGState):
    query = state["query"]
    print(f"===== 노드 시작: retrieve ({query}) =====")
    retriever = st.session_state.get("compression_retriever")
    if not retriever:
        print(
            "오류: compression_retriever가 세션에 없습니다. PDF를 먼저 업로드해주세요."
        )
        return {"documents": []}
    documents = retriever.invoke(query)
    return {"documents": documents}


def reasoning(state: RAGState):
    query = state["query"]
    documents = state.get("documents", [])
    print(f"===== 노드 시작: reasoning ({query}) =====")

    if not documents:
        context = "제공된 문서 없음."
    else:
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

    current_thinking_content = ""
    for chunk in reasoning_chain.stream({"query": query, "context": context}):
        if chunk:
            current_thinking_content += chunk
            yield {"thinking": current_thinking_content}

    return {"thinking": current_thinking_content}


def generate(state: RAGState):
    query = state["query"]
    thinking = state.get("thinking", "")
    documents = state.get("documents", [])
    print(f"===== 노드 시작: generate ({query}) =====")

    context = "\n\n".join([doc.page_content for doc in documents])

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

    current_answer_content = ""
    for chunk in answer_chain.stream(
        {"query": query, "context": context, "thinking": thinking}
    ):
        if chunk:
            current_answer_content += chunk
            yield {"answer": current_answer_content}

    return {"answer": current_answer_content}


def route_by_mode(state: RAGState) -> Literal["retrieve", "generate"]:
    return state["mode"]
