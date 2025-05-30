from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import START, StateGraph, END

from utils.state import RAGState
# from utils.creat_compression_retriever import creat_compression_retriever # 직접 임포트 대신 세션에서 가져옴
from utils.model import reasoning_llm, answer_llm
import streamlit as st # Streamlit 세션 상태에 접근하기 위해 추가

classify_llm_chain = (
    ChatPromptTemplate.from_template(
        """사용자의 질문이 문서 검색이 필요한 질문인지, 
        아니면 일반적인 상식이나 대화로 답변할 수 있는 질문인지 판단해주세요.
        문서 검색이 필요하다고 판단되면 'retrieve', 일반적인 답변이면 'generate'라고만 정확히 응답해주세요.

        사용자 질문: {query}

        판단:
        """
    )
    | reasoning_llm # reasoning_llm을 사용
    | StrOutputParser()
)

# 1. 질문 분류 함수 - 중요: 여기서는 상태를 업데이트하는 노드 함수
def classify_node(state: RAGState):
    """질문을 분류하여 처리 모드를 결정합니다."""
    query = state["query"]
    print(f"=====질문 분류 시작: {query}=====")

    # 정의된 classify_llm_chain을 사용하여 질문 의도를 분류합니다.
    classification = classify_llm_chain.invoke({"query": query})
    
    # LLM의 답변을 소문자로 변환하여 'retrieve' 포함 여부로 모드를 결정합니다.
    if "retrieve" in classification.lower():
        print("=====검색 모드 진입=====")
        return {"mode": "retrieve"}
    else: # 'retrieve'가 아니면 모두 'generate'로 간주합니다.
        print("=====생성 모드 진입=====")
        return {"mode": "generate"}


# 2. 라우팅 함수 - 중요: 이 함수는 조건부 엣지에서 사용하며 문자열 반환
def route_by_mode(state: RAGState) -> Literal["retrieve", "generate"]:
    """모드에 따라 다음 단계를 결정합니다."""
    return state["mode"]



def retrieve(state: RAGState):
    """질의를 기반으로 관련 문서를 검색합니다."""
    query = state["query"]
    print("=====문서 검색 시작=====")

    if "compression_retriever" not in st.session_state:
        print("오류: compression_retriever가 세션에 없습니다. PDF를 먼저 업로드해주세요.")
        return {"documents": []} # 문서가 없으면 빈 리스트 반환

    compression_retriever = st.session_state["compression_retriever"]
    documents = compression_retriever.invoke(query) # 올바른 객체에 invoke 호출

    for doc in documents:
        print(doc.page_content)
        print("-" * 100)
    print("=====문서 검색 완료=====")
    return {"documents": documents}

def reasoning(state: RAGState):
    """쿼리를 분석하여 사고 과정을 생성합니다."""
    query = state["query"]
    documents = state["documents"]

    if not documents: # 문서가 없을 경우 추론 과정 스킵 또는 다르게 처리
        print("=====문서 없음. 추론 스킵=====")
        return {"thinking": "제공된 문서가 없어 심층 추론이 어렵습니다."}

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

    print("=====추론 시작=====")
    thinking = reasoning_chain.invoke({"query": query, "context": context})
    print("=====추론 완료=====")
    return {"thinking": thinking}


# 3. 답변 생성 노드 (Answer LLM)
def generate(state: RAGState):
    """문서와 추론 과정을 기반으로 최종 답변을 생성합니다."""

    query = state["query"]
    thinking = state.get("thinking", "")  # get 메서드로 안전하게 접근
    documents = state.get("documents", [])  # get 메서드로 안전하게 접근

    # 문서 내용 추출
    context = "\n\n".join([doc.page_content for doc in documents])

    # 최종 답변 생성을 위한 프롬프트
    answer_prompt = ChatPromptTemplate.from_template(
        """사용자 질문에 한글로 답변하세요. 제공된 문서와 추론 과정이 있다면, 최대한 활용하세요.
        문서나 추론 과정이 부족하더라도 사용자 질문에 자연스럽게 답변을 시도하세요.

        질문:
        {query}

        추론 과정:
        {thinking}

        문서 내용:
        {context}

        답변:"""
    )
    print("=====답변 생성 시작=====")

    answer_chain = answer_prompt | answer_llm | StrOutputParser()

    answer = answer_chain.invoke({
        "query": query,
        "thinking": thinking,
        "context": context
    })
    print("=====답변 생성 완료=====")
    # 메시지에 답변 추가는 app.py에서 담당하므로 여기서는 answer만 반환합니다.
    return {"answer": answer}