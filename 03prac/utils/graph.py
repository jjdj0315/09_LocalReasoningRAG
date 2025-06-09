from typing import Literal
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage # <-- AIMessage, BaseMessage 추가 임포트
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # <-- MessagesPlaceholder 추가 임포트
from langgraph.graph import START, StateGraph, END
from utils.state import RAGState
import streamlit as st

# creat_compression_retriever는 여기서 직접 사용하지 않으므로 임포트 삭제
# from utils.creat_compression_retriever import creat_compression_retriever
from utils.model import reasoning_llm, answer_llm


# 분류를 위한 LLM 체인 정의 (reasoning_llm 재활용)
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
    """질문을 분류하여 처리 모드를 결정합니다."""
    query = state["query"]
    print(f"=====질문 분류 시작: {query}=====")

    classification = classify_llm_chain.invoke({"query": query})
    
    if "retrieve" in classification.lower():
        print("=====검색 모드 진입=====")
        return {"mode": "retrieve"}
    else:
        print("=====생성 모드 진입=====")
        return {"mode": "generate"}

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
    documents = compression_retriever.invoke(query)

    for doc in documents:
        print(doc.page_content)
        print("-" * 100)
    print("=====문서 검색 완료=====")
    return {"documents": documents}

def reasoning(state: RAGState):
    """쿼리를 분석하여 사고 과정을 생성합니다."""
    query = state["query"]
    documents = state["documents"]
    chat_history = state.get("chat_history", []) # <-- chat_history 추가

    if not documents: # 문서가 없을 경우 추론 과정 스킵 또는 다르게 처리
        print("=====문서 없음. 추론 스킵=====")
        return {"thinking": "제공된 문서가 없어 심층 추론이 어렵습니다."}

    context = "\n\n".join([doc.page_content for doc in documents])

    reasoning_prompt = ChatPromptTemplate.from_messages( # <-- MessagesPlaceholder를 사용하기 위해 from_messages로 변경
        [
            ("system", "주어진 문서를 활용하여 사용자의 질문에 가장 적절한 답변을 작성해주세요. 문서가 없다면 일반적인 지식으로 답변을 시도하세요."),
            MessagesPlaceholder(variable_name="chat_history"), # <-- 이전 대화 기록 추가
            ("human", "질문: {query}"),
            ("system", "문서 내용:\n{context}"),
            ("system", "상세 추론:"),
        ]
    )

    reasoning_chain = reasoning_prompt | reasoning_llm | StrOutputParser()

    print("=====추론 시작=====")
    thinking = reasoning_chain.invoke({"query": query, "context": context, "chat_history": chat_history}) # <-- chat_history 전달
    print("=====추론 완료=====")
    return {"thinking": thinking}


# 3. 답변 생성 노드 (Answer LLM)
def generate(state: RAGState):
    """문서와 추론 과정을 기반으로 최종 답변을 생성합니다."""

    query = state["query"]
    thinking = state.get("thinking", "")
    documents = state.get("documents", [])
    chat_history = state.get("chat_history", []) # <-- chat_history 추가

    context = "\n\n".join([doc.page_content for doc in documents])

    # 최종 답변 생성을 위한 프롬프트
    answer_prompt = ChatPromptTemplate.from_messages( # <-- MessagesPlaceholder를 사용하기 위해 from_messages로 변경
        [
            ("system", "사용자 질문에 한글로 답변하세요. 제공된 문서와 추론 과정이 있다면, 최대한 활용하세요. 문서나 추론 과정이 부족하더라도 사용자 질문에 자연스럽게 답변을 시도하세요."),
            MessagesPlaceholder(variable_name="chat_history"), # <-- 이전 대화 기록 추가
            ("human", "질문:\n{query}"),
            ("system", "추론 과정:\n{thinking}"),
            ("system", "문서 내용:\n{context}"),
            ("system", "답변:"),
        ]
    )
    print("=====답변 생성 시작=====")

    answer_chain = answer_prompt | answer_llm | StrOutputParser()

    answer = answer_chain.invoke({
        "query": query,
        "thinking": thinking,
        "context": context,
        "chat_history": chat_history # <-- chat_history 전달
    })
    print("=====답변 생성 완료=====")
    return {"answer": answer}