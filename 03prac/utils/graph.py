from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import START, StateGraph, END

from utils.state import RAGState
from utils.creat_compression_retriever import creat_compression_retriever
from utils.model import reasoning_llm, answer_llm

# 1. 질문 분류 함수 - 중요: 여기서는 상태를 업데이트하는 노드 함수
def classify_node(state: RAGState):
    """질문을 분류하여 처리 모드를 결정합니다."""
    query = state["query"]

    if "Docling" in query:
        print("=====검색 시작=====")
        return {"mode": "retrieve"}
    else:
        print("=====생성 시작=====")
        return {"mode": "generate"}


# 2. 라우팅 함수 - 중요: 이 함수는 조건부 엣지에서 사용하며 문자열 반환
def route_by_mode(state: RAGState) -> Literal["retrieve", "generate"]:
    """모드에 따라 다음 단계를 결정합니다."""
    return state["mode"]

def retrieve(state: RAGState):
    """질의를 기반으로 관련 문서를 검색합니다."""
    query = state["query"]
    print("=====검색 시작=====")
    documents = creat_compression_retriever.invoke(query)
    for doc in documents:
        print(doc.page_content)
        print("-"*100)
    print("=====검색 완료=====")
    return {"documents": documents}

def reasoning(state: RAGState):
    """쿼리를 분석하여 사고 과정을 생성합니다."""
    query = state["query"]
    documents = state["documents"]

    context = "\n\n".join([doc.page_content for doc in documents])

    reasoning_prompt = ChatPromptTemplate.from_template(
        """주어진 문서를 활용하여 사용자의 질문에 가장 적절한 답변을 작성해주세요.

        질문: {query}

        문서 내용:
        {context}


        상세 추론:"""
    )

    reasoning_chain = reasoning_prompt | reasoning_llm | StrOutputParser()

    print("=====추론 시작=====")
    thinking = reasoning_chain.invoke({"query": query, "context": context})

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
    # 메시지에 답변 추가
    return {
        "answer": answer,
        "messages": [HumanMessage(content=answer)]
    }