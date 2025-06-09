from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .state import RAGState
from .model import reasoning_llm

classify_llm_chain = ChatPromptTemplate.from_template(
    """
다음 사용자 질문이 외부 문서의 정보가 필요한지 판단하세요.

- 특정 문서, 보고서, 사용자 업로드 파일, 수치, 통계, 날짜 등 **문서 기반 정보가 필요한 경우**: 'retrieve'
- 일반 상식, 대화, 감정, 일상적인 질문처럼 **문서 없이도 대답 가능한 경우**: 'generate'

오직 아래 두 단어 중 하나만 출력하세요: retrieve 또는 generate  
그 외 문장은 절대 출력하지 마세요.

사용자 질문: {query}

판단:
    """
    | reasoning_llm
    | StrOutputParser
)


# 1. 질문 분류 함수 - 중요: 여기서는 상태를 업데이트하는 노드 함수
def classify_node(state: RAGState):
    """질문을 분류하여 처리모드를 결저합니다."""
    query = state["query"]
    print("질문 분류 시작 : {query}")

    # 정의된 classify_llm_chain을 사용하야 질문 의도 분류
    classification = classify_llm_chain.invoke({"query": query})

    print(f"====질문 분류 완료 : {classification}====")

    # 분류 결과에 따라 mode상태를 설정합니다.
    if "retrieve" in classification.lower():
        mode = "retreive"
    else:
        mode = "generate"

    print("모드 결정 {mode}")
    return {"mode": mode}


# 추론 노드
def reasoning(state: RAGState):
    """검색된 문서와 질문을 기반으로 추론 과정을 생성합니다."""
    query = state["query"]
    documents = state.get("documents", [])

    # 문서 내용을 추출하여 컨텍스트로 사용
    context = "\n\n".join([doc.page_context for doc in documents])

    reasoning_prompt = ChatPromptTemplate.from_template(
        """주어진 문서를 활용하여 사용자의 질문에 가장 적절한 답변을 작성해주세요.
        문서가 없다면 일반적인 지식으로 답변을 시도하세요.

        질문: {query}

        문서 내용:
        {context}


        상세 추론:"""
    )

    reasoning_chain = reasoning_prompt | reasoning_llm | StrOutputParser()

    print("추론 시작")

    thinking = reasoning_chain.invoke({"query": query, "context": context})
    print("===추론완료===")
    return {"thinking": thinking}
